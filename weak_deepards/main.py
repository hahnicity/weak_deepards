from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from weak_deepards.dataset import ApneaECGDataset, ARDSRawDataset
from weak_deepards.models.base.densenet import densenet18
from weak_deepards.models.base.resnet import resnet18, resnet34, resnet50
from weak_deepards.models.base.dey import DeyNet
from weak_deepards.models.base.urtnasan import UrtnasanNet
from weak_deepards.models.prm import peak_response_mapping
from weak_deepards.results import ModelCollection


class ARDSSystem(pl.LightningModule):
    def __init__(self, hparams, dataset, results_obj, fold_num):
        super().__init__()
        self.hparams = hparams
        self.fold_num = fold_num
        self.results = results_obj
        with open(hparams.config_file) as conf:
            config = yaml.safe_load(conf)
        self.input_units = hparams.input_units
        if hparams.base_net == 'resnet18':
            base = resnet18(initial_kernel_size=hparams.kernel_size, initial_stride=hparams.stride)
        elif hparams.base_net == 'densenet18':
            base = densenet18()
        self.model = peak_response_mapping(base, **config['model'])
        self.dataset = dataset
        if self.hparams.loss_func == 'ce':
            self.loss_func = F.cross_entropy
        elif self.hparams.loss_func == 'bce':
            self.loss_func = F.binary_cross_entropy_with_logits

    def perform_inference(self):
        val_loader = self.val_dataloader()
        # need to enable grad because pytorch lightning disables in validation step
        # It doesn't matter how much we try to explicitly enable grad, if we are under
        # a with.no_grad header no grads will ever be calculated
        with torch.enable_grad():
            # Get batch from loader
            for rand_batch in self.val_loader:
                break
            batch_data = rand_batch[2]
            rand_idx = np.random.randint(batch_data.shape[0])
            rand_seq = batch_data[rand_idx].float().reshape(
                (1, 1, self.hparams.input_units)).requires_grad_()
            # run an inference to get an intuition for the model
            self.model.eval()
            self.model.inference()
            # XXX The visual cues dont seem to spike above 3 or so with this application. Why is this?
            # why does this differ so much compared to img learning? Is there something that
            # I can do to boost response? Could it be due to ambuiguity in class response?
            # Could it be because we are not multiplying by height here and are only using
            # width?
            visual_cues = self.model(rand_seq, peak_threshold=self.hparams.pr_thresh)
            if visual_cues is None:
                print('No class response detected')
            else:
                class_names = ['Non-ARDS', 'ARDS']
                conf, crm, cpr, prm = visual_cues
                interp_scale = rand_seq.shape[-1] / crm.shape[-1]
                upsampled_crm = F.upsample(crm, scale_factor=interp_scale, mode='linear')
                upsampled_crm, cpr, prm = upsampled_crm.cpu().numpy(), cpr.cpu().numpy(), prm.cpu().numpy()
                rand_seq = rand_seq.detach().cpu().numpy()

                _, class_idx = torch.max(conf, dim=1)
                class_idx = int(class_idx.cpu())
                num_plots = 3 + len(prm)
                fig = plt.figure(figsize=(3 * 4, 3 * 4))

                ax_prm0 = fig.add_subplot(3, 1, 3)
                ax_prm1 = fig.add_subplot(3, 1, 2)

                ax1 = fig.add_subplot(3, 3, 1)
                ax2 = fig.add_subplot(3, 3, 2)
                ax3 = fig.add_subplot(3, 3, 3)

                ax1.plot(rand_seq.ravel())
                ax1.set_title('Sequence')
                ax1.axis('off')
                ax2.plot(upsampled_crm[0, class_idx])
                ax2.set_title('Class Response Map ("%s")' % class_names[class_idx])
                ax2.axis('off')
                ax3.scatter(np.arange(rand_seq.shape[-1]), rand_seq.ravel(), c=upsampled_crm[0, class_idx], s=rand_seq.shape[-1] / 255)
                ax3.plot(rand_seq.ravel())
                ax3.set_title('Sequence with CRM ("%s")' % class_names[class_idx])
                ax3.axis('off')

                ax_prm0.set_title('Peak Response Map ("%s")' % (class_names[0]))
                ax_prm1.set_title('Peak Response Map ("%s")' % (class_names[1]))
                ax_prm0.axis('off')
                ax_prm1.axis('off')
                for idx, (resp_map, peak) in enumerate(sorted(zip(prm, cpr), key=lambda v: v[-1][-1])):
                    if peak[1].item() == 0:
                        ax_prm0.plot(resp_map)
                    elif peak[1].item() == 1:
                        ax_prm1.plot(resp_map)

                plt.show()

            # reset back to training state
            self.model.train()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        idxs, pt, x, y = batch
        x = x.unsqueeze(1)
        x = Variable(x)
        y = Variable(y)
        y_hat = self.forward(x.float())
        # XXX make sure this is cross functional with bce and ce
        loss = self.loss_func(y_hat, y)
        board_logs = {'train_loss': loss}
        return {'loss': loss, 'log': board_logs}

    def validation_step(self, batch, batch_idx):
        idxs, pt, x, y = batch
        x = x.unsqueeze(1)
        y_hat = self.forward(x.float())
        # XXX make sure this is cross functional with bce and ce
        loss = self.loss_func(y_hat, y)
        # XXX should add accuracy here so we could add it to the early stopping step
        return {'val_loss': loss, 'pt': pt, 'y': y, 'pred': y_hat, 'idxs': idxs}

    def validation_epoch_end(self, outputs):
        if len(outputs) <= 5:
            return dict()

        val_loss_mean = 0
        for out in outputs:
            val_loss_mean += out['val_loss']

        y_val = self.val_loader.dataset.get_ground_truth_df()
        for dict_ in outputs:
            y_val.loc[dict_['idxs'].cpu(), 'pred'] = F.softmax(dict_['pred'])[:, 1].cpu().numpy()
        self.results.add_model(y_val, y_val.pred, self.fold_num, self.current_epoch)
        self.results.calc_epoch_stats(self.current_epoch)
        return {
            'progress_bar': {'val_loss': val_loss_mean / len(outputs)},
            'val_loss': val_loss_mean / len(outputs)
        }

    def test_step(self):
        pt, x, y = batch
        y = torch.stack(y, dim=1)
        y_hat = self.forward(x.float())
        loss = self.loss_func(y_hat, y.float())
        return {'test_loss': loss}

    def configure_optimizers(self):
        sgd = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(sgd, 1, gamma=.2)
        return [sgd], [scheduler]

    def train_dataloader(self):
        self.dataset.set_kfold_indexes_for_fold(self.fold_num)
        train_loader = DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        train_loader = self.train_dataloader()
        val = ARDSRawDataset.make_test_dataset_if_kfold(train_loader.dataset)
        self.val_loader = DataLoader(val, batch_size=self.hparams.batch_size)
        return self.val_loader


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--train-to-pickle', default='')
    parser.add_argument('--train-from-pickle', default='')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--cuda', action='store_true')
    # damnit. need to figure out how to do this with multi-gpu. the way ptl works
    # is that it splits the trainer class across 2 processes so you dont really
    # have a single results obj
    parser.add_argument('--n-gpus', default=1, type=int)
    parser.add_argument('-nn', choices=['cnn'], default='cnn')
    parser.add_argument('-o', '--optimizer', choices=['adam', 'sgd', 'adamw', 'rmsprop'], default='sgd')
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-emin', '--min-epochs', type=int, default=5)
    parser.add_argument('-emax', '--max-epochs', type=int, default=20)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('-wd', '--weight-decay', type=float, default=.00001)
    parser.add_argument('--config-file', default='config.yml')
    parser.add_argument('-ks', '--kernel-size', type=int, default=7, help="Size of the initial CNN filters. It is 7 on resnet.")
    parser.add_argument('-st', '--stride', type=int, default=2, help='Stride of the initial CNN filters. Normally in ResNet it is 2')
    parser.add_argument('--lr-decay-epochs', type=int, default=2)
    parser.add_argument('--loss-func', choices=['ce', 'bce'], default='bce')
    parser.add_argument('--base-net', choices=['resnet18', 'densenet18'], default='densenet18')

    subparsers = parser.add_subparsers(help='Just pick ARDS like a good scientist')
    ards_parser = subparsers.add_parser('ards')
    ards_parser.set_defaults(dataset='ards', model='prm')
    ards_parser.add_argument('-dp', '--dataset-path', default='/fastdata/ardsdetection')
    ards_parser.add_argument('-c', '--cohort', default='/fastdata/ardsdetection/cohort-description.csv')
    ards_parser.add_argument('--input-units', type=int, default=5096, help='size of input sequence')
    ards_parser.add_argument('--pr-thresh', type=float, default=2.0, help='peak response threshold')

    return parser


def get_dataset(args):
    if args.train_from_pickle:
        dataset = ARDSRawDataset.from_pickle(args.train_from_pickle)
        dataset.total_kfolds = args.n_folds
        dataset.kfold_num = 0
    else:
        dataset = ARDSRawDataset(
            args.dataset_path,
            '1',
            args.cohort,
            args.input_units,
            all_sequences=[],
            kfold_num=0,
            total_kfolds=args.n_folds,
            oversample_minority=True,
        )
    if args.train_to_pickle:
        pd.to_pickle(dataset, args.train_to_pickle)

    return dataset


def main():
    args = build_parser().parse_args()

    dataset = get_dataset(args)
    results = ModelCollection()
    gpus = {True: args.n_gpus, False: None}[args.cuda]
    for fold_num in range(args.n_folds):
        # XXX can add checkpoint callbacks if you want too. See the checkpoint
        # documentation if you want
        #
        # XXX need to add a custom accuracy monitor. however for now we can
        # just use val_loss
        stopping = EarlyStopping(min_delta=0.0, monitor='val_loss', patience=3, verbose=True, mode='min')
        trainer = pl.Trainer(max_epochs=args.max_epochs, min_epochs=args.min_epochs, gpus=gpus, callbacks=[stopping])
        model = ARDSSystem(args, dataset, results, fold_num)
        trainer.fit(model)
    import IPython; IPython.embed()
    # XXX add results viz
    # we can do inferencing later
    #model.perform_inference()


if __name__ == "__main__":
    main()
