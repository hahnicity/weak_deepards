from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from weak_deepards.dataset import ARDSRawDataset
from weak_deepards.models.base.resnet import resnet18
from weak_deepards.models.prm import peak_response_mapping


class System(pl.LightningModule):
    def __init__(self, hparams):
        super(System, self).__init__()
        with open(hparams.config_file) as conf:
            config = yaml.safe_load(conf)
        self.input_units = hparams.input_units
        base = resnet18()
        self.model = peak_response_mapping(base, **config['model'])
        self.hparams = hparams

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pt, x, y = batch
        x = x.unsqueeze(1)
        x = Variable(x)
        y = Variable(y)
        y_hat = self.forward(x.float())
        # Can probably just use BCE on this one because we're only supposed to have one class
        # in each snippet of data. However I wonder what would happen if we stretched the window
        # wide enough and we started saying that there were both norm and ARDS pattern in it.
        # Could be a interesting experiment even if its a total failure
        #
        # XXX I dunno if using CE is right or if argmax is correct, but I'd like to see how
        # it pans out
        loss = F.cross_entropy(y_hat, y.long().argmax(dim=1))
        board_logs = {'train_loss': loss}
        return {'loss': loss, 'log': board_logs}

    def validation_step(self, batch, batch_idx):
        pt, x, y = batch
        x = x.unsqueeze(1)
        y_hat = self.forward(x.float())
        # XXX loss may be a bad metric for validation sets because we could def. have
        # non-ARDS-like data in a frame taken from an ARDS patient. Maybe F1 would be
        # a better metric?
        loss = F.cross_entropy(y_hat, y.long().argmax(dim=1))
        return {'val_loss': loss}

    def perform_inference(self):
        # need to enable grad because pytorch lightning disables in validation step
        # It doesn't matter how much we try to explicitly enable grad, if we are under
        # a with.no_grad header no grads will ever be calculated
        with torch.enable_grad():
            # Get batch from loader
            loader = self.val_dataloader()[0]
            for rand_batch in loader:
                break
            rand_idx = np.random.randint(rand_batch[1].shape[0])
            rand_seq = rand_batch[1][rand_idx].float().reshape(
                (1, 1, self.hparams.input_units)).cuda().requires_grad_()
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

    def validation_end(self, outputs):
        val_loss_mean = 0
        for out in outputs:
            val_loss_mean += out['val_loss']
        self.perform_inference()

        return {
            'progress_bar': {'val_loss': val_loss_mean / len(outputs)},
            'val_loss': val_loss_mean / len(outputs)
        }

    def test_step(self):
        pt, x, y = batch
        y = torch.stack(y, dim=1)
        y_hat = self.forward(x.float())
        loss = F.binary_cross_entropy(y_hat, y.float())
        return {'test_loss': loss}

    def configure_optimizers(self):
        sgd = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(sgd, 1, gamma=.2)
        return [sgd], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.train_from_pickle:
            dataset = ARDSRawDataset.from_pickle(self.hparams.train_from_pickle)
        else:
            dataset = ARDSRawDataset(
                self.hparams.dataset_path,
                '1',
                self.hparams.cohort,
                self.hparams.input_units,
                to_pickle=self.hparams.train_to_pickle,
                all_sequences=[],
                holdout_set_type='proto',
            )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams.test_from_pickle:
            dataset = ARDSRawDataset.from_pickle(self.hparams.test_from_pickle)
        else:
            dataset = ARDSRawDataset(
                self.hparams.dataset_path,
                '1',
                self.hparams.cohort,
                self.hparams.input_units,
                to_pickle=self.hparams.test_to_pickle,
                train=False,
                all_sequences=[],
                holdout_set_type='proto',
            )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)


def main():
    parser = ArgumentParser()
    parser.add_argument('--train-to-pickle', default='')
    parser.add_argument('--train-from-pickle', default='')
    parser.add_argument('--test-to-pickle', default='')
    parser.add_argument('--test-from-pickle', default='')
    parser.add_argument('-dp', '--dataset-path', default='/fastdata/ardsdetection_data_anon_non_consent_filtered')
    parser.add_argument('-c', '--cohort', default='/fastdata/ardsdetection_data_anon_non_consent_filtered/cohort-description.csv')
    parser.add_argument('--config-file', default='config.yml')
    parser.add_argument('--input-units', type=int, default=5096)
    parser.add_argument('-ps', '--pattern-size', type=int, default=10)
    parser.add_argument('-pf', '--pattern-freq', type=float, default=.5)
    parser.add_argument('-emin', '--min-epochs', type=int, default=5)
    parser.add_argument('-emax', '--max-epochs', type=int, default=20)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-nn', choices=['cnn'], default='cnn')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('--pr-thresh', type=float, default=2.0)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    # if we use multi-gpu pytorch-lightning gets all screwed up with reporting losses. I'm probably
    # just going to drop the lightning component
    gpus = {True: 1, False: None}[args.cuda]
    model = System(args)
    trainer = pl.Trainer(max_epochs=args.max_epochs, min_epochs=args.min_epochs, gpus=gpus)
    trainer.fit(model)
    model.inference()


if __name__ == "__main__":
    main()
