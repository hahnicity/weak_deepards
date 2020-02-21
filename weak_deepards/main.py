from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from weak_deepards.dataset import ARDSRawDataset
from weak_deepards.models.base.resnet import resnet18
from weak_deepards.models.modules.peak_response_mapping import PeakResponseMapping


class System(pl.LightningModule):
    def __init__(self, hparams):
        super(System, self).__init__()
        with open(hparams.config_file) as conf:
            config = yaml.safe_load(conf)
        self.input_units = hparams.input_units
        base = resnet18()
        self.model = PeakResponseMapping(base, **config['model'])
        self.hparams = hparams

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pt, x, y = batch
        x = x.unsqueeze(1)
        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)
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
        loss = F.cross_entropy(y_hat, y.long().argmax(dim=1))
        return {'val_loss': loss}

    def validation_end(self, outputs):
        val_loss_mean = 0
        for out in outputs:
            val_loss_mean += out['val_loss']

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
        return torch.optim.SGD(self.parameters(), lr=0.01)

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
            )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)


def main():
    parser = ArgumentParser()
    parser.add_argument('--train-to-pickle', default='')
    parser.add_argument('--train-from-pickle', default='')
    parser.add_argument('--test-to-pickle', default='')
    parser.add_argument('--test-from-pickle', default='')
    parser.add_argument('-dp', '--dataset-path', default='/fastdata/ardsdetection_data')
    parser.add_argument('-c', '--cohort', default='cohort-description.csv')
    parser.add_argument('--config-file', default='config.yml')
    parser.add_argument('--input-units', type=int, default=5096)
    parser.add_argument('-ps', '--pattern-size', type=int, default=10)
    parser.add_argument('-pf', '--pattern-freq', type=float, default=.5)
    parser.add_argument('-emin', '--min-epochs', type=int, default=5)
    parser.add_argument('-emax', '--max-epochs', type=int, default=20)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-nn', choices=['cnn'], default='cnn')
    args = parser.parse_args()

    model = System(args)
    trainer = pl.Trainer(max_epochs=args.max_epochs, min_epochs=args.min_epochs)
    trainer.fit(model)


if __name__ == "__main__":
    main()
