from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from weak_deepards.dataset import ARDSRawDataset


class System(pl.LightningModule):
    def __init__(self, hparams):
        super(System, self).__init__()
        self.input_units = hparams.input_units
        # XXX add network
        self.pattern_size = hparams.pattern_size
        self.hparams = hparams
        self.batch_idx = 0

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        self.batch_idx = batch_idx
        x, y = batch
        y = torch.autograd.Variable(torch.stack(y, dim=1).reshape(len(y[0]), 2))
        y_hat = self.forward(x.float())
        loss = F.binary_cross_entropy(y_hat, y.float())
        board_logs = {'train_loss': loss}
        return {'loss': loss, 'log': board_logs}

    def validation_step(self, batch, batch_idx):
        self.batch_idx = batch_idx
        x, y = batch
        y = torch.stack(y, dim=1).reshape(len(y[0]), 1, 2)
        y_hat = self.forward(x.float())
        loss = F.binary_cross_entropy(y_hat, y.float())
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
        x, y = batch
        y = torch.stack(y, dim=1)
        y_hat = self.forward(x.float())
        loss = F.binary_cross_entropy(y_hat, y.float())
        return {'test_loss': loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    @pl.data_loader
    def train_dataloader(self):
        # XXX
        return DataLoader()

    @pl.data_loader
    def val_dataloader(self):
        # XXX
        return DataLoader()


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-units', type=int, default=224)
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
