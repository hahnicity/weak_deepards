"""
urtnasan
~~~~~~~~

Model based on Urtnasan's ECG Apnea work
"""
from torch import nn


class UrtnasanNet(nn.Module):
    def __init__(self):
        super(UrtnasanNet, self).__init__()
        # Can just have 10 "layers" for now
        self.features = nn.Sequential(
            nn.BatchNorm1d(1),
            # kernel sizes are modified because of our input size being different
            #
            nn.Conv1d(1, 20, kernel_size=50, stride=1), # downsampled to 5950 size
            nn.ReLU(),
            nn.MaxPool1d(2), # downsampled to 2975
            nn.Conv1d(20, 20, kernel_size=50, stride=1), # downsampled to 2925 size
            nn.ReLU(),
            nn.Dropout(p=.25),
            nn.MaxPool1d(2), # downsampled to 1462
            nn.Conv1d(20, 24, kernel_size=40, stride=1), # downsampled to 1442 size
            nn.ReLU(),
            nn.MaxPool1d(2), # downsampled to 721
            nn.Dropout(p=.25),
            nn.Conv1d(24, 24, kernel_size=40, stride=1), # downsampled to 681 size
            nn.ReLU(),
            nn.MaxPool1d(2), # downsampled to 340
            nn.Dropout(p=.25),
            nn.Conv1d(24, 24, kernel_size=30, stride=1), # downsampled to 310 size
            nn.ReLU(),
            nn.MaxPool1d(2), # downsampled to 155
            nn.Dropout(p=.25),
            nn.Conv1d(24, 30, kernel_size=30, stride=1), # downsampled to 125 size
            nn.ReLU(),
            nn.MaxPool1d(2), # downsampled to 62
            nn.Dropout(p=.25),
            nn.Conv1d(30, 12, kernel_size=12, stride=1), # downsampled to 50 size
            nn.ReLU(),
            nn.MaxPool1d(2), # downsampled to 25
            nn.Dropout(p=.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(25*12, 2),
        )

    def forward(self, x):
        batches, chans, len = x.shape
        x = self.features(x)
        x = x.view(batches, -1)
        return self.classifier(x)
