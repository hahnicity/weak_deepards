"""
dey
~~~

Model based on Dey's 2017 paper on using CNN for apnea-ecg
"""
from torch import nn


class DeyNet(nn.Module):
    def __init__(self):
        super(DeyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 20, stride=10, kernel_size=60, bias=False, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4, padding=0),
            nn.Conv1d(20, 28, stride=2, kernel_size=3, bias=False, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1036, 2072),
            nn.Linear(2072, 2),
        )

    def forward(self, x):
        x = self.features(x)
        batches = x.shape[0]
        return self.classifier(x.view(batches, -1))
