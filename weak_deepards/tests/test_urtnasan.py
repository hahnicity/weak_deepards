import torch
from torch import nn

from weak_deepards.models.base import urtnasan


class TestUrtnasan(object):
    def __init__(self):
        self.model = urtnasan.UrtnasanNet()

    def test_sunny_day(self):
        rand = torch.rand((16, 1, 6000))
        output = self.model(rand)
        assert list(output.shape) == [16, 2]
