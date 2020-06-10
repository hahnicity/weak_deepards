import os

import numpy as np
from weak_deepards.dataset import ApneaECGDataset
from wfdb.io import rdann, rdsamp
import yaml


class TestApnea(object):
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), 'test_config.yml')
        with open(config_path) as conf:
            self.config = yaml.safe_load(conf)
        self.dataset = ApneaECGDataset(self.config['test_dataset_path'], 'train', 'main', 'intra')

    def test_sunny_day(self):
        item = self.dataset[0]
        first_record = self.dataset.record_set[0]
        assert item[1] == first_record
        annos = rdann(os.path.join(self.config['test_dataset_path'], first_record), 'apn')
        if (item[-1] == np.array([1, 0])).all():
            assert annos.symbol[0] == 'N'
        else:
            assert annos.symbol[0] == 'A'
        assert list(item[2].shape) == [1, 6000]
        assert first_record in self.dataset.coefs
        assert 'mean' in self.dataset.coefs[first_record]
        assert 'std' in self.dataset.coefs[first_record]
        mu = self.dataset.coefs[first_record]['mean']
        std = self.dataset.coefs[first_record]['std']
        samp = rdsamp(os.path.join(self.config['test_dataset_path'], first_record))[0]
        assert (((samp[0:6000] - mu)).ravel() / std == item[2].ravel()).all()

    def test_seqs_properly_aligned(self):
        annos = None
        samp = None
        last_rec = None
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            if item[1] != last_rec:
                annos = rdann(os.path.join(self.config['test_dataset_path'], item[1]), 'apn')
                samp = rdsamp(os.path.join(self.config['test_dataset_path'], item[1]))[0]
                minutes = annos.sample

            j = item[0]
            mu = self.dataset.coefs[item[1]]['mean']
            std = self.dataset.coefs[item[1]]['std']
            start = minutes[j]
            end = start + 6000
            assert (((samp[start:end] - mu)).ravel() / std == item[2].ravel()).all(), "record: {}, start: {}, end: {}".format(item[1], start, end)
            last_rec = item[1]
