from copy import copy
from glob import glob
import math
import os
import re

from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from ventmap.raw_utils import extract_raw, read_processed_file
from wfdb.io import rdann, rdsamp


class ARDSRawDataset(Dataset):
    def __init__(self,
                 data_path,
                 experiment_num,
                 cohort_file,
                 seq_len,
                 all_sequences=[],
                 train=True,
                 kfold_num=None,
                 total_kfolds=None,
                 oversample_minority=False,
                 holdout_set_type='main'):
        """
        Dataset to generate sequences of data for ARDS Detection
        """
        self.seq_len = seq_len
        self.train = train
        self.kfold_num = kfold_num
        self.all_sequences = all_sequences
        self.total_kfolds = total_kfolds
        self.cohort_file = cohort_file
        self.oversample = oversample_minority

        self.cohort = pd.read_csv(cohort_file)
        self.cohort = self.cohort.rename(columns={'Patient Unique Identifier': 'patient_id'})
        self.cohort['patient_id'] = self.cohort['patient_id'].astype(str)

        if kfold_num is None and holdout_set_type == 'proto':
            data_subdir = 'prototrain' if train else 'prototest'
        elif kfold_num is None and holdout_set_type == 'main':
            data_subdir = 'training' if train else 'testing'
        else:
            data_subdir = 'all_data'

        if self.all_sequences != []:
            self.finalize_dataset_create(kfold_num)
            return

        raw_dir = os.path.join(data_path, 'experiment{}'.format(experiment_num), data_subdir, 'raw')
        self.meta_dir = os.path.join(data_path, 'experiment{}'.format(experiment_num), data_subdir, 'meta')
        if not os.path.exists(raw_dir):
            raise Exception('No directory {} exists!'.format(raw_dir))
        self.raw_files = sorted(glob(os.path.join(raw_dir, '*/*.raw.npy')))
        self.processed_files = sorted(glob(os.path.join(raw_dir, '*/*.processed.npy')))
        self.meta_files = sorted(glob(os.path.join(self.meta_dir, '*/*.csv')))
        self.get_dataset()
        self.finalize_dataset_create(kfold_num)

    def finalize_dataset_create(self, kfold_num):
        self.derive_scaling_factors()
        if kfold_num is not None:
            self.set_kfold_indexes_for_fold(kfold_num)

    def set_oversampling_indices(self):
        # Cannot oversample with testing set
        if not self.train:
            return

        if not self.oversample:
            return

        if self.total_kfolds:
            x = self.non_oversampled_kfold_indexes
            y = [self.all_sequences[idx][-1].argmax() for idx in x]
            ros = RandomOverSampler()
            x_resampled, y_resampled = ros.fit_resample(np.array(x).reshape(-1, 1), y)
            self.kfold_indexes = x_resampled.ravel()
        else:
            raise NotImplementedError('We havent implemented oversampling for holdout sets yet')

    def derive_scaling_factors(self):
        is_kfolds = self.total_kfolds is not None
        if is_kfolds:
            indices = [self.get_kfold_indexes_for_fold(kfold_num) for kfold_num in range(self.total_kfolds)]
        else:
            indices = [range(len(self.all_sequences))]

        self.scaling_factors = {
            kfold_num if is_kfolds else None: self._get_scaling_factors_for_indices(idxs)
            for kfold_num, idxs in enumerate(indices)
        }

    @classmethod
    def make_test_dataset_if_kfold(self, train_dataset):
        test_dataset = ARDSRawDataset(
            None,
            None,
            train_dataset.cohort_file,
            train_dataset.seq_len,
            all_sequences=train_dataset.all_sequences,
            train=False,
            kfold_num=train_dataset.kfold_num,
            total_kfolds=train_dataset.total_kfolds,
        )
        return test_dataset

    @classmethod
    def from_pickle(self, data_path, oversample_minority=True):
        dataset = pd.read_pickle(data_path)
        if not isinstance(dataset, ARDSRawDataset):
            raise ValueError('The pickle file you have specified is out-of-date. Please re-process your dataset and save the new pickled dataset.')
        self.oversample = oversample_minority
        self.train = True
        return dataset

    def set_kfold_indexes_for_fold(self, kfold_num):
        self.kfold_num = kfold_num
        self.kfold_indexes = self.get_kfold_indexes_for_fold(kfold_num)
        self.non_oversampled_kfold_indexes = copy(list(self.kfold_indexes))
        self.set_oversampling_indices()

    def get_kfold_indexes_for_fold(self, kfold_num):
        ground_truth = self._get_all_sequence_ground_truth()
        other_patients = ground_truth[ground_truth.y == 0].patient.unique()
        ards_patients = ground_truth[ground_truth.y == 1].patient.unique()
        all_patients = np.append(other_patients, ards_patients)
        patho = [0] * len(other_patients) + [1] * len(ards_patients)
        kfolds = StratifiedKFold(n_splits=self.total_kfolds)
        for split_num, (train_pt_idx, test_pt_idx) in enumerate(kfolds.split(all_patients, patho)):
            train_pts = all_patients[train_pt_idx]
            test_pts = all_patients[test_pt_idx]
            if split_num == kfold_num and self.train:
                return ground_truth[ground_truth.patient.isin(train_pts)].index
            elif split_num == kfold_num and not self.train:
                return ground_truth[ground_truth.patient.isin(test_pts)].index

    def get_dataset(self):
        last_patient = None

        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, self.processed_files[fidx])
            patient_id = self._get_patient_id_from_file(filename)

            if patient_id != last_patient:
                seq_arr = list()

            last_patient = patient_id
            start_time = self._get_patient_start_time(patient_id)

            for bidx, breath in enumerate(gen):
                # cutoff breaths if they have too few points. It is unlikely ML
                # will ever learn anything useful from them. 21 is chosen because the mean
                if len(breath['flow']) < 21:
                    continue

                if isinstance(breath['abs_bs'], bytes):
                    abs_bs = breath['abs_bs'].decode('utf8')
                else:
                    abs_bs = breath['abs_bs']

                try:
                    breath_time = pd.to_datetime(abs_bs, format='%Y-%m-%d %H-%M-%S.%f')
                except:
                    breath_time = pd.to_datetime(abs_bs, format='%Y-%m-%d %H:%M:%S.%f')

                if breath_time < start_time:
                    continue
                elif breath_time > start_time + pd.Timedelta(hours=24):
                    break

                flow = breath['flow']
                # XXX for now don't drop breaths. But later might have to try this.
                if len(flow) + len(seq_arr) >= self.seq_len:
                    remainder = self.seq_len - len(seq_arr)
                    seq_arr.extend(flow[:remainder])
                    self.all_sequences.append([patient_id, np.array(seq_arr), self._pathophysiology_target(patient_id)])
                    seq_arr = flow[remainder:]
                else:
                    seq_arr.extend(flow)

    def _get_scaling_factors_for_indices(self, indices):
        """
        Get mu and std for a specific set of indices
        """
        std_sum = 0
        mean_sum = 0
        obs_count = 0

        for idx in indices:
            obs = self.all_sequences[idx][1]
            obs_count += len(obs)
            mean_sum += obs.sum()
        mu = mean_sum / obs_count

        # calculate std
        for idx in indices:
            obs = self.all_sequences[idx][1]
            std_sum += ((obs - mu) ** 2).sum()
        std = np.sqrt(std_sum / obs_count)
        return mu, std

    def _pathophysiology_target(self, patient_id):
        patient_row = self.cohort[self.cohort['patient_id'] == patient_id]
        try:
            patient_row = patient_row.iloc[0]
        except:
            raise ValueError('Could not find patient {} in cohort file'.format(patient_id))
        patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0
        target = np.zeros(2)
        target[patho] = 1
        return target

    def _get_patient_start_time(self, patient_id):
        patient_row = self.cohort[self.cohort['patient_id'] == patient_id]
        patient_row = patient_row.iloc[0]
        patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0
        if patho == 1:
            start_time = pd.to_datetime(patient_row['Date when Berlin criteria first met (m/dd/yyy)'])
        else:
            start_time = pd.to_datetime(patient_row['vent_start_time'])

        if start_time is pd.NaT:
            raise Exception('Could not find valid start time for {}'.format(patient_id))
        return start_time

    def __getitem__(self, index):
        if self.kfold_num is not None:
            index = self.kfold_indexes[index]
        seq = self.all_sequences[index]
        pt, data, target = seq
        try:
            mu, std = self.scaling_factors[self.kfold_num]
        except AttributeError:
            raise AttributeError('Scaling factors not found for dataset. You must derive them using the `derive_scaling_factors` function.')
        data = (data - mu) / std

        return index, pt, data, target

    def __len__(self):
        if self.kfold_num is None:
            return len(self.all_sequences)
        else:
            return len(self.kfold_indexes)

    def get_ground_truth_df(self):
        if self.kfold_num is None:
            return self._get_all_sequence_ground_truth()
        else:
            return self._get_kfold_ground_truth()

    def _get_all_sequence_ground_truth(self):
        rows = []
        for seq in self.all_sequences:
            patient, _, target = seq
            rows.append([patient, np.argmax(target, axis=0)])
        return pd.DataFrame(rows, columns=['patient', 'y'])

    def _get_kfold_ground_truth(self):
        rows = []
        for idx in self.kfold_indexes:
            seq = self.all_sequences[idx]
            patient, _, target = seq
            rows.append([patient, np.argmax(target, axis=0)])
        return pd.DataFrame(rows, columns=['patient', 'y'], index=self.kfold_indexes)

    def _get_patient_id_from_file(self, filename):
        pt_id = filename.split('/')[-2]
        # sanity check to see if patient
        match = re.search(r'(0\d{3}RPI\d{10})', filename)
        if match:
            return match.groups()[0]
        try:
            # id is from anonymous dataset
            float(pt_id)
            return pt_id
        except:
            raise ValueError('could not find patient id in file: {}'.format(filename))


class ApneaECGDataset(object):
    def __init__(self, dataset_path, dataset_type, split_name, scaling_type, processing_version):
        """
        :param dataset_path: directory path to the Apnea-ECG dataset
        :param dataset_type: What set we are using (train/val/test)
        :param split_name: What is the name of our split. Eg. foo_split
        :param scaling_type: "inter" for inter patient scaling or "intra" for intra-patient scaling
        :param processing_version: 'v1' if we want to process data only from beginning of each minute. v2 if we want to start at the middle of each minute
        """
        if dataset_type not in ['train', 'val', 'test']:
            raise Exception('dataset_type must be either "train", "val", or "test"')
        if scaling_type not in ['inter', 'intra']:
            raise Exception('scaling_type must be either "inter" or "intra"')
        if processing_version not in ['v1', 'v2']:
            raise Exception('processing version must be either v1 or v2')

        self.dataset_path = os.path.join(dataset_path, split_name+dataset_type)
        self.scaling_type = scaling_type
        self.record_set = [
            os.path.splitext(os.path.basename(f))[0]
            for f in glob(os.path.join(self.dataset_path, '*.dat'))
        ]
        self.all_sequences = []
        if processing_version == 'v1':
            self.process_dataset_v1()
        elif processing_version == 'v2':
            self.process_dataset_v2()

        if self.scaling_type == 'intra':
            self.obtain_intra_patient_scaling_coefs()
        elif self.scaling_type == 'inter':
            self.obtain_inter_patient_scaling_coefs()
        self.dt = 0.01

    def process_dataset_v1(self):
        """
        Process the dataset given the assumption that apnea is within the 1 minute interval.
        For practical purposes, we will version this as v1. This is not the totally correct
        interpretation of the Apnea-ECG dataset, but from what I've seen there seem to be
        varying annotation patterns in the dataset and for now this template will serve
        sufficiently for v1 purposes.
        """
        for record_name in self.record_set:
            record_path = os.path.join(self.dataset_path, record_name)
            data, metadata = rdsamp(record_path)
            annos = rdann(record_path, 'apn')

            for i, anno in enumerate(annos.symbol):
                start_idx = i * 6000
                end_idx = (i+1) * 6000

                # Ensure that we have uniform sized vectors
                try:
                    minute_data = data[start_idx:end_idx].reshape(1, 6000)
                except:
                    continue

                one_hot = {'A': [0, 1], 'N': [1, 0]}[anno]
                self.all_sequences.append((i, record_name, minute_data, np.array(one_hot)))

    def process_dataset_v2(self):
        """
        Second variant of dataset processing. I don't know if the v2 means its significantly
        better or not, but we do try to shift the data frame so that it takes the input more
        into account.
        """
        for record_name in self.record_set:
            record_path = os.path.join(self.dataset_path, record_name)
            data, metadata = rdsamp(record_path)
            annos = rdann(record_path, 'apn')
            # first annotation is skipped
            for i, anno in enumerate(annos.symbol[1:]):
                start_idx = i * 6000 + 3000
                end_idx = (i+1) * 6000 + 3000

                # Ensure that we have uniform sized vectors
                try:
                    minute_data = data[start_idx:end_idx].reshape(1, 6000)
                except:
                    continue

                one_hot = {'A': [0, 1], 'N': [1, 0]}[anno]
                self.all_sequences.append((i, record_name, minute_data, np.array(one_hot)))

    def obtain_inter_patient_scaling_coefs(self):
        self.coefs = {'sum': 0, 'len': 0, 'stdsum': 0}
        for idx, record_name, data, one_hot in self.all_sequences:
            self.coefs['sum'] += data[0].sum()
            self.coefs['len'] += len(data[0])

        self.coefs['mean'] = self.coefs['sum'] / self.coefs['len']
        for idx, record_name, data, one_hot in self.all_sequences:
            self.coefs['stdsum'] += ((data[0] - self.coefs['mean']) ** 2).sum()
        self.coefs['std'] = np.sqrt(self.coefs['stdsum'] / self.coefs['len'])

    def obtain_intra_patient_scaling_coefs(self):
        """
        Get scaling coefficients for each patient
        """
        last_record = self.all_sequences[0][1]
        self.coefs = {last_record: {'sum': 0, 'len': 0, 'stdsum': 0}}

        for idx, record_name, data, one_hot in self.all_sequences:
            if record_name != last_record:
                self.coefs[last_record]['mean'] = self.coefs[last_record]['sum'] / self.coefs[last_record]['len']
                last_record = record_name
                self.coefs[record_name] = {'sum': 0, 'len': 0, 'stdsum': 0}
            self.coefs[record_name]['sum'] += data[0].sum()
            self.coefs[record_name]['len'] += len(data[0])
        else:
            self.coefs[last_record]['mean'] = self.coefs[last_record]['sum'] / self.coefs[last_record]['len']


        last_record = self.all_sequences[0][1]
        for idx, record_name, data, one_hot in self.all_sequences:
            if record_name != last_record:
                self.coefs[last_record]['std'] = np.sqrt(self.coefs[last_record]['stdsum'] / self.coefs[last_record]['len'])
                last_record = record_name
            self.coefs[record_name]['stdsum'] += ((data[0] - self.coefs[record_name]['mean']) ** 2).sum()
        else:
            self.coefs[last_record]['std'] = np.sqrt(self.coefs[last_record]['stdsum'] / self.coefs[last_record]['len'])

    def __getitem__(self, idx):
        """
        get next sequence
        """
        idxs, record, data, y = self.all_sequences[idx]

        if self.scaling_type == 'intra':
            mu, std = self.coefs[record]['mean'], self.coefs[record]['std']
        elif self.scaling_type == 'inter':
            mu, std = self.coefs['mean'], self.coefs['std']

        data = (data - mu) / std
        return (idxs, record, data, y)

    def __len__(self):
        return len(self.all_sequences)
