import argparse
from glob import glob
import math
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import yaml

ards_train =  ['0723RPI2120190416', '0015RPI0320150401', '0021RPI0420150513', '0026RPI1020150523', '0027RPI0620150525', '0093RPI0920151212', '0098RPI1420151218', '0099RPI0120151219', '0102RPI0120151225', '0120RPI1820160118', '0129RPI1620160126', '0147RPI1220160213', '0148RPI0120160214', '0149RPI1820160212', '0153RPI0720160217', '0194RPI0320160317', '0209RPI1920160408', '0224RPI3020160414', '0243RPI0720160512', '0245RPI1420160512', '0253RPI1220160606', '0260RPI2420160617', '0265RPI2920160622', '0266RPI1720160622', '0268RPI1220160624', '0271RPI1220160630', '0372RPI2220161211', '0381RPI2320161212', '0390RPI2220161230', '0412RPI5520170121', '0484RPI4220170630', '0506RPI3720170807', '0511RPI5220170831', '0514RPI5420170905', '0527RPI0420171028', '0546RPI5120171216', '0549RPI4420171213', '0551RPI0720180102', '0569RPI0420180116', '0640RPI2820180822']

other_train = ['0033RPI0520150603', '0108RPI0120160101', '0111RPI1520160101', '0112RPI1620160105', '0124RPI1220160123', '0125RPI1120160123', '0133RPI0920160127', '0144RPI0920160212', '0145RPI1120160212', '0157RPI0920160218', '0163RPI0720160222', '0166RPI2220160227', '0170RPI2120160301', '0173RPI1920160303', '0257RPI1220160615', '0304RPI1620160829', '0306RPI3520160830', '0317RPI3220160910', '0336RPI3920161006', '0343RPI3920161016', '0347RPI4220161016', '0354RPI5820161029', '0356RPI2220161101', '0361RPI4620161115', '0365RPI5820161125', '0387RPI3920161224', '0398RPI4220170104', '0423RPI3220170205', '0434RPI4520170224', '0460RPI2220170518', '0463RPI3220170522', '0544RPI2420171204', '0545RPI0520171214', '0552RPI2520180101', '0585RPI2720180206', '0593RPI1920180226', '0624RPI0320180708', '0624RPI1920180702', '0625RPI2820180628', '0705RPI5020190318']

ards_test = ['0127RPI0120160124', '0411RPI5820170119', '0261RPI1220160617',
       '0235RPI1320160426', '0160RPI1420160220', '0122RPI1320160120',
       '0251RPI1820160609', '0139RPI1620160205', '0357RPI3520161101',
       '0558RPI0820180104']

other_test = ['0443RPI1620170319', '0410RPI4120170118', '0380RPI3920161212',
       '0745RPI1900000000', '0135RPI1420160203', '0231RPI1220160424',
       '0137RPI1920160202', '0315RPI2720160910', '0132RPI1720160127',
       '0225RPI2520160416']

aim1_train = ['0271RPI1220160630',
 '0027RPI0620150525',
 '0625RPI2820180628',
 '0343RPI3920161016',
 '0209RPI1920160408',
 '0372RPI2220161211',
 '0194RPI0320160317',
 '0149RPI1820160212',
 '0245RPI1420160512',
 '0257RPI1220160615',
 '0357RPI3520161101',
 '0268RPI1220160624',
 '0260RPI2420160617',
 '0412RPI5520170121',
 '0365RPI5820161125',
 '0434RPI4520170224',
 '0387RPI3920161224',
 '0546RPI5120171216',
 '0527RPI0420171028',
 '0593RPI1920180226',
 '0253RPI1220160606',
 '0108RPI0120160101',
 '0170RPI2120160301',
 '0390RPI2220161230',
 '0133RPI0920160127',
 '0511RPI5220170831',
 '0111RPI1520160101',
 '0225RPI2520160416',
 '0304RPI1620160829',
 '0398RPI4220170104',
 '0033RPI0520150603',
 '0347RPI4220161016',
 '0231RPI1220160424',
 '0144RPI0920160212',
 '0315RPI2720160910',
 '0265RPI2920160622',
 '0544RPI2420171204',
 '0098RPI1420151218',
 '0261RPI1220160617',
 '0624RPI1920180702',
 '0460RPI2220170518',
 '0463RPI3220170522',
 '0624RPI0320180708',
 '0317RPI3220160910',
 '0251RPI1820160609',
 '0585RPI2720180206',
 '0166RPI2220160227',
 '0423RPI3220170205',
 '0153RPI0720160217',
 '0021RPI0420150513',
 '0551RPI0720180102',
 '0102RPI0120151225',
 '0361RPI4620161115',
 '0160RPI1420160220',
 '0127RPI0120160124',
 '0545RPI0520171214',
 '0235RPI1320160426',
 '0122RPI1320160120',
 '0139RPI1620160205',
 '0266RPI1720160622',
 '0129RPI1620160126',
 '0484RPI4220170630',
 '0506RPI3720170807',
 '0354RPI5820161029',
 '0093RPI0920151212',
 '0640RPI2820180822',
 '0356RPI2220161101',
 '0443RPI1620170319',
 '0124RPI1220160123',
 '0410RPI4120170118']

aim1_test = ['0411RPI5820170119',
 '0224RPI3020160414',
 '0336RPI3920161006',
 '0147RPI1220160213',
 '0514RPI5420170905',
 '0099RPI0120151219',
 '0558RPI0820180104',
 '0552RPI2520180101',
 '0148RPI0120160214',
 '0243RPI0720160512',
 '0549RPI4420171213',
 '0163RPI0720160222',
 '0132RPI1720160127',
 '0026RPI1020150523',
 '0015RPI0320150401',
 '0380RPI3920161212',
 '0120RPI1820160118',
 '0137RPI1920160202',
 '0381RPI2320161212',
 '0112RPI1620160105',
 '0135RPI1420160203',
 '0723RPI2120190416',
 '0306RPI3520160830',
 '0173RPI1920160303',
 '0569RPI0420180116',
 '0125RPI1120160123',
 '0705RPI5020190318',
 '0745RPI1900000000',
 '0157RPI0920160218',
 '0145RPI1120160212']


class Splitting(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def create_ards_split(self, pts, main_dirname):
        all_data_dir = os.path.join(self.dataset_path, 'experiment1/all_data')
        all_data_raw_dir = os.path.join(all_data_dir, 'raw')
        all_data_meta_dir = os.path.join(all_data_dir, 'meta')

        dir = os.path.join(self.dataset_path, 'experiment1', main_dirname)
        try:
            shutil.rmtree(dir)
        except OSError:
            pass
        os.mkdir(dir)
        raw_dir = os.path.join(dir, 'raw')
        meta_dir = os.path.join(dir, 'meta')
        os.mkdir(raw_dir)
        os.mkdir(meta_dir)

        for pt in pts:
            proc = subprocess.Popen(['ln', '-s', os.path.join(all_data_raw_dir, pt), raw_dir])
            proc.communicate()
            proc = subprocess.Popen(['ln', '-s', os.path.join(all_data_meta_dir, pt), meta_dir])
            proc.communicate()

    def create_apnea_ecg_split(self, records, dataset_name):
        """
        Create split for the Apnea-ECG dataset.

        :param records: Records we want to incorporate into the split
        :param dataset_name: Name of the dataset Eg. foo_splittrain
        """
        new_dir = os.path.join(self.dataset_path, dataset_name)
        try:
            shutil.rmtree(new_dir)
        except OSError:
            pass
        os.mkdir(new_dir)
        for record in records:
            all_record_files = glob(os.path.join(self.dataset_path, record + ".*"))
            for r in all_record_files:
                proc = subprocess.Popen(['ln', '-s', r, new_dir])
                proc.communicate()


def perform_random_split(dataset_path, split_ratio, validation_ratio, out_dir_prefix):
    ards_pts = ards_train + ards_test
    other_pts = other_train + other_test
    all_pts = ards_pts + other_pts
    len_patho_test_pts = int((len(all_pts) * split_ratio) / 2)
    other_test_pts = list(np.random.choice(other_pts, size=len_patho_test_pts, replace=False))
    ards_test_pts = list(np.random.choice(ards_pts, size=len_patho_test_pts, replace=False))
    test_pts = other_test_pts + ards_test_pts
    train_pts = set(all_pts).difference(set(test_pts))
    dir_prefix = out_dir_prefix if out_dir_prefix is not None else 'random'
    val_n = int(math.ceil(len(test_pts) * validation_ratio))

    splitter = Splitting(dataset_path)
    splitter.create_ards_split(train_pts, '{}train'.format(dir_prefix))

    if val_n > 0:
        ards_val_pts = np.random.choice(ards_test_pts, size=val_n/2, replace=False)
        other_val_pts = np.random.choice(other_test_pts, size=val_n/2, replace=False)
        val_pts = list(ards_val_pts) + list(other_val_pts)
        splitter.create_ards_split(val_pts, '{}val'.format(dir_prefix))
        test_pts = list(set(test_pts).difference(val_pts))

    splitter.create_ards_split(test_pts, '{}test'.format(dir_prefix))


def perform_preset_proto_split(dataset_path):
    splitter = Splitting(dataset_path)
    splitter.create_ards_split(ards_train+other_train, 'prototrain')
    splitter.create_ards_split(ards_test+other_test, 'prototest')


def perform_preset_aim1_split(dataset_path):
    splitter = Splitting(dataset_path)
    splitter.create_ards_split(aim1_train, 'training')
    splitter.create_ards_split(aim1_test, 'testing')


def perform_preset_file_split(dataset_path, file_path):
    with open(file_path) as preset_file:
        conf = yaml.load(preset_file, Loader=yaml.FullLoader)
    train_pts = conf['train']
    test_pts = conf['test']
    split_name = os.path.splitext(os.path.basename(file_path))[0]
    splitter = Splitting(dataset_path)
    splitter.create_ards_split(train_pts, split_name + 'train')
    splitter.create_ards_split(test_pts, split_name + 'test')


def perform_random_apnea_ecg_split(dataset_path, split_ratio, validation_ratio, dataset_name):
    with open(os.path.join(dataset_path, 'RECORDS')) as f:
        all_records = [r.strip() for r in f.readlines()]
    apneic_pts = np.array([r for r in all_records if r.startswith('a') and 'r' not in r and 'er' not in r])
    borderline_pts = np.array([r for r in all_records if r.startswith('b') and 'r' not in r and 'er' not in r])
    non_apneic_pts = np.array([r for r in all_records if r.startswith('c') and 'r' not in r and 'er' not in r])
    # want as even split of apneic, non-apneic and borderline pts as possible
    x = np.concatenate([apneic_pts, borderline_pts, non_apneic_pts])
    y = np.concatenate([np.zeros(len(apneic_pts)), np.ones(len(borderline_pts)), np.ones(len(non_apneic_pts))+1])
    # train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio)
    train_idx, test_idx = next(sss.split(x, y))
    train_records = x[train_idx]
    test_records = x[test_idx]
    test_y = y[test_idx]

    # val/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-validation_ratio)
    val_idx, test_idx = next(sss.split(test_records, test_y))
    val_records = test_records[val_idx]
    test_records = test_records[test_idx]

    splitter = Splitting(dataset_path)
    splitter.create_apnea_ecg_split(train_records, dataset_name+'train')
    splitter.create_apnea_ecg_split(val_records, dataset_name+'val')
    splitter.create_apnea_ecg_split(test_records, dataset_name+'test')


def perform_preset_apnea_ecg_split(dataset_path):
    with open(os.path.join(dataset_path, 'RECORDS')) as f:
        all_records = set([r.strip()[:3] for r in f.readlines()])
    test_records = [r for r in all_records if r.startswith('x')]
    train_records = all_records.difference(set(test_records))
    splitter = Splitting(dataset_path)
    splitter.create_apnea_ecg_split(list(train_records), 'maintrain')
    splitter.create_apnea_ecg_split(list(test_records), 'maintest')


def perform_apnea_ecg_split(args):
    if args.set_type == 'random':
        perform_random_apnea_ecg_split(args.dataset_path, args.split_ratio, args.validation_ratio, args.split_name)
    elif args.set_type == 'preset_apnea':
        perform_preset_apnea_ecg_split(args.dataset_path)
    else:
        raise Exception('Can currently only set to random set type when splitting with apnea ECG')


def perform_ards_split(args):
    if args.set_type == 'preset_proto':
        perform_preset_proto_split(args.dataset_path)
    elif args.set_type == 'random':
        perform_random_split(args.dataset_path, args.split_ratio, args.validation_ratio, args.split_name)
    elif args.set_type == 'preset_aim1':
        perform_preset_aim1_split(args.dataset_path)
    elif args.set_type == 'preset_file':
        if args.preset_file is None:
            raise Exception('If you are using preset_file split you must set --preset-file flag to a valid filepath')
        perform_preset_file_split(args.dataset_path, args.preset_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('set_type', choices=['preset_proto', 'preset_aim1', 'random', 'preset_file', 'preset_apnea'], help="""
        Split your data in a specific format:

        *preset_proto:* Utilize the proto train/test split. As it implies, used for prototyping purposes and shouldn't be used for result reporting
        *preset_aim1:* Use the preset holdout split we used for initial Aim 1 paper.
        *random:* Use a random split of the patients with a validation set.
        *preset_apnea:* Use standard apnea ecg train/test split
    """)
    parser.add_argument('-sr', '--split-ratio', type=float, default=.4)
    parser.add_argument('-vr', '--validation-ratio', type=float, default=.375, help='Ratio of the testing set to split into the validation set. Only used for the random split type. If you dont want a validation set, set this argument to 0')
    parser.add_argument('-n', '--split-name', help='New name train/test splits. Only used for random splits. If unset will just revert to default "main"', default='main')
    parser.add_argument('-f', '--preset-file', help='Path to file where we set our train/test splits')
    subparsers = parser.add_subparsers(help='Specify which dataset you are going to use')
    ards = subparsers.add_parser('ards')
    ards.set_defaults(dataset='ards')
    ecg = subparsers.add_parser('apnea_ecg')
    ecg.set_defaults(dataset='apnea_ecg')
    args = parser.parse_args()

    if args.dataset == 'ards':
        perform_ards_split(args)
    elif args.dataset == 'apnea_ecg':
        perform_apnea_ecg_split(args)


if __name__ == "__main__":
    main()
