#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(name='weak deepards',
      version="1.0",
      description='Weakly Supervised Deep Learning For ARDS detection with Ventilator Waveform Data',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'prettytable',
          'scipy',
          'scikit-learn<0.21.0',
          'ventmap',
          'imbalanced-learn==0.4.3',
      ],
      entry_points={
      },
      )
