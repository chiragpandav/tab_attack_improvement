# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '0.10.3.dev0'

from CTGAN.ctgan.demo import load_demo
from CTGAN.ctgan.synthesizers.ctgan import CTGAN
from CTGAN.ctgan.synthesizers.tvae import TVAE

__all__ = ('CTGAN', 'TVAE', 'load_demo')
