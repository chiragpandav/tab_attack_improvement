"""Synthesizers module."""

from CTGAN.ctgan.synthesizers.ctgan import CTGAN
from CTGAN.ctgan.synthesizers.tvae import TVAE

__all__ = ('CTGAN', 'TVAE')


def get_all_synthesizers():
    return {name: globals()[name] for name in __all__}
