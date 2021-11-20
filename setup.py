"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""
# import setuptools
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('bbox', ['/content/drive/MyDrive/repos/S3FD_RepVGG_A1/box_overlaps.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))
