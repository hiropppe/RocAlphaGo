#!/usr/bin/env python

import numpy

from setuptools import setup, find_packages

from distutils import core
from Cython.Build import cythonize

core.setup(
  ext_modules=cythonize(
      ["AlphaGo/cython/go.pyx",
       "AlphaGo/cython/point.pyx",
       "AlphaGo/cython/pattern.pyx",
       "AlphaGo/cython/printer.pyx",
       "AlphaGo/cython/policy_feature.pyx",
       "AlphaGo/cython/arrtest.pyx",
       "AlphaGo/cython/parseboard.pyx",
       "AlphaGo/cython/test_go.pyx",
       "AlphaGo/cython/test_ladder.pyx",
       "AlphaGo/cython/test_policy_feature.pyx",
       "AlphaGo/preprocessing/rollout_feature.pyx",
       "AlphaGo/preprocessing/nakade.pyx"],
      language="c++",
      extra_compile_args=["-std=c++11"],
      extra_link_args=["-std=c++11"],
      gdb_debug=True
  ),
  include_dirs=[numpy.get_include()]
)


requires = [
]

setup(
  name='RocAlphaGo',
  version='0.0.1',
  author='take',
  url='',
  packages=find_packages(),
  scripts=[
    'scripts/rag',
    'scripts/ragc'
  ],
  install_requires=requires,
  license='MIT',
  test_suite='test',
  classifiers=[
    'Operating System :: OS Independent',
    'Environment :: Console',
    'Programming Language :: Python',
    'License :: OSI Approved :: MIT License',
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'Topic :: Utilities',
  ],
  data_files=[
  ]
)
