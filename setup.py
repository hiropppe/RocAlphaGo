import numpy

from distutils.core import setup
from Cython.Build import cythonize

setup(
  ext_modules=cythonize(
      ["AlphaGo/preprocessing/rollout_feature.pyx"],
      language="c++",
      extra_compile_args=["-std=c++11"],
      extra_link_args=["-std=c++11"]
  ),
  include_dirs=[numpy.get_include()]
)
