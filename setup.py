from __future__ import print_function
import sys
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import platform


try:
    import numpy  # NOQA
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy  # NOQA
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

try:
    import Cython  # NOQA
except ImportError:
    print('Cython is required during installation')
    sys.exit(1)

import numpy as np

if platform.system() == 'Darwin':
    extra_compile_args = [
        '-I/System/Library/Frameworks/vecLib.framework/Headers'
    ]
    if 'ppc' in platform.machine():
        extra_compile_args.append('-faltivec')

    extra_link_args = ["-Wl,-framework", "-Wl,Accelerate"]
    include_dirs = [np.get_include()]

else:
    include_dirs = [np.get_include(), "/usr/local/include"]
    extra_compile_args = ['-msse2', '-O2', '-fPIC', '-w']
    extra_link_args = ["-llapack"]


# pyquic extension
# --> inverse_covariance.pyquic.pyquic (contains func quic)
ext_module = Extension(
    name="pyquic.pyquic",  # note: we ext_package= flag in setup()
    sources=[
        "inverse_covariance/pyquic/QUIC.C",
        "inverse_covariance/pyquic/pyquic.pyx"],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++"
)


with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


setup(
    name='skggm',
    version='0.2.7',
    description='Gaussian graphical models for scikit-learn.',
    author='Jason Laska and Manjari Narayan',
    license='MIT',
    packages=[
        'inverse_covariance',
        'inverse_covariance.profiling',
        'inverse_covariance.pyquic'],
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/skggm/skggm',
    author_email='jlaska@gmail.com',
    ext_package='inverse_covariance',
    ext_modules=cythonize(ext_module),
)
