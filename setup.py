from __future__ import print_function
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

from build_deps import setup_pyquic

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

class InstallWithSetup(install):
    def run(self):
        install.run(self)
        print('Compiling pyquic...')
        setup_pyquic()

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)


setup(name='skggm',
      version='0.1.0',
      description='Gaussian graphical models for scikit-learn.',
      author='Jason Laska and Manjari Narayan',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      url='https://github.com/jasonlaska/skggm',
      author_email='jlaska@gmail.com',
      cmdclass={'install': InstallWithSetup},
)