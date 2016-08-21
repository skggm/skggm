"""Script to build pyquic from base dir.
"""
import os


class temp_cd():
    def __init__(self, temp_dir):
        self._temp_dir = temp_dir
        self._return_dir = os.path.dirname(os.path.realpath(__file__))
    def __enter__(self):
        os.chdir(self._temp_dir)
    def __exit__(self, type, value, traceback):
        os.chdir(self._return_dir)

def setup_pyquic():
    with temp_cd('inverse_covariance/pyquic'):
        os.system('make')
        
def clean_pyquic():
    with temp_cd('inverse_covariance/pyquic'):
        os.system('make clean')

if __name__ == "__main__":
    setup_pyquic()
