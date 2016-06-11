'''
Setup script that:

/pyquic:
    - compiles pyquic
    - copies py_quic into base directory so that we can use the module directly
'''
import os
import shutil


class temp_cd():
    def __init__(self, temp_dir):
        self._temp_dir = temp_dir
        self._return_dir = os.path.dirname(os.path.realpath(__file__))
    def __enter__(self):
        os.chdir(self._temp_dir)
    def __exit__(self, type, value, traceback):
        os.chdir(self._return_dir)

def setup_pyquic():
    with temp_cd('pyquic/py_quic'):
        os.system('make')

    shutil.copytree('pyquic/py_quic', 'py_quic')
        
def clean_pyquic():
    shutil.rmtree('py_quic')
    os.system('git submodule update --checkout --remote -f')

if __name__ == "__main__":
    setup_pyquic()