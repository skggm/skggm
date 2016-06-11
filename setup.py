'''
Setup script that:

/pyquic:
    - flattens pyquic module 
    - compiles pyquic
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
    with temp_cd('pyquic'):
        for filename in os.listdir('py_quic'):
            if filename.startswith('.'):
                continue

            filepath = os.path.join('py_quic', filename)
            shutil.move(filepath, '.')
            
        shutil.rmtree('py_quic')
        os.system('make')


if __name__ == "__main__":
    setup_pyquic()