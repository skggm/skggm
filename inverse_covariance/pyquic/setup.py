import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import platform
import os


if platform.system() == 'Darwin':
    extra_compile_args = [
        '-I/System/Library/Frameworks/vecLib.framework/Headers'
    ]
    if 'ppc' in platform.machine():
        extra_compile_args.append('-faltivec')

    extra_link_args = ["-Wl,-framework", "-Wl,Accelerate"]
    include_dirs = [numpy.get_include()]

else:
    if "MKLROOT" in os.environ:
        include_dirs = [numpy.get_include(), "/usr/local/include", 
        os.path.join(os.environ.get('MKLROOT'),'include')]
    else:
        include_dirs = [numpy.get_include(), "/usr/local/include"]
    extra_compile_args = ['-msse2', '-O2', '-fPIC', '-w']
    if "MKLROOT" in os.environ:
        extra_link_args = ["-L" + os.environ.get('MKLROOT') + 
        " -lmkl_def -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_sequential" + \
        " -lmkl_core -lmkl_gnu_thread -lmkl_lapack95_lp64"]
    else:
        extra_link_args = ["-llapack"]


ext_module = Extension(
    name="pyquic",
    sources=["QUIC.C", "pyquic.pyx"],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++"
)

if __name__ == "__main__":
    setup(
        name="pyquic",
        cmdclass={"build_ext": build_ext},
        ext_modules=[ext_module]
    )
