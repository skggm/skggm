import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import platform


if platform.system() == 'Darwin':
    extra_compile_args = ['-I/System/Library/Frameworks/vecLib.framework/Headers']
    if 'ppc' in platform.machine():
        extra_compile_args.append('-faltivec')
        
    ext_modules = [Extension(
        name="pyquic",
        sources=["QUIC.C", "pyquic.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=["-Wl,-framework", "-Wl,Accelerate"],
        language="c++"
        )]
else:
    ext_modules = [Extension(
        name="pyquic",
        sources=["QUIC.C", "pyquic.pyx"],
        include_dirs = [numpy.get_include(), "/usr/local/include"],
        extra_compile_args=['-msse2', '-O2', '-fPIC', '-w'],
        extra_link_args=["-llapack"],
        language="c++"
        )]

setup(
    name = "pyquic",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
    )
