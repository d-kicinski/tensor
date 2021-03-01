import setuptools
from setuptools import Extension

extensions = []
_torch = Extension(name="tensor.libtensor", sources=[])
extensions.append(_torch)

setuptools.setup(
    ext_modules=extensions
)
