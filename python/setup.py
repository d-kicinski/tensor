import setuptools
from setuptools import Extension

extensions = []
libtensor = Extension(name="tensor.libtensor", sources=[])
extensions.append(libtensor)

setuptools.setup(
    ext_modules=extensions
)
