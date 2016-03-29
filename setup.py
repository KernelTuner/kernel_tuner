import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "kernel_tuner",
    version = "0.0.1",
    author = "Ben van Werkhoven",
    author_email = "b.vanwerkhoven@esciencecenter.nl",
    description = ("A simple CUDA kernel tuner in Python"),
    license = "Apache 2.0",
    keywords = "auto-tuning pycuda cuda gpu",
    url = "http://benvanwerkhoven.github.io/kernel_tuner/",
    packages=['kernel_tuner'],
    long_description=read('README.md'),
    classifiers=[
        'Topic :: System :: Distributed Computing',
        'Development Status :: 2 - Pre-Alpha',
    ],
)

