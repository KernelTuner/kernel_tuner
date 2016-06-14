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
    keywords = "auto-tuning gpu pycuda cuda pyopencl opencl",
    url = "http://benvanwerkhoven.github.io/kernel_tuner/",
    packages=['kernel_tuner'],
    long_description=read('README.md'),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: System :: Distributed Computing',
        'Development Status :: 4 - Beta',
    ],
)

