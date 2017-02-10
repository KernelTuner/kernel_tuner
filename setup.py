import os
from setuptools import setup

import kernel_tuner
from kernel_tuner import interface

setup(
    name = "kernel_tuner",
    version = "0.1.1",
    author = "Ben van Werkhoven",
    author_email = "b.vanwerkhoven@esciencecenter.nl",
    description = ("A simple CUDA/OpenCL kernel tuner in Python"),
    license = "Apache 2.0",
    keywords = "auto-tuning gpu computing pycuda cuda pyopencl opencl",
    url = "http://benvanwerkhoven.github.io/kernel_tuner/",
    packages=['kernel_tuner', 'kernel_tuner.runners'],
    long_description=interface.__doc__,
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

