import sys
from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


if sys.version_info[0] >= 3:
    pynvml = 'nvidia-ml-py3'
else:
    pynvml = 'nvidia-ml-py'

setup(
    name="kernel_tuner",
    version="0.4.1",
    author="Ben van Werkhoven",
    author_email="b.vanwerkhoven@esciencecenter.nl",
    description=("An easy to use CUDA/OpenCL kernel tuner in Python"),
    license="Apache 2.0",
    keywords="auto-tuning gpu computing pycuda cuda pyopencl opencl",
    url="http://benvanwerkhoven.github.io/kernel_tuner/",
    packages=['kernel_tuner', 'kernel_tuner.runners', 'kernel_tuner.strategies'],
    long_description=readme(),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: System :: Distributed Computing',
        'Development Status :: 5 - Production/Stable',
    ],
    install_requires=['numpy>=1.13.3', 'scipy>=0.18.1', 'jsonschema'],
    extras_require={
        'doc': ['sphinx', 'sphinx_rtd_theme', 'nbsphinx', 'pytest', 'ipython'],
        'cuda': ['pycuda', pynvml],
        'opencl': ['pyopencl'],
        'cuda_opencl': ['pycuda', 'pyopencl'],
        'tutorial': ['jupyter', 'matplotlib', 'pandas'],
        'dev': [
            'numpy>=1.13.3', 'scipy>=0.18.1', 'mock>=2.0.0', 'pytest>=3.0.3', 'Sphinx>=1.4.8', 'scikit-learn>=0.24.2', 'scikit-optimize>=0.8.1',
            'sphinx-rtd-theme>=0.1.9', 'nbsphinx>=0.2.13', 'jupyter>=1.0.0', 'matplotlib>=1.5.3', 'pandas>=0.19.1', 'pylint>=1.7.1',
            'bayesian-optimization>=1.0.1'
        ]
    },
)
