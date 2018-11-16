
from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name="kernel_tuner",
    version="0.2.0",
    author="Ben van Werkhoven",
    author_email="b.vanwerkhoven@esciencecenter.nl",
    description=("A simple CUDA/OpenCL kernel tuner in Python"),
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: System :: Distributed Computing',
        'Development Status :: 4 - Beta',
    ],
    install_requires=[
        'numpy>=1.13.3',
        'scipy>=0.18.1'],
    extras_require={
        'doc': ['sphinx', 'sphinx_rtd_theme', 'nbsphinx',
                'noodles', 'ipython'],
        'cuda': ['pycuda'],
        'opencl': ['pyopencl'],
        'cuda_opencl': ['pycuda', 'pyopencl'],
        'tutorial': ['jupyter', 'matplotlib', 'pandas'],
        'dev': [
            'numpy>=1.13.3', 'scipy>=0.18.1', 'mock>=2.0.0',
            'pytest>=3.0.3', 'Sphinx>=1.4.8',
            'sphinx-rtd-theme>=0.1.9', 'nbsphinx>=0.2.13',
            'jupyter>=1.0.0', 'matplotlib>=1.5.3', 'pandas>=0.19.1',
            'pylint>=1.7.1']
    },
)

