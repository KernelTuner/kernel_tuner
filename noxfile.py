"""Configuration file for the Nox test runner.

This instantiates the specified sessions in isolated environments and runs the tests.
This allows for locally mirroring the testing occuring with GitHub-actions.
Be careful that the general setup of tests is left to pyproject.toml.
"""


import platform

import nox

# from nox import Session, session
from nox_poetry import Session, session

python_versions_to_test = ["3.8", "3.9", "3.10", "3.11"]
nox.options.stop_on_first_error = True
nox.options.error_on_missing_interpreters = True

# @nox.session
# def lint(session: nox.Session) -> None:
#     """Ensure the code is formatted as expected."""
#     session.install("ruff")
#     session.run("ruff", "--format=github", "--config=pyproject.toml", ".")


# @session  # uncomment this line to only run on the current python interpreter
@session(python=python_versions_to_test)  # missing versions can be installed with `pyenv install ...`
# do not forget check / set the versions with `pyenv global`, or `pyenv local` in case of virtual environment
def tests(session: Session) -> None:
    """Run the tests for the specified Python versions."""
    # check if optional dependencies have been disabled by user arguments (e.g. `nox -- skip-gpu`, `nox -- skip-cuda`)
    install_cuda = True
    install_hip = True
    install_opencl = True
    install_additional_tests = False
    if session.posargs:
        for arg in session.posargs:
            if arg.lower() == "skip-gpu":
                install_cuda = False
                install_hip = False
                install_opencl = False
                break
            elif arg.lower() == "skip-cuda":
                install_cuda = False
            elif arg.lower() == "skip-hip":
                install_hip = False
            elif arg.lower() == "skip-opencl":
                install_opencl = False
            else:
                raise ValueError(f"Unrecognized argument {arg}")

            if arg.lower() == "additional_tests":
                install_additional_tests = True

    # check if there are optional dependencies that can not be installed
    if install_hip:
        if platform.system().lower() != 'linux':
            session.warn("HIP is only available on Linux, disabling dependency and tests")
            install_hip = False

    # set extra arguments based on optional dependencies
    extras_args = []
    if install_cuda:
        extras_args.extend(["-E", "cuda"])
    if install_hip:
        extras_args.extend(["-E", "hip"])
    if install_opencl:
        extras_args.extend(["-E", "opencl"])

    # separately install optional dependencies with weird dependencies / build process
    install_warning = """Installation failed, this likely means that the required hardware or drivers are missing.
                  Run with `-- skip-gpu` or one of the more specific options (e.g. `-- skip-cuda`) to avoid this."""
    if install_cuda:
        # if we need to install the CUDA extras, first install pycuda seperately.
        #   since version 2022.2 it has `oldest-supported-numpy` as a build dependency which doesn't work with Poetry
        try:
            session.install("pycuda")       # Attention: if changed, check `pycuda` in pyproject.toml as well
        except Exception as error:
            print(error)
            session.warn(install_warning)
    if install_opencl and (session.python == "3.7" or session.python == "3.8"):
        # if we need to install the OpenCL extras, first install pyopencl seperately.
        # it has `oldest-supported-numpy` as a build dependency which doesn't work with Poetry, but only for Python<3.9
        try:
            session.install("pyopencl")       # Attention: if changed, check `pyopencl` in pyproject.toml as well
        except Exception as error:
            print(error)
            session.warn(install_warning)

    # finally, install the dependencies, optional dependencies and the package itself
    try:
        session.run_always("poetry", "install", "--with", "test", *extras_args, external=True)
    except Exception as error:
        session.warn(install_warning)
        raise error

    # if applicable, install the dependencies for additional tests
    if install_additional_tests and install_cuda:
        install_additional_warning = """Installation failed, this likely means that the required hardware or drivers are missing.
                  Run without `-- additional_tests` to avoid this."""
        try:
            session.install("cuda-python")
        except Exception as error:
            print(error)
            session.warn(install_additional_warning)
        try:
            cuda_version = session.run("nvcc", "--version", "|", "sed", "-n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p'", silent=True)
            print(f"CUDA version: {cuda_version}")
            if cuda_version[:3] == "12.":
                session.install("cupy-cuda12x")
            elif cuda_version[:3] == "11.":
                session.install("cupy-cuda11x")
            else:
                session.install("cupy")
        except Exception as error:
            print(error)
            session.warn(install_additional_warning)

    # for the last Python version session if all optional dependencies are enabled:
    if session.python == python_versions_to_test[-1] and install_cuda and install_hip and install_opencl:
        # run pytest on the package to generate the correct coverage report
        session.run("pytest")
    else:
        # for the other Python version sessions:
        # run pytest without coverage reporting
        session.run("pytest", "--no-cov")

    # report if no coverage report
    if not (install_cuda and install_hip and install_opencl):
        session.warn("Tests ran successfully, but only a subset. Coverage file not generated.")
