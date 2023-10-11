"""Configuration file for the Nox test runner.

This instantiates the specified sessions in isolated environments and runs the tests.
This allows for locally mirroring the testing occuring with GitHub-actions.
Be careful that the general setup of tests is left to pyproject.toml.
"""


import platform
from pathlib import Path

import nox
from nox_poetry import Session, session

# set the test parameters
python_versions_to_test = ["3.8", "3.9", "3.10", "3.11"]
nox.options.stop_on_first_error = True
nox.options.error_on_missing_interpreters = True

# set the default environment from the 'noxenv' file, if it exists
environment_file_path = Path("./noxenv.txt")
if environment_file_path.exists():
    env_values = ('none', 'virtualenv', 'conda', 'mamba', 'venv')  # from https://nox.thea.codes/en/stable/usage.html#changing-the-sessions-default-backend
    environment = environment_file_path.read_text()
    assert isinstance(environment, str), "File 'noxenv.txt' does not contain text"
    environment = environment.strip()
    assert environment in env_values, f"File 'noxenv.txt' contains {environment}, must be one of {','.join(env_values)}"
    nox.options.default_venv_backend = environment


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
    small_disk = False
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
            elif arg.lower() == "additional-tests":
                install_additional_tests = True
            elif arg.lower() == "small-disk":
                small_disk = True
            else:
                raise ValueError(f"Unrecognized argument {arg}")

    # check if there are optional dependencies that can not be installed
    if install_hip:
        if platform.system().lower() != 'linux':
            session.warn("HIP is only available on Linux, disabling dependency and tests")
            install_hip = False
    full_install = install_cuda and install_hip and install_opencl and install_additional_tests

    # if the user has a small disk, remove the other environment caches before each session is ran
    if small_disk:
        try:
            session_folder = session.name.replace('.', '*').strip()
            folders_to_delete: str = session.run(
                "find", "./.nox", "-mindepth", "1", "-maxdepth", "1", "-type", "d", "-not", "-name", session_folder,
                silent=True, external=True)
            folders_to_delete: list[str] = folders_to_delete.split('\n')
            for folder_to_delete in folders_to_delete:
                if len(folder_to_delete) > 0:
                    session.warn(f"Removing environment cache {folder_to_delete} because of 'small-disk' argument")
                    session.run("rm", "-rf", folder_to_delete, external=True)
        except Exception as error:
            session.warn("Could not delete Nox caching directories, reason:")
            session.warn(error)

    # remove temporary files leftover from the previous session
    session.run("rm", "-f", "temp_*.c", external=True)

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
    if install_opencl and session.python == "3.8":
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
        install_additional_warning = """
                Installation failed, this likely means that the required hardware or drivers are missing.
                Run without `-- additional-tests` to avoid this."""
        import re
        try:
            session.install("cuda-python")
        except Exception as error:
            print(error)
            session.warn(install_additional_warning)
        try:
            # use NVCC to get the CUDA version
            nvcc_output: str = session.run("nvcc", "--version", silent=True)
            nvcc_output = "".join(nvcc_output.splitlines())  # convert to single string for easier REGEX
            cuda_version = re.match(r"^.*release ([0-9]+.[0-9]+).*$", nvcc_output, flags=re.IGNORECASE).group(1).strip()
            session.warn(f"Detected CUDA version: {cuda_version}")
            try:
                try:
                    # based on the CUDA version, try installing the exact prebuilt cupy version
                    cuda_cupy_version = f"cupy-cuda{''.join(cuda_version.split('.'))}"
                    session.install(cuda_cupy_version)
                except Exception:
                    # if the exact prebuilt is not available, try the more general prebuilt
                    cuda_cupy_version_x = f"cupy-cuda{cuda_version.split('.')[0]}x"
                    session.warn(f"CuPy exact prebuilt not available for {cuda_version}, trying {cuda_cupy_version_x}")
                    session.install(cuda_cupy_version_x)
            except Exception:
                # if no compatible prebuilt wheel is found, try building CuPy ourselves
                session.warn(f"No prebuilt CuPy found for CUDA {cuda_version}, building from source...")
                session.install("cupy")
        except Exception as error:
            print(error)
            session.warn(install_additional_warning)

    # for the last Python version session if all optional dependencies are enabled:
    if session.python == python_versions_to_test[-1] and full_install:
        # run pytest on the package to generate the correct coverage report
        session.run("pytest")
    else:
        # for the other Python version sessions:
        # run pytest without coverage reporting
        session.run("pytest", "--no-cov")

    # warn if no coverage report
    if not full_install:
        session.warn("""
                     Tests ran successfully, but only a subset.
                     Coverage file not generated.
                     Run with 'additional-tests' and without 'skip-gpu', 'skip-cuda' etc. to avoid this.
                    """)
