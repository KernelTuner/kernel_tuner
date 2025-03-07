"""Configuration file for the Nox test runner.

This instantiates the specified sessions in isolated environments and runs the tests.
This allows for locally mirroring the testing occuring with GitHub-actions.
Be careful that the general setup of tests is left to pyproject.toml.
"""


import platform
import re
from pathlib import Path

import nox
from nox_poetry import Session, session

# set the test parameters
verbose = False
python_versions_to_test = ["3.10", "3.11", "3.12", "3.13"] # 3.14 has not yet been officially released so is not tested against, but is allowed by the pyproject.toml
nox.options.stop_on_first_error = True
nox.options.error_on_missing_interpreters = True
nox.options.default_venv_backend = 'virtualenv'

# workspace level settings
settings_file_path = Path("./noxsettings.toml")
venvbackend_values = ('none', 'virtualenv', 'conda', 'mamba', 'venv')  # from https://nox.thea.codes/en/stable/usage.html#changing-the-sessions-default-backend

# TODO remove this from a session function, session is only needed to receive trigger argument
@session    # to only run on the current python interpreter
def create_settings(session: Session) -> None:
    """One-time creation of noxsettings.toml."""
    arg_trigger = False
    if session.posargs:
        # check if the trigger argument was used
        arg_trigger = any(arg.lower() == "create-settings-file" for arg in session.posargs)
    # create settings file if the trigger is used or old settings exist
    noxenv_file_path = Path("./noxenv.txt")
    if arg_trigger or (noxenv_file_path.exists() and not settings_file_path.exists()):
        # default values
        venvbackend = nox.options.default_venv_backend
        envdir = ""
        # conversion from old notenv.txt
        if noxenv_file_path.exists(): 
            venvbackend = noxenv_file_path.read_text().strip()
            noxenv_file_path.unlink()
        # write the settings
        assert venvbackend in venvbackend_values, f"{venvbackend=}, must be one of {','.join(venvbackend_values)}"
        settings = (f'venvbackend = "{venvbackend}"\n'
                    f'envdir = "{envdir}"\n')
        settings_file_path.write_text(settings)
        # exit to make sure the user checks the settings are correct
        if arg_trigger:
            session.warn(f"Settings file '{settings_file_path}' created, exiting. Please check settings are correct before running Nox again.")
            exit(1)

# obtain workspace level settings from the 'noxsettings.toml' file
if settings_file_path.exists():
    with settings_file_path.open(mode="rb") as fp:
        import tomli
        nox_settings = tomli.load(fp)
        venvbackend = nox_settings['venvbackend']
        envdir = nox_settings['envdir']
        assert venvbackend in venvbackend_values, f"File '{settings_file_path}' has {venvbackend=}, must be one of {','.join(venvbackend_values)}"
        nox.options.default_venv_backend = venvbackend
        nox.options.venvbackend = venvbackend
        if envdir is not None and len(envdir) > 0:
            nox.options.envdir = envdir

# @session    # to only run on the current python interpreter
# def lint(session: Session) -> None:
#     """Ensure the code is formatted as expected."""
#     session.install("ruff")
#     session.run("ruff", "--output-format=github", "--config=pyproject.toml", ".")

@session    # to only run on the current python interpreter
def check_poetry(session: Session) -> None:
    """Check whether Poetry is correctly configured."""
    session.run("poetry", "check", "--no-interaction", external=True)

@session    # to only run on the current python interpreter
def check_development_environment(session: Session) -> None:
    """Check whether the development environment is up to date with the dependencies, and try to update if necessary."""
    if session.posargs:
        if 'github-action' in session.posargs:
            session.log("Skipping development environment check on the GitHub Actions runner, as this is always up to date.")
            return None
    output: str = session.run("poetry", "install", "--sync", "--dry-run", "--with", "test", silent=True, external=True)
    match = re.search(r"Package operations: (\d+) (?:install|installs), (\d+) (?:update|updates), (\d+) (?:removal|removals), \d+ skipped", output)
    assert match is not None, f"Could not check development environment, reason: {output}"
    groups = match.groups()
    installs, updates, removals = int(groups[0]), int(groups[1]), int(groups[2])
    if installs > 0 or updates > 0:
        # packages = re.findall(r"• Installing .* | • Updating .*", output, flags=re.MULTILINE)
        # assert packages is not None
        session.warn(f"""
            Your development environment is out of date ({installs} installs, {updates} updates). 
            Update with 'poetry install --sync', using '--with' and '-E' for optional dependencies, extras respectively.
            Note: {removals} packages are not in the specification (i.e. installed manually) and may be removed.
            To preview changes, run 'poetry install --sync --dry-run' (with optional dependencies and extras).""")

@session(python=python_versions_to_test)  # missing versions can be installed with `pyenv install ...`
# do not forget check / set the versions with `pyenv global`, or `pyenv local` in case of virtual environment
def tests(session: Session) -> None:
    """Run the tests for the specified Python versions."""
    session.log(f"Testing on Python {session.python}")
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
        # use NVCC to get the CUDA version
        import re
        nvcc_output: str = session.run("nvcc", "--version", silent=True, external=True)
        nvcc_output = "".join(nvcc_output.splitlines())  # convert to single string for easier REGEX
        cuda_version = re.match(r"^.*release ([0-9]+.[0-9]+).*$", nvcc_output, flags=re.IGNORECASE).group(1).strip()
        session.warn(f"Detected CUDA version: {cuda_version}")
        # if we need to install the CUDA extras, first install pycuda seperately, reason:
        #   since version 2022.2 it has `oldest-supported-numpy` as a build dependency which doesn't work with Poetry
        if " not found: " in session.run("pip", "show", "pycuda", external=True, silent=True, success_codes=[0,1]):
            # if PyCUDA is not installed, install it
            session.warn("PyCUDA not installed")
            try:
                session.install("pycuda", "--no-cache-dir", "--force-reinstall") # Attention: if changed, check `pycuda` in pyproject.toml as well
            except Exception as error:
                session.log(error)
                session.warn(install_warning)
        else:
            session.warn("PyCUDA installed")
            # if PyCUDA is already installed, check whether the CUDA version PyCUDA was installed with matches the current CUDA version
            session.install("numpy")    # required by pycuda.driver
            pycuda_version = session.run("python", "-c", "import pycuda.driver as drv; drv.init(); print('.'.join(list(str(d) for d in drv.get_version())))", silent=True)
            shortest_string, longest_string = (pycuda_version, cuda_version) if len(pycuda_version) < len(cuda_version) else (cuda_version, pycuda_version)
            if longest_string[:len(shortest_string)] != shortest_string:
                session.warn(f"PyCUDA was compiled with a version of CUDA ({pycuda_version}) that does not match the current version ({cuda_version}). Re-installing.")
                try:
                    session.install("pycuda", "--no-cache-dir", "--force-reinstall")  # Attention: if changed, check `pycuda` in pyproject.toml as well
                except Exception as error:
                    session.log(error)
                    session.warn(install_warning)

    # finally, install the dependencies, optional dependencies and the package itself
    poetry_env = Path(session.run_always("poetry", "env", "info", "--executable", silent=not verbose, external=True).splitlines()[-1].strip()).resolve()
    session_env = Path(session.bin, "python/").resolve()
    assert poetry_env.exists(), f"{poetry_env=} does not exist"
    assert session_env.exists(), f"{session_env=} does not exist"
    # if the poetry virtualenv is not set to the session env, use requirements file export instead of Poetry install
    if poetry_env != session_env:
        session.warn(f"Poetry env ({str(poetry_env)}) is not session env ({str(session_env)}), falling back to install via requirements export")
        requirements_file = Path(f"tmp_test_requirements_{session.name}.txt")
        if requirements_file.exists():
            requirements_file.unlink()
        if verbose:
            print(session.run_always('conda', 'list'))
        session.run_always('poetry', 'export', '-f', 'requirements.txt', '-o', requirements_file.name, '--with=test', '--without-hashes', *extras_args, external=True, silent=not verbose)
        session.install('-r', requirements_file.name)
        session.install('.')
        requirements_file.unlink()
        if verbose:
            print(session.run_always('conda', 'list'))
    else:
        try:
            session.run_always("poetry", "install", "--with", "test", *extras_args, external=True, silent=False)
        except Exception as error:
            session.warn(install_warning)
            raise error

    # if applicable, install the dependencies for additional tests
    if install_additional_tests and install_cuda:
        install_additional_warning = """
                Installation failed, this likely means that the required hardware or drivers are missing.
                Run without `-- additional-tests` to avoid this."""
        # install cuda-python
        try:
            session.install("cuda-python")
        except Exception as error:
            session.log(error)
            session.warn(install_additional_warning)
        # install cupy
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

    # for the last Python version session if all optional dependencies are enabled:
    if session.python == python_versions_to_test[-1] and full_install:
        # run pytest on the package to generate the correct coverage report
        session.run("pytest", external=False)
    else:
        # for the other Python version sessions:
        # run pytest without coverage reporting
        session.run("pytest", "--no-cov", external=False)

    # warn if no coverage report
    if not full_install:
        session.warn("""
                     Tests ran successfully, but only a subset.
                     Coverage file not generated.
                     Run with 'additional-tests' and without 'skip-gpu', 'skip-cuda' etc. to avoid this.
                    """)
