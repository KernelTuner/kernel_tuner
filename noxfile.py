"""Configuration file for the Nox test runner.

This instantiates the specified sessions in isolated environments and runs the tests.
This allows for locally mirroring the testing occuring with GitHub-actions.
Be careful that the general setup of tests is left to pyproject.toml.
"""


import nox

# from nox_poetry import Session, session   # nox_poetry is a better option, but <=1.0.3 has a bug with filename-URLs

python_versions_to_test = ["3.8", "3.9", "3.10", "3.11"]
nox.options.stop_on_first_error = True
nox.options.error_on_missing_interpreters = True

# @nox.session
# def lint(session: nox.Session) -> None:
#     """Ensure the code is formatted as expected."""
#     session.install("ruff")
#     session.run("ruff", "--format=github", "--config=pyproject.toml", ".")


# @nox.session  # uncomment this line to only run on the current python interpreter
@nox.session(python=python_versions_to_test)  # missing versions can be installed with `pyenv install ...`
# do not forget check / set the versions with `pyenv global`, or `pyenv local` in case of virtual environment
def tests(session: nox.Session) -> None:
    """Run the tests for the specified Python versions."""
    session.install("poetry")
    session.run("poetry", "install", "--with", "test", external=True)

    # for the last Python version session:
    if session.python == python_versions_to_test[-1]:
        # run pytest on the package without C-extensions to generate the correct coverage report
        session.run("pytest")
    else:
        # for the other Python version sessions:
        # run pytest without coverage reporting
        session.run("pytest", "--no-cov")
