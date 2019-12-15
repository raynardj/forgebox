from setuptools import setup
from setuptools import find_packages

# data_files = [('etc', ['etc/forgebox.cfg'])]

setup(
    name="ai-forge",
    version="0.1.0",
    author="raynardj",
    author_email="raynard@rasenn.com",
    description="Tools for managing machine learning",
    packages = find_packages(),
    include_package_data=True,
    py_modules=['forgebox',],
    # scripts = ['forge/bin/forge', 'forge/bin/forge_db',],
    # package_data={'forge':['./forge/templates/*','./forge/static/*']},
    install_requires = [
        "pandas>=0.18.0",
        "tqdm>=4.25.0",
    ],
)
