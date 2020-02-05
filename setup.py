from setuptools import setup
from setuptools import find_packages

# data_files = [('etc', ['etc/forgebox.cfg'])]

setup(
    name="forgebox",
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
        "tqdm>=4.25.0",
    ],
)
