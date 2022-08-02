import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()


with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line
                    for line in requirements_file.read().splitlines()
                    if not line.startswith('#') and not line.startswith('-e ')]


with open('Readme.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="arpys",
    version="0.0.1",
    author="Kyle Gordon",
    author_email="kgord831@gmail.com",
    description="ARPES analysis with python and xarray",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
