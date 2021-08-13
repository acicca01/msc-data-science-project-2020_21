import os
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MSc",
    version="0.0.1",
    author="Antonello Ciccarone",
    description="MSc Project code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Birkbeck/msc-data-science-project-2020_21---files-acicca01",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OSx",
    ],
    packages=find_packages(include=['hRAC','hRAC.*']),
    python_requires=">=3.7",
) 
