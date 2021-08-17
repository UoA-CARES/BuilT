#!/usr/bin/python
# -*- coding: utf8 -*-
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), "r", encoding="utf-8") as fh:
    long_description = fh.read()    

setup(
    name="BuilT",
    version="0.0.3",
    author="JongYoon Lim",
    author_email="jy.lim@auckland.ac.nz",
    description="Easily build your trainer for DNNs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UoA-CARES/BuilT",
    project_urls={
        "Bug Tracker": "https://github.com/UoA-CARES/BuilT/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
)