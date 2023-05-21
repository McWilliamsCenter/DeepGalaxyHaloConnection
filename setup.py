#!/usr/bin/env python
import io
import os

from setuptools import find_packages
from setuptools import setup

setup(
    name="deepghc",
    packages=find_packages(),
    install_requires=["jax", "jaxlib"],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
