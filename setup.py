#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="jqrtc",
    version="0.1.0",
    description="Python/NumPy port of quadrotor tracking control simulation",
    author="Python port by Serrano, original MATLAB by wjxjmj",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
)
