#!/usr/bin/env python3
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup(**generate_distutils_setup(
    packages=["piper_eye_to_hand", "piper_static_scene"],
    package_dir={"": "src"},
))
