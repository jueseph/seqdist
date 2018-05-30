#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
  name     = "seqdist",
  author   = "Jue Wang",
  author_email = "juewang@post.harvard.edu",
  packages = ['seqdist'],
  entry_points = {
    'console_scripts': ['seqdist = seqdist:main']
  },
  setup_requires = [
    'biopython',
    'pandas',
    'numpy',
    'scipy',
  ],
  version  = 0.1
)
