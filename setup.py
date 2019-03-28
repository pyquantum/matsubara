#!/usr/bin/env python
from setuptools import setup


REQUIRES = ['numpy', 'scipy', 'qutip']

setup(name='matsubara',
      version='0.1',
      description='Virtual excitations in the ultra-strongly-coupled spin-boson model',
      author='Shahnawaz Ahmed, Neill Lambert, Mauro Cirio',
      author_email='shahnawaz.ahmed95@gmail.com',
      packages = ['matsubara'],
      requires = REQUIRES,
     )
