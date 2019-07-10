#!/usr/bin/env python
from setuptools import setup

REQUIRES = ['numpy', 'scipy', 'qutip']

setup(name='matsubara',
      version='0.1.4',
      description='Modelling the ultra-strongly coupled spin-boson model with unphysical modes',
      author='Neill Lambert, Shahnawaz Ahmed, Mauro Cirio, Franco Nori',
      author_email='shahnawaz.ahmed95@gmail.com',
      packages = ['matsubara'],
      requires = REQUIRES,
     )
