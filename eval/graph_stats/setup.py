#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:03:15 2021

@author: yoann
"""

from distutils.core import setup, Extension

orca_module = Extension('orca',
                        sources=['orcamodule.cpp'],
                        extra_compile_args=['-std=c++11'], )

setup(name='orca',
      version='1.0',
      description='ORCA motif counting package',
      ext_modules=[orca_module])