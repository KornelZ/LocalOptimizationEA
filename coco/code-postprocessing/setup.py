#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import setuptools
from distutils.core import setup

_name = 'cocopp'
setup(
    name = _name,
    version = "2.2.1.10",
    packages = [_name, _name + '.comp2', _name + '.compall'],
    package_dir = {_name: 'cocopp'},
    package_data={_name: ['*enchmarkshortinfos.txt',
                          '*enchmarkinfos.txt',
                          'best*algentries*.pickle',
                          'best*algentries*.pickle.gz',
                          'refalgs/best*.tar.gz',
                          'pprldistr2009*.pickle.gz',
                          'latex_commands_for_html.html',
    # this is not supposed to copy to the subfolder, see https://docs.python.org/2/distutils/setupscript.html
    # but it does. 
                          'js/*', 'tth/*', 
                          '../latex-templates/*.tex',
                          '../latex-templates/*.cls',
                          '../latex-templates/*.sty',
                          '../latex-templates/*.bib',
                          '../latex-templates/*.bst',
                          ]},
    url = 'https://github.com/numbbo/coco',
    license = 'BSD',
    maintainer = 'Dejan Tusar',
    maintainer_email = 'dejan.tusar@inria.fr',
    # author = ['Nikolaus Hansen', 'Raymond Ros', 'Dejan Tusar'],
    description = 'Benchmarking framework for all types of black-box optimization algorithms, postprocessing. ',
    long_description = '...',
    # install_requires = ['numpy>=1.7'],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Environment :: ??',
        'Intended Audience :: ??',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: ??'
    ]
)
