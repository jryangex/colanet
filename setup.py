#!/usr/bin/env python
from setuptools import find_packages, setup

__version__ = "0.4.0"

if __name__ == '__main__':
    setup(
        name='colanet',
        version=__version__,
        description='Fast Object Detection',
        url='https://github.com/jryangex/colanet',
        author='colaYang',
        author_email='m10815q07@gapps.ntust.edu.tw',
        keywords='deep learning',
        packages=find_packages(exclude=('config', 'tools', 'demo')),
        classifiers=[
            'Development Status :: Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        zip_safe=False)
