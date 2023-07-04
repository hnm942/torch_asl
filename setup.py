from setuptools import setup, find_packages

setup(
    name='models',
    version='1.0.0',
    author='huynm12',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas'
    ],
)
