from setuptools import setup, find_packages

try:
    requirements = open("requirements.txt", "r").readlines()
except FileNotFoundError:
    requirements = []

setup(
    name='graphsage_benchmark',
    packages=find_packages(),
    install_requires=requirements
)