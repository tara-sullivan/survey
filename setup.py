import glob

from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages

setup(
    name='survey',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(i))[0] for i in glob.glob('src/*.py')],
    include_package_data=True,
)
