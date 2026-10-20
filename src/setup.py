import io
import re
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent


def read(file_path):
    with io.open(ROOT / file_path, 'r', encoding='utf-8') as f:
        return f.read()


readme = read('../docs/source/index.md')
# clearing local versions
# https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers
requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$', read('requirements.txt'),
               flags=re.MULTILINE))

setup(
    # metadata
    name='remix',
    version='1.0.0',
    license='MIT',
    description='ReMix: gradient estimation for mixture distributions',
    long_description=readme,
    url='https://github.com/intsystems/remix',
    # options
    packages=find_packages(where=str(ROOT)),
    package_dir={'': str(ROOT)},
    install_requires=requirements.splitlines(),
)
