import os

from setuptools import find_packages
from setuptools import setup

version = '0.1a.dev'

install_requires = [
    'altair',
    'click',
    'newspaper3k',
    'pandas',
    'pandas',
    'requests',
    'streamlit',
    'tqdm',
    ]

test_requires = [
    'pytest',
    'pytest-cov',
    'pytest-flakes',
    'pytest-pep8',
    ]


here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst'), encoding='utf-8').read()
except IOError:
    README = ''


setup(name='twitter-discovery',
      version=version,
      description='Twitter Discovery lets you search articles referenced '
      'from your tweets, retweets, and favorites.',
      long_description=README,
      url='https://github.com/naturalvision/twitter-discovery',
      author='Daniel Nouri',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      extras_require={
          'testing': test_requires,
          'all': install_requires + test_requires,
          },
      entry_points={
          'console_scripts': [
              'twitter-discovery = twidis.main:cli',
              ],
          },
      )
