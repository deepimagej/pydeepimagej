from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'pydeepimagej',
  packages = find_packages(),
  version = '1.0',   
  license = 'BSD 2-Clause License',   
  description = long_description,
  author = 'C. Garcia-Lopez-de-Haro, E. Gomez-de-Mariscal, L. Donati, M. Unser, A. Munoz-Barrutia, D. Sage.',
  author_email = 'esgomezm@pa.uc3m.com, daniel.sage@epfl.ch, mamunozb@ing.uc3m.es',
  url = 'https://deepimagej.github.io/deepimagej/',
  download_url = 'https://github.com/deepimagej/pydeepimagej',
  keywords = ['Fiji', 'ImageJ', 'DeepImageJ', 'Deep Learning', 'Image processing'],  
  python_requires='>=3.0',
  install_requires=[
	'numpy',
	'scikit-image',
    'tensorflow<=2.2'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Research',      
    'Topic :: Science/Engineering :: Image processing',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',
    ],
)