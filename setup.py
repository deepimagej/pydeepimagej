try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
  name = 'pydeepimagej',
  packages = find_packages(),
  version = '1.0',   
  license = 'BSD 2-Clause License',   
  description = 'Python package to export tensorflow models as DeepImageJ bundled models',   
  author = 'C. Garcia-Lopez-de-Haro, E. Gomez-de-Mariscal, L. Donati, M. Unser, A. Munoz-Barrutia, D. Sage.',
  author_email = 'esgomezm@pa.uc3m.com, daniel.sage@epfl.ch, mamunozb@ing.uc3m.es',
  url = 'https://deepimagej.github.io/deepimagej/',
  download_url = 'https://github.com/deepimagej/pydeepimagej',
  keywords = ['Fiji', 'ImageJ', 'DeepImageJ', 'Deep Learning', 'Image processing'],  
  install_requires=[
	'numpy',
	'xml',
  	'time',
	'urllib',
	'shutil',
	'skimage',
    'tensorflow<=2.2'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Research',      
    'Topic :: Science/Engineering :: Image processing',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)