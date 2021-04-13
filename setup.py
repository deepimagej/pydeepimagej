from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
# parse_requirements() returns generator of pip.req.InstallRequirement objects

setup(
    name='pydeepimagej',
    packages=find_packages(include=['pydeepimagej', 'pydeepimagej.*']),
    version='2.1.1',
    license='BSD 2-Clause License',
    description='Python package to export TensorFlow models as DeepImageJ bundled models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="C. Garcia-Lopez-de-Haro, E. Gomez-de-Mariscal, L. Donati, M. Unser, A. Munoz-Barrutia, D. Sage.",
    author_email='esgomezm@pa.uc3m.com, daniel.sage@epfl.ch, mamunozb@ing.uc3m.es',
    url='https://deepimagej.github.io/deepimagej/',
    download_url='https://github.com/deepimagej/pydeepimagej/archive/v1.0.0.tar.gz',
    keywords=['Fiji', 'ImageJ', 'deepImageJ', 'Deep Learning', 'Image processing', 'bioimage.io', 'BioImage Model Zoo'],
    python_requires='>=3.0',
    install_requires=[
        'numpy',
        'scikit-image==0.17.2',
        'ruamel.yaml',
        'datetime'
      ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
    ],
)
