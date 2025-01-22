# Spectrally regularised LVMs
![GitHub license](https://img.shields.io/github/license/RyanBalshaw/spectrally-regularised-LVMs)
![GitHub last commit](https://img.shields.io/github/last-commit/RyanBalshaw/spectrally-regularised-LVMs)
![PyPI](https://img.shields.io/pypi/v/spectrally-regularised-lvms)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/spectrally-regularised-lvms?color=blueviolet)
![Read the Docs](https://img.shields.io/readthedocs/spectrally-regularised-lvms?color=informational)
![GitHub issues](https://img.shields.io/github/issues/RyanBalshaw/spectrally-regularised-LVMs?color=critical)
[![DOI](https://zenodo.org/badge/633742820.svg)](https://doi.org/10.5281/zenodo.14717880)

*Spectrally-regularised-LVMs* is a Python-based [package](https://pypi.org/project/spectrally-regularised-lvms/) which facilitates the estimation of the linear latent variable model (LVM) parameters with a unique spectral regularisation term in single channel time-series applications.

## Purpose
LVMs are a statistical methodology which try to capture the underlying structure in some observed data. This package caters to single channel time-series applications and provides a methodology to estimate the LVM parameters. The model parameters are encouraged to capture non-duplicate information via a spectral regularisation term which penalises source duplication of the spectral information captured by the latent sources.

The purpose of this package is to provide a complete framework for LVMs with spectral regularisation that caters to a variety of LVM objective functions.

# Documentation
Please visit the [documentation](http://spectrally-regularised-lvms.readthedocs.io/) page for all supporting documentation for this package.

# Installation
The package is designed to be used through the Python API, and  can be installed using [pip](https://pypi.org/project/pip/):
```console
$ pip install spectrally-regularised-LVMs
```

A more detailed discussion regarding installation is given in the [documentation](http://spectrally-regularised-lvms.readthedocs.io/).

# Requirements

This package used Python ≥ 3.11 or later to run. For other python dependencies, please check the `pyproject.toml`
[file](https://github.com/RyanBalshaw/spectrally-regularised-LVMs/blob/main/pyproject.toml) included in this repository. The dependencies of this package are as follows:

|           Package                   	           | Version 	  |
|:-----------------------------------------------:|:----------:|
|    [Python](https://www.python.org/)      	     | ≥ 3.11  	  |
|      [Numpy](https://numpy.org/)         	      | ≥ 1.23.1 	 |
|   [Matplotlib](https://matplotlib.org/)    	    | ≥ 3.5.2 	  |
|      [SciPy](https://scipy.org/)         	      | ≥ 1.8.1 	  |
|  [scikit-learn](https://scikit-learn.org/)  	   | ≥ 1.1.2 	  |
|   [tqdm](https://github.com/tqdm/tqdm)     	    | ≥ 4.64.1 	 |
| [SymPy](https://www.sympy.org/en/index.html) 	  | ≥ 1.1.1 	  |
| [Poetry](https://python-poetry.org/) 	 | ≥ 2.0 	  |

# API usage
Please visit [the docs](http://spectrally-regularised-lvms.readthedocs.io/) for all supporting API documentation for this package.

# Contributing
This package uses [Poetry](https://python-poetry.org/) for dependency management and Python packaging and [git](https://git-scm.com/) for version control. To get started, first install git and Poetry. Then one may clone this repository via
```console
$ git clone git@github.com:RyanBalshaw/spectrally-regularised-LVMs.git
$ cd spectrally-regularised-LVMs
```

Then, install the necessary dependencies in a local environment via
```console
$ poetry install --with dev,docs
$ poetry shell
```

This will install all necessary package dependencies and activate the virtual environment. You can then set up the [pre-commit](https://pre-commit.com/) hooks via
```console
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

# License
This project is licensed under MIT License - see the [LICENSE](https://github.com/RyanBalshaw/spectrally-regularised-LVMs/blob/main/LICENSE) file for details.
