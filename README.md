# Spectrally constrained LVMs

![GitHub](https://img.shields.io/github/license/RyanBalshaw/spectrally-constrained-LVMs)
![GitHub issues](https://img.shields.io/github/issues-raw/RyanBalshaw/spectrally-constrained-LVMs)
![PyPI](https://img.shields.io/pypi/v/spectrally-constrained-lvms)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/spectrally-constrained-lvms?color=blueviolet)
![GitHub last commit](https://img.shields.io/github/last-commit/RyanBalshaw/spectrally-constrained-LVMs)
[![Documentation Status](https://readthedocs.org/projects/spectrally-constrained-lvms/badge/?version=latest)](https://spectrally-constrained-lvms.readthedocs.io/en/latest/?badge=latest)

[//]: # (![Read the Docs]&#40;https://img.shields.io/readthedocs/spectrally-constrained-lvms&#41;)

*Current version:* 0.1.1

Spectrally-constrained-LVMs is a Python-based package which facilitates the estimation of the linear latent variable model (LVM) parameters with a unique spectral constraint in single-channel time-series applications.

## Purpose
LVMs are a statistical methodology which tries to capture the underlying structure in some observed data. This package caters to single channel time-series applications and provides a methodology to estimate the LVM parameters. The model parameters are encouraged to be diverse via a spectral constraint which enforces non-duplication of the spectral information captured by the latent sources.

The purpose of this package is to provide a complete methodology that caters to a variety of LVM objective functions.

# Documentation
Please visit [the docs](http://spectrally-constrained-lvms.readthedocs.io/) for all supporting documentation for this package.

# Installation
The package is designed to be used through the Python API, and  can be installed using [pip](https://pypi.org/project/pip/):
```console
$ pip install spectrally-constrained-LVMs
```

# Requirements

This package used Python ≥ 3.10 or later to run. For other python dependencies, please check the `pyproject.toml`
[file](https://github.com/RyanBalshaw/spectrally-constrained-LVMs/blob/main/pyproject.toml) included in this repository. The dependencies of this package are as follows:

|          Package                   	           | Version 	  |
|:----------------------------------------------:|:----------:|
|    [Python](https://www.python.org/)      	    | ≥ 3.10  	  |
|     [Numpy](https://numpy.org/)         	      | ≥ 1.23.1 	 |
|   [Matplotlib](https://matplotlib.org/)    	   | ≥ 3.5.2 	  |
|     [SciPy](https://scipy.org/)         	      | ≥ 1.8.1 	  |
|  [scikit-learn](https://scikit-learn.org/)  	  | ≥ 1.1.2 	  |
|   [tqdm](https://github.com/tqdm/tqdm)     	   | ≥ 4.64.1 	 |
| [SymPy](https://www.sympy.org/en/index.html) 	 | ≥ 1.1.1 	  |

# API usage

## Model parameter estimation
A generic example is shown below:
```python
import spectrally_constrained_LVMs as scLVMs

# Load in some data
signal_data = ... # Load a single channel time-series signal
Fs = ... # Sampling frequency of the data

# Hankelise the data
X = scLVMs.hankel_matrix(signal_data,
                          Lw = 512,
                          Lsft = 1)

# Define a cost function for latent sources with maximum variance
cost_inst = scLVMs.VarianceCost()

# Define a model instance
model_inst = scLVMs.LinearModel(n_sources=10,
                                 cost_instance=cost_inst,
                                 whiten=False,
                                 alpha_reg=1.0)

# Estimate the model parameters
model_inst.fit(X, Fs = Fs)
```

## Cost function implementation
This package allows users to implement their own objective functions. Two examples are shown here.

### Method one - user defined

This method allows users to implement their objective function and all required higher order derivatives manually. This is demonstrated through:
```python
import numpy as np
import spectrally_constrained_LVMs as scLVMs

# Define objective function (maximise source variance)
def cost(X, w, y):

    return -1 * np.mean((X @ w) ** 2, axis=0) # Framework performs minimisation

# Define gradient vector
def grad(X, w, y):

    return -2 * np.mean(y * X, axis=0, keepdims=True).T

# Define Hessian matrix
def hess(X, w, y):

    return -2 * np.cov(X, rowvar=False)

# Initialise the cost instance
user_cost = scLVMs.UserCost(use_hessian = True)

# Define the objective function, gradient and Hessian
user_cost.set_cost(cost)
user_cost.set_gradient(grad)
user_cost.set_hessian(hess)

# Check the implementation
X_ = np.random.randn(1000, 16)
w_ = np.random.randn(16, 1)
y_ = X_ @ w_

res_grad = user_cost.check_gradient(X_, w_, y_,step_size = 1e-4)
res_hess = user_cost.check_hessian(X_, w_, y_,step_size = 1e-4)
```

### Method two - SymPy defined
Users can also use [SymPy](https://www.sympy.org/en/index.html) to implement their objective function, which allows for all higher order derivatives to be obtained symbolically. An example of this is given through
```python
import sympy as sp
import numpy as np
import spectrally_constrained_LVMs as scLVMs

n_samples = 1000 # Fix the number of samples in the data
n_features = 16 # Fix the number of features

# Initialise the cost function instance
user_cost = scLVMs.SympyCost(n_samples, n_features, use_hessian=True)

# Get the SymPy representations of the model parameters
X_sp, w_sp, iter_params = user_cost.get_model_parameters()
i, j = iter_params

# Calculate the objective function (maximise source variance)
loss_i = sp.Sum(w_sp[j, 0] * X_sp[i, j], (j, 0, n_features - 1))
loss = -1 / n_samples * sp.Sum(loss_i**2, (i, 0, n_samples - 1))

# Set the properties within the instance
user_cost.set_cost(loss)

# Use SymPy to calculate the first and second order derivatives
user_cost.implement_methods()

# Check the implementation
X_ = np.random.randn(n_samples, n_features)
w_ = np.random.randn(n_features, 1)
y_ = X_ @ w_

res_grad = user_cost.check_gradient(X_, w_, y_,step_size = 1e-4)
res_hess = user_cost.check_hessian(X_, w_, y_,step_size = 1e-4)
```

# Contributing
This package uses [Poetry](https://python-poetry.org/) for dependency management and Python packaging and [git](https://git-scm.com/) for version control. To get started, first install git and Poetry and then clone this repository via
```console
$ git clone git@github.com:RyanBalshaw/spectrally-constrained-LVMs.git
$ cd spectrally-constrained-LVMs
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
This project is licensed under MIT License - see the [LICENSE](https://github.com/RyanBalshaw/spectrally-constrained-LVMs/blob/main/LICENSE) file for details.
