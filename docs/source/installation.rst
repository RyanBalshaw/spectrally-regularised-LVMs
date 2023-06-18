============
Installation
============

The spectrally-regularised-LVMs package has been coded in Python3 and has been tested on Linux and Windows operating systems. The package was originally written on a system that uses Windows, but makes use of the `os.path <https://docs.python.org/3/library/os.path.html#module-os.path>`_ module to ensure that any differences between operating system's is accounted for if anything is stored in a user's directory.

The package can either be installed via the PyPi package installer `pip <https://packaging.python.org/en/latest/tutorials/installing-packages/>`_, or by cloning the repository, creating a local `Poetry <https://python-poetry.org/>`_ environment, and then installing the package in `development mode <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_ via Poetry.

Pip installation
================

To use spectrally-regularised-LVMs within a virtual environment, the package can be installed using pip.

.. code-block:: console

   (.venv) $ pip install spectrally-regularised-LVMs

Cloning from Github
===================

You can clone the `GitHub repository <https://github.com/RyanBalshaw/spectrally-regularised-LVMs>`_ using git. It is recommended here that you first create a local Poetry environment

.. code-block:: console

    $ poetry init

More details on setting up a local Poetry evironment can be found `here <https://python-poetry.org/docs/basic-usage/>`_. Then, the package can be cloned from Github using git.

.. code-block:: console

    $ git clone git@github.com:RyanBalshaw/spectrally-regularised-LVMs.git

The repository can then be installed in the Poetry environment by using Poetry's `add <https://python-poetry.org/docs/cli/#add>`_ command.

.. code-block:: console

    $ poetry add ./spectrally-regularised-LVMs/
    $ poetry add --editable ./spectrally-regularised-LVMs/

Equivalently, the git clone step can be bypassed by directly referencing the Github repository when adding the dependency to the Poetry environment.

.. code-block:: console
    $ poetry add git+ssh://github.com/RyanBalshaw/spectrally-regularised-LVMs.git
    $ poetry add --editable git+ssh://github.com/RyanBalshaw/spectrally-regularised-LVMs.git

The ``--editable`` option can be used if you wish to install the package in editable mode.
