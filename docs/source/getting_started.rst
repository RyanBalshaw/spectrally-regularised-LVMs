Getting Started
===============

.. _installation:

Installation
------------

To use spectrally-constrained-LVMs, first install it using pip:

.. code-block:: console

   (.venv) $ pip install spectrally-constrained-LVMs

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``spectrally-constrained-lvms.Hankel_matrix()`` function:

.. autofunction:: spectrally-constrained-lvms.helper_methods.Hankel_matrix

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``.
For example:

>>> import spectrally_constrained_LVMs as scLVMs
>>> scLVMs.linear_model()
