Usage
=====

.. _installation:

Installation
------------

To use spectrally-constrained-LVMs, first install it using pip:

.. code-block:: console

   (.venv) $ pip install spectrally-constrained-LVMs

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``scLVM.get_random_ingredients()`` function:

.. autofunction:: scLVM.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import spectrally_constrained_LVMs as scLVMs
>>> scLVMs.linear_model()
