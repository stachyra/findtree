========
findtree
========

Cluster-Based Object Segmentation
---------------------------------

This package holds updated versions of two files that I originally posted as a response_ to a stackoverflow question, challenging users to solve a `toy problem`_ in image object segmentation.

.. _response: https://stackoverflow.com/questions/20772893/how-to-detect-a-christmas-tree/20850922#20850922
.. _toy problem: https://en.wikipedia.org/wiki/Toy_problem

Contents
--------

The repository comprises only two files:

    ``findtree.py``: Given an RGB image containing a single, prominent Christmas tree, returns a list of line segments that may be used to draw a bounding polygon around the tree

    ``testfindtree.py``: Runs ``findtree.py`` on 6 example input images that were originally supplied with the question, and produces several plots to help illustrate how the object segmentation algorithm works

Dependencies
------------

Runs on Python 3.x.  There are five open source external dependencies, all of which are widely used across the Python community:

    - Pillow_
    - numpy_
    - scipy_
    - matplotlib_
    - scikit-learn_

.. _Pillow:                    https://pillow.readthedocs.io/
.. _numpy:                     https://www.numpy.org/
.. _scipy:                     https://www.scipy.org/
.. _matplotlib:                https://matplotlib.org/
.. _scikit-learn:              https://scikit-learn.org/

Install
-------

This isn't really a proper package per se, just some demo code, so it's not meant to be installed.  To use, simply download both files to a directory, and run the test file.  E.g., within ipython_, you may do::

	In [1]: %run testfindtree

and the demo will produce several plots.

.. _ipython:                   https://ipython.org/
