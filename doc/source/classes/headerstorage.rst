HeaderStorage
=============

``HeaderStorage`` is the base class for any object that contributes
fields to an experiment's CSV header file. The header is the
human-readable, semicolon-delimited record of every parameter that
defined an acquisition: hardware settings, RF and chirp configuration,
digitizer setup, flow setpoints, validation thresholds, and so on. The
file uses six columns — object key, array key, array index, key, value,
unit — and ``HeaderStorage`` packs values into that schema on the way
out and unpacks them back when an experiment is loaded from disk.

``Experiment`` is the root of a tree of ``HeaderStorage`` nodes, with
``FtmwConfig``, ``RfConfig``, the digitizer, the pulse generator,
the LIF config, and the validator as children, each of which may add
its own grandchildren. The framework dispatches incoming rows to the
correct subtree by matching the object key in column 0, and on the
write side it walks the tree depth-first to produce the full set of
rows. The on-disk layout of an experiment, including the header file,
is described in :doc:`/user_guide/data_storage`.

The class-level Doxygen documentation below contains the full
developer how-to: which two virtuals to override, how to declare
children, the read/write call sequences, and the conventions for
choosing an object key.

.. highlight:: cpp

.. doxygenclass:: HeaderStorage
   :members:
   :protected-members:
   :undoc-members:
