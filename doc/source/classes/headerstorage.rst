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

The two virtuals you must implement
-----------------------------------

Subclasses override two pure virtuals:

- ``storeValues()`` runs just before the header is written. Inside it,
  call ``store()`` once per scalar field and ``storeArrayValue()`` once
  per cell of any array fields. Each row is buffered in this object's
  cache.
- ``retrieveValues()`` runs after the header has been parsed and all
  matching rows have been routed to this object. Inside it, call
  ``retrieve()`` and ``retrieveArrayValue()`` to extract the cached
  values into your own members.

Each call to ``retrieve()`` or ``retrieveArrayValue()`` *removes* the
row from the cache. Asking for the same key twice yields the default
the second time. ``arrayStoreSize()`` reports how many entries an
array has before you start consuming it.

Composing a tree
----------------

A ``HeaderStorage`` may own children. Children are declared by
overriding ``prepareChildren()`` and calling ``addChild()`` once per
child:

.. code-block:: cpp

   void Experiment::prepareChildren()
   {
       addChild(ps_ftmwConfig.get());
       addChild(ps_validator.get());
       for(auto &p : d_hwCfgs)
           addChild(p.get());
       addChild(ps_lifCfg.get());
   }

The framework calls ``prepareChildren()`` at the start of every read
or write pass, after wiping any prior child list, so the tree always
reflects the current shape of the data. Children themselves do not
call ``addChild`` on their parent — the parent owns the relationship
in ``prepareChildren()``. The framework then recurses into each
child's ``prepareChildren()``, allowing arbitrary nesting.

``removeChild()`` exists for the rare case of a parent that needs to
detach a child mid-flight (``Experiment`` uses it when the user
disables an FTMW or LIF subsystem). It does not delete the child
object; ownership lives with whoever holds the smart pointer.

Write flow
----------

1. A caller invokes ``getStrings()`` on the root (``Experiment`` does
   this inside ``save()`` via ``BlackchirpCSV::writeHeader``).
2. Each node's ``prepareToStore()`` runs, which calls
   ``prepareChildren()`` and recurses into each child.
3. Each node's ``storeValues()`` runs, populating its cache via
   ``store()`` and ``storeArrayValue()``.
4. The framework converts cached entries to the six-column form,
   merges in each child's ``getStrings()`` output, and clears the
   in-memory cache.

Therefore: never call ``store()`` outside ``storeValues()``; the rows
would be cleared on the next write.

Read flow
---------

1. The caller (``Experiment``'s reading constructor) calls
   ``prepareToStore()`` on the root once, so the child tree is built.
2. The caller reads the CSV file line by line and hands each row to
   ``storeLine()`` on the root. Rows are dispatched to whichever
   node's ``d_headerKey`` matches column 0 (children searched first
   if the root does not match).
3. After all lines have been routed, the caller invokes
   ``readComplete()`` on the root, which calls ``retrieveValues()``
   on every node depth-first.
4. Each ``retrieveValues()`` implementation pulls rows back out via
   ``retrieve()`` / ``retrieveArrayValue()`` and assigns them to the
   object's members.

Therefore: never call ``retrieve()`` outside ``retrieveValues()`` (or
after ``readComplete()`` has run) — the cache is empty by then.

Enum cells
----------

``store()`` / ``storeArrayValue()`` template overloads wrap their
argument with ``QVariant::fromValue``, so a ``Q_ENUM``-registered
enumerator is written by name rather than as an opaque integer. On
the read side ``retrieve()`` and ``retrieveArrayValue()`` dispatch
to ``BC::CSV::enumFromVariant`` whenever the requested type carries
``Q_ENUM`` or ``Q_ENUM_NS``, so historical fixtures whose cells held
the integer form continue to round-trip back to the typed value
without subclasses having to call the helper directly. The dual-form
contract that motivates this is described under
:ref:`Enum cells: writing names, reading both
<persistence-enum-cells>`.

Object-key conventions
----------------------

- Singleton-style objects (``Experiment``, ``RfConfig``,
  ``ChirpConfig``, etc.) pass a constant key from their
  ``BC::Store::*`` namespace.
- Per-instance objects (``PulseGenConfig`` and other hardware
  configs) pass the hardware key for the specific instance
  (e.g. ``"PulseGenerator.main"``). This guarantees that experiments
  with several instances of the same hardware type produce
  distinguishable header rows.

The chosen key is stored in ``d_headerKey`` and cannot be changed
afterward.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: HeaderStorage
   :members:
   :protected-members:
   :undoc-members:
