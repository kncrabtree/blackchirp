.. index::
   single: EnumComboBox
   single: combo box; enum-driven
   single: Q_ENUM; combo box population

EnumComboBox
============

``EnumComboBox<T>`` is a ``QComboBox`` subclass that auto-populates its items from a
``Q_ENUM``-registered enumeration at construction time.  It is used in Blackchirp
wherever a combo box must reflect a scoped or unscoped enum whose enumerators should
be presented to the user without a hand-written population loop.

The constructor calls ``QMetaEnum::fromType<T>()`` to obtain the enum's metadata, iterates
over every enumerator, and adds one item per enumerator.  The display label is the
enumerator's string name with underscores replaced by spaces.  The integer value of the
enumerator is stored as each item's ``itemData`` so that ``currentValue()`` and
``setCurrentValue()`` can work without needing to know index positions.

The type parameter ``T`` must be declared with ``Q_ENUM`` (inside a ``QObject``-derived
class) or ``Q_ENUM_NS`` (in a namespace).  If ``QMetaEnum::fromType<T>()`` returns an
invalid meta-enum — which happens when ``T`` has no ``Q_ENUM`` registration — the
constructor adds no items and the combo box is empty.

Item access
-----------

Beyond the standard ``QComboBox`` index-based API, ``EnumComboBox`` provides two
methods that return the underlying ``QStandardItem*`` for a given entry.  These are
useful when individual items need to be disabled (grayed out) or given a custom
foreground color:

- ``itemForValue(T v)`` — looks up the item by enum value.
- ``itemAt(int i)`` — looks up the item by row index.

Both return ``nullptr`` when the requested entry does not exist or when the combo box
model is not a ``QStandardItemModel``.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: EnumComboBox
   :members:
   :undoc-members:
