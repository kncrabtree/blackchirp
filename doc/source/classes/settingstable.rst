.. index::
   single: SettingsTable
   single: settings table; two-column
   single: section row; checkable
   single: QTableWidget; settings idiom
   single: UI; settings layout

SettingsTable
=============

``SettingsTable`` is a ``QTableWidget`` subclass that renders a compact,
borderless two-column "Setting / Value" grid. It is the standard
building block for Blackchirp's configuration surfaces: the hardware
settings widgets, the overlay-configuration panels, the LIF and FTMW
view-dock panels, the experiment-type setup pages, and the
:doc:`/classes/zoompanplot` curve-appearance editor all build their
forms by appending rows to one of these tables rather than hand-rolling
a ``QGridLayout`` of nested ``QGroupBox`` and ``QFormLayout`` blocks.

The table is non-selectable, read-only at the item level (value widgets
are still interactive), sizes itself to its contents, and never shows a
vertical scrollbar. Section heading bands are shaded with
:doc:`/classes/themecolors` so they track the active light or dark
palette. A new configuration UI should reach for ``SettingsTable``
before composing form layouts by hand; doing so keeps spacing, the
borderless contract, and the heading-band styling consistent across
every settings surface.

Row types
---------

Rows are appended in document order and come in three flavors:

- **Value rows** (``addSettingRow``) — a left-column label and a value
  widget in the right column. The two-widget overload places a pair of
  widgets side by side in the value cell (for example a line edit and a
  browse button, or a checkbox and a spin box); a horizontally
  expanding widget fills the cell while a fixed one keeps its hint.
- **Section rows** (``addSectionRow``) — a bold, centered,
  theme-shaded band spanning both columns. It replaces the title of a
  former ``QGroupBox`` without nesting another frame.
- **Checkable section rows** (``addCheckableSectionRow``) — a section
  band with a leading checkbox. Value rows bound to it with
  ``bindSectionRows`` collapse (via ``setRowHidden``) when the box is
  unchecked and reappear when it is checked, reproducing a
  checkable-group-box without the frame.

Checkable sections
------------------

A checkable section row is flexible enough that a single row can stand
in for several states of the form it replaces:

- ``setSectionCheckable`` swaps the leading checkbox for a plain
  centered heading (and back) without destroying the underlying
  ``QCheckBox`` — so signal/slot connections and the bound-row wiring
  survive the mode change.
- ``setBoundRowsEnabled`` greys a section's rows out in place, the
  disabled counterpart of the hide-on-uncheck collapse.
- ``setSectionVisible`` shows or hides a whole section as a unit, and
  is collapse-aware so a plain container section can wrap nested
  collapsible sub-sections without fighting their state.
- ``sectionCheckBox`` returns the backing checkbox so a caller can
  drive or observe the collapse directly.

The first user-initiated expand of a section that started collapsed
grows the enclosing window by exactly the revealed rows' height so the
new rows are not clipped behind the suppressed scrollbar; the growth
happens once and is never reversed, so repeated toggling does not creep
the window size.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: SettingsTable
   :members:
   :undoc-members:
