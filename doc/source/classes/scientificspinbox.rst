.. index::
   single: ScientificSpinBox
   single: spin box; scientific notation
   single: DisplayMode; ScientificSpinBox
   single: StepMode; ScientificSpinBox
   single: input; floating-point

ScientificSpinBox
=================

``ScientificSpinBox`` is a ``QAbstractSpinBox`` subclass for entering and displaying
double-precision floating-point values in either fixed or scientific notation.  It is
used throughout Blackchirp wherever a numeric field must accept values that span many
orders of magnitude — RF frequencies, timing delays, voltage amplitudes, and similar
quantities.

The widget enforces an 18-character input ceiling (16 significant digits plus a decimal
point and a sign), validates input with a ``QDoubleValidator`` that accepts scientific
notation, and carries an optional unit suffix (e.g. ``" MHz"``) that is appended to the
displayed text but stripped before parsing.

Display modes
-------------

The ``DisplayMode`` enum controls how the value is formatted when the widget is not
being edited:

- **Auto** — uses fixed-point notation when the absolute value falls in ``[1e-6, 1e6)``;
  switches to scientific notation outside that range.  This is the default.
- **Fixed** — always fixed-point, regardless of magnitude.
- **Scientific** — always scientific notation, rendered with superscript exponents in
  the display text.

The user can change the active mode at runtime via the right-click context menu.  The
mode can also be set programmatically with ``setDisplayMode()``.

When the widget receives focus for editing, the text switches to standard ``printf``-style
scientific notation (``1.234567e+03``) regardless of the display mode, and the full
content is selected so the user can type a replacement value immediately.  When focus is
lost the edited text is parsed, the value is clamped to ``[minimum(), maximum()]``, and
the display text is regenerated in the current display mode.

Step modes
----------

The ``StepMode`` enum controls how large each increment or decrement step is:

- **Adaptive** (default) — the step size tracks the place value of the least-significant
  digit shown in the display.  For a fixed-mode value of ``1234.56`` the step is ``0.01``;
  for a scientific-mode value of ``1.23456e+03`` with four decimal places in the mantissa
  the step is ``0.1`` (i.e. ``1e+03 × 10^{-4}``).  The nominal ``singleStep()`` value is
  not used in Adaptive mode.
- **Fixed** — every step adds or subtracts the constant returned by ``fixedStepSize()``.
  Set both with ``setFixedStepSize()`` and ``setStepMode(StepMode::Fixed)``.

In either mode, holding **Ctrl** while stepping changes behavior:

- **Adaptive mode with Ctrl** — doubles the value on up-step and halves it on down-step
  (``value × 2^steps``).  If the value is zero, Ctrl-up sets it to ``1.0`` and
  Ctrl-down sets it to ``-1.0``.
- **Fixed mode with Ctrl** — multiplies ``fixedStepSize()`` by 10 for that step.

Suffixes
--------

A unit suffix set with ``setSuffix()`` is appended to every display string and stripped
before any parse operation.  Setting a suffix adjusts the line-edit ``maxLength``
to ``MAX_INPUT_LENGTH + suffix.length()`` so the suffix never displaces input characters.

Precision
---------

By default (``displayPrecision() == -1``) the widget infers precision from the text the
user typed.  Call ``setDisplayPrecision(n)`` to fix the number of decimal places; pass
``-1`` to restore automatic inference.  The precision spinner in the context menu exposes
the same control interactively.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: ScientificSpinBox
   :members:
   :protected-members:
   :undoc-members:
