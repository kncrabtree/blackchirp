.. index::
   single: ThemeColors
   single: ColorRole
   single: theme; color management
   single: theme; dark mode
   single: icon; SVG theming
   single: accessibility; WCAG contrast

ThemeColors
===========

``ThemeColors`` is an all-static utility class that centralizes color
management for Blackchirp's user interface. Every color it returns is derived
from the active system palette and adjusted so that status indicators, accent
text, and SVG icons remain readable under both light and dark system themes.
It is the right starting point for any new UI surface that requires a
theme-respecting accent color or a status-feedback color.

Color roles
-----------

The ``ColorRole`` enum divides the available colors into three families:

**Status colors** — used for feedback and state indicators:

- ``StatusSuccess`` — valid states and success messages.
- ``StatusWarning`` — caution states and warning messages.
- ``StatusError`` — invalid states and error messages.
- ``StatusInfo`` — informational or neutral-positive messages.
- ``StatusNeutral`` — default or unclassified status.

**Text colors** — used for emphasis levels in labels and tooltips:

- ``SubtleText`` — secondary or less-important text.
- ``EmphasisText`` — highlighted or important text.
- ``DisabledText`` — inactive or disabled text. Also used as the default
  color role for the disabled state in ``createThemedIconWithStates()``.

**Icon colors** — used when rendering SVG icons:

- ``IconPrimary`` — main interface element icons.
- ``IconSecondary`` — supporting or secondary icons.
- ``IconAccent`` — special-highlight icons.

Palette-derived colors and CSS
------------------------------

``getThemeAwareColor(role, widget)`` returns a ``QColor`` appropriate for the
current theme. Pass a widget pointer to use that widget's palette for context;
pass ``nullptr`` to fall back to the application palette. ``getCSSColor()``
returns the same color as a hex string (e.g., ``"#1a9e3f"``) suitable for use
in Qt stylesheets.

``isDarkTheme(widget)`` inspects the palette and returns ``true`` when a dark
system theme is active. UI code that renders custom graphics can use this to
choose appropriate base colors before calling ``getThemeAwareColor()``.

Contrast helpers
----------------

``ensureContrast(color, background, minContrastRatio)`` adjusts ``color``
until its WCAG contrast ratio against ``background`` meets
``minContrastRatio``. The default target of 4.5 corresponds to WCAG AA
compliance for normal-weight text. ``calculateContrastRatio(color1, color2)``
performs the raw calculation; the ratio ranges from 1.0 (identical colors) to
21.0 (black on white).

SVG icon helpers
----------------

``createThemedIcon(svgPath, colorRole, widget)`` loads an SVG from a Qt
resource path and recolors it using the color for ``colorRole``, returning a
``QIcon`` sized for the platform's icon metrics.
``createThemedIconWithStates(svgPath, enabledRole, disabledRole, widget)``
produces a ``QIcon`` with separate colors for enabled and disabled states,
which Qt uses automatically when a toolbar button or menu action is disabled.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: ThemeColors
   :members:
   :undoc-members:
