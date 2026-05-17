.. index::
   single: ClickableLabel
   single: label; clickable
   single: folder link; status bar
   single: QLabel; folder link

ClickableLabel
==============

``ClickableLabel`` is a ``QLabel`` subclass whose rendered text acts as
a link to a folder. When it carries a non-empty folder path, hovering
the text shows a hand cursor and underlines it, and a left click opens
that folder in the system file manager (via
``QDesktopServices::openUrl``). It backs the data-path label in the
main window status bar and the experiment-number labels in the FTMW and
LIF view widgets, which link to the experiment's storage directory.

The hit and hover target is only the bounding rectangle of the rendered
text — computed from the non-underlined font and honoring the label's
alignment — not the full widget width. A centered label stretched
across a tab therefore does not swallow clicks aimed elsewhere. An
empty path makes the widget an ordinary, non-interactive label, so a
single label can be switched between the active and inert states at
runtime (for example a numbered experiment versus a Peak-Up acquisition
that has no stored data).

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: ClickableLabel
   :members:
   :undoc-members:
