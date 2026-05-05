.. index::
   single: Overlays
   single: Catalog data
   single: SPCAT
   single: XIAM
   single: Spectral comparison
   single: Theoretical spectra

Overlays
========

An overlay is an additional curve that is drawn on top of an FT plot for comparison with the experimental spectrum. Overlays are used to compare two experiments side-by-side, to display predicted line lists from spectroscopic fitting programs, and to import arbitrary XY data from other sources. Overlays are saved with the experiment and are restored automatically the next time the experiment is opened. The on-disk format is described on the :doc:`data_storage` page.

The Overlay Manager is opened from the squares-plus icon on the CP-FTMW toolbar. See :doc:`cp-ftmw` for the toolbar context.

.. image:: /_static/user_guide/overlays/overlay_types_comparison.png
   :width: 800
   :align: center
   :alt: Example showing different types of overlays on an FTMW spectrum


Overlay Types
.............

Blackchirp supports three overlay types, each backed by a different file format. The type is selected by the user when the overlay is created; the settings panel in the creation dialog adapts to the chosen type.

Blackchirp Experiment
---------------------

Loads the FT data from another Blackchirp experiment. The experiment metadata (LO frequency, shot counts, FT processing settings) is preserved, and the FT can be reprocessed using the standard processing controls. This is the most direct way to compare two experiments acquired under different conditions (e.g., discharge on vs. off, or different sample backing pressures).

Catalog
-------

Displays a stick spectrum or convolved lineshape from a spectroscopic fitting program. SPCAT and XIAM output formats are supported. Each transition retains its quantum numbers and source-program metadata, which are shown in the curve tooltip.

For comparison with experimental data, the stick spectrum can be convolved with a Lorentzian or Gaussian lineshape of user-defined FWHM. Convolution runs in a background thread; for large catalogs it may take a minute or more, and progress is reported in a cancellable dialog. Convolution results are cached so that repeating the same parameters returns immediately.

.. image:: /_static/user_guide/overlays/catalog_convolution_settings.png
   :width: 800
   :align: center
   :alt: Catalog overlay convolution settings dialog

The overlay's frequency range can be limited to a subset of the catalog. Restricting the range before convolution reduces both processing time and memory usage. For very large catalogs (>100,000 transitions), pre-filtering the file to the relevant frequency range is recommended.

Generic XY
----------

Loads arbitrary XY data from a delimited text file. Comma-, semicolon-, tab-, and space-separated formats are recognized, and a custom delimiter may be specified manually. The dialog displays the parsed file in a preview table so the column mapping can be verified before the overlay is created.

.. image:: /_static/user_guide/overlays/generic_xy_preview.png
   :width: 800
   :align: center
   :alt: Generic XY data preview showing parsed file contents

The X and Y columns are selected explicitly, the number of header lines to skip is configurable, and the X range can be filtered to a subset of the file. Numeric values must use the period as the decimal separator.


Creating an Overlay
...................

To create an overlay:

1. Click the ``Overlays`` button on the CP-FTMW toolbar.
2. In the Overlay Manager, click the ``Add`` button (plus icon). The unified overlay creation dialog opens.
3. Choose the overlay type. The settings panel updates to match the selection.
4. Click ``Browse`` and select a data file.
5. For Catalog overlays, configure the lineshape, linewidth, and frequency range. For Generic XY overlays, choose the X and Y columns, the number of header lines, and the X range. For Blackchirp Experiment overlays, the standard FT processing controls are available.
6. The plot updates with a live preview as settings change.
7. Click ``OK`` to create the overlay, or ``Cancel`` to discard it.

.. image:: /_static/user_guide/overlays/overlay_creation_dialog.png
   :width: 800
   :align: center
   :alt: Unified overlay creation dialog


Overlay Manager
...............

Once created, overlays appear in a table in the Overlay Manager. Each row corresponds to one overlay.

.. image:: /_static/user_guide/overlays/overlay_manager_main.png
   :align: center
   :alt: Overlay Manager interface

The columns are:

* ``Configure``: Gear icon. Opens the same dialog used during creation; changes apply on close.
* ``Enabled``: Checkbox. Toggles the overlay's visibility on the plot without deleting it.
* ``Label``: User-editable name for the overlay. Must be unique within the experiment. Edited via the Configure dialog. Special characters (including semicolons) are replaced with underscores when the label is written to disk.
* ``Plot``: Identifies which plot the overlay is drawn on. The same source can be added multiple times to display on different plots.
* ``Type``: Catalog, Generic XY, or Blackchirp Experiment.
* ``Comment``: Free-form notes. May not contain semicolons.

Right-clicking on a row provides additional actions:

* ``Copy Settings``: Copies the overlay's data and processing settings (Y scale, offset, frequency filtering) to the clipboard.
* ``Paste Settings``: Applies copied data settings to the selected overlay.
* ``Copy Appearance``: Copies the curve appearance (color, line style, thickness).
* ``Paste Appearance``: Applies copied appearance settings to the selected overlay.
* ``Remove``: Deletes the selected overlay(s).

Keyboard shortcuts are available for the copy/paste actions:

* ``Ctrl+Shift+C`` / ``Ctrl+Shift+V``: Copy/paste data settings.
* ``Ctrl+C`` / ``Ctrl+V``: Copy/paste appearance.
* ``Ctrl+Z``: Undo the most recent paste.


Troubleshooting
...............

If a catalog file is not recognized, verify that it is the unmodified output of a supported program (SPCAT or XIAM) and that the file is complete and UTF-8 encoded. To request support for an additional catalog format, file an issue on `GitHub <https://github.com/kncrabtree/blackchirp/issues>`_.

If a Generic XY file fails to parse or its columns are misaligned, set the delimiter manually rather than relying on auto-detection, verify that the column mapping in the preview table is correct, and confirm that the numeric values use periods as the decimal separator. Comment lines or partial header rows embedded within the data can cause silent column misalignment; they should be removed or the header line count should be increased.

If a Blackchirp Experiment overlay fails to load, confirm that the experiment directory contains a complete set of FID files and that the experiment number is reachable from the configured data storage location. Loading the experiment directly via :menuselection:`File --> View Experiment` first is a useful way to verify that the data is intact.
