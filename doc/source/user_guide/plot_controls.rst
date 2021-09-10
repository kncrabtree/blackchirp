Plot Controls
=============

All of the plots in Blackchirp share a common set of zoom/pan controls and customization options.
Each plot can be configured individually, and the most recently-used settings should be recalled each time the program is started.
You can configure the appearance of each curve, change which vertical axis a curve is plotted against, control the appearance of a plot grid, and more.

Zooming and Panning
-------------------

Zooming is accomplished by using the mouse wheel.
Scrolling up zooms in, and scrolling down zooms out.
The zoom limits are determined by the X and Y range spanned by the data.
By default, zooming in or out will affect the X axis and both Y axes at the same time.
This behavior can be changed by pressing keys while scrolling:

- ``Ctrl``: Zoom X axis only; Y axes are fixed.
- ``Shift``: Zoom both Y axes, keeping X axis fixed.
- ``Meta`` (Windows key): Zoom left axis only.
- ``Alt``: Zoom right axis only.

To zoom out immediately to the full range of the data, press ``Ctrl`` and left-click the plot.
Alternatively, you can right-click the plot and choose the ``Autoscale`` option.

Panning the plot is accomplished by middle-clicking anywhere on the plot and dragging.
Like with zooming, the panning range is limited to the range of the data displayed on the plot, so panning is not possible when the plot is at full scale.

Plot Configuration Options
--------------------------

The right-click context menu contains a variety of customization options to control the appearance and behavior of the plot as a whole.

