Plot Controls
=============

All of the plots in Blackchirp share a common set of zoom/pan controls and customization options.
Each plot can be configured individually, and the most recently-used settings should be recalled each time the program is started.
You can configure the appearance of each curve, change which vertical axis a curve is plotted against, control the appearance of a plot grid, and more.

Zooming and Panning
-------------------

Zooming is accomplished by using the mouse wheel or by left-clicking and dragging a rectangle on the plot.
While dragging a rectangle, press the ``z`` key to cancel.
With the mouse wheel, scrolling up zooms in, and scrolling down zooms out.
It is currently not possible to zoom out using the rectangle dragging method.
By default, zooming will affect the X axis and both Y axes at the same time.
This behavior can be changed using modifier keys:

- ``Ctrl``: Zoom X axis only; Y axes are fixed.
- ``Shift``: Zoom both Y axes, keeping X axis fixed.
- ``Meta`` (Windows key): Zoom left axis only (only for scroll zooming).
- ``Alt``: Zoom right axis only (only for scroll zooming).

To zoom out immediately to the full range of the data, double-click the plot or press the ``Home`` key.
Alternatively, you can right-click the plot and choose the ``Autoscale`` option.
The zoom limits are determined by the X and Y range spanned by the data.

Panning the plot is accomplished by middle-clicking anywhere on the plot and dragging.
Like with zooming, the panning range is limited to the range of the data displayed on the plot, so panning is not possible when the plot is at full scale.

Keyboard controls are also available for zooming and panning:

- Arrow keys: Step the plot in the direction pressed by 50% of the scale width.
- ``Alt`` + Arrow keys: Step the plot in the direction pressed by 10% of the scale width.
- ``Shift`` + ``Up/Down``: Zoom in/out vertically by 10%. By default, the zoom is symmetric about 0.0 (but see the ``Y Center?`` setting below).
- ``Ctrl`` + ``Up/Down``: Zoom in/out vertically by 50%. By default, the zoom is symmetric about 0.0.
- ``Shift`` + ``Right/Left``: Zoom in/out horizontally by 10%. The zoom is symmetric about the center of the plot.
- ``Ctrl`` + ``Right/Left``: Zoom in/out horizontally by 50%. The zoom is symmetric about the center of the plot.


Plot Configuration Options
--------------------------

.. image:: /_static/user_guide/plot_controls/contextmenu.png
   :width: 400
   :align: center
   :alt: Plot context menu

The right-click context menu contains a variety of customization options to control the appearance and behavior of the plot as a whole.

- ``Autoscale``: resets the X and Y axes to show the full range of the data on the plot.
- ``Zoom Settings``:
   - ``Y Center?``: Toggles whether zooming with the ``Up/Down`` arrows is symmetric about the plot center (checked) or about 0.0 (unchecked).
   - ``Wheel Zoom Factors``: Sets the zooming speed while scrolling the mouse wheel for a given axis. Larger numbers will zoom by a greater factor per mouse wheel step. The default value for each axis is 0.1.
- ``Tracker``: Enable the tracker to display the coordinates of the mouse cursor on the plot. For each axis, you can configure the number of decimals displayed and, optionally, switch to scientific notation if desired.
- ``Grid``: Configure the appearance of major and minor gridlines. It is possible to control the line style and color for each type of gridline.
- ``Curves``: Options for configuring the appearance of curves on the plot. More details are below.

Additionally, on the Rolling and Aux Data plots, two additional options are available.

- ``Push X Axis``: Set the X scale of all plots to match the selected plot.
- ``Autoscale All``: Apply the autoscale action to all plots.

Curve Configuration Options
---------------------------

As shown in the image above, each curve that is displayed on the plot can be individually configured. Under the ``Curves`` context menu entry, a submenu is available for each curve with the following options:

- ``Export XY``: Generate a csv file containing the currently displayed data.
- ``Color...``: Change the color of the curve.
- ``Line Width``: Change the thickness of the line drawn for the curve.
- ``Line Style``: Change the style (solid, dashed, etc) of the line. Set to "None" if no line is desired.
- ``Marker``: Change the plot marker used for each data point. Set to "None" if no marker is desired.
- ``Marker Size``: Change the size of the marker.
- ``Visible``: Control whether the curve is displayed. If a curve is not visible, it is not included in the autoscale calculation.
- ``Y Axis``: Change which Y axis the curve is plotted on.
- ``Change Plot``: Rolling/Aux Data only. Move the curve to a different plot. The plots are numbered from left to right, then top to bottom.


