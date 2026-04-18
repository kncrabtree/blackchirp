# HwStatusBox Updates

This project has 2 main purposes:

1. Make HwStatusBox display more compact.
2. Improve user experience by showing more data in tooltips and clickable targets to
   quickly open associated configuration dialogs.
   
## Goal 1: QSpinBox/QDoubleSpinBox -> QLabel

Numerical data is currently displayed in read-only QSpinBox and QDoubleSpinBox, which
takes up considerable space. Replace these with QLabels. Use std::to_chars with general
format to configure display, replacing the "eXX" suffix on scientific notation with
" x 10^XX" (see ScientificSpinBox::applySuperscript). Consider extracting some utility
functions from ScientificSpinBox into bcglobals.cpp or another appropriate file for
reuse. Preserve suffixes that are currently in QSpinBox/QDoubleSpinBox and append to
QLabel instead.

## Goal 2: UX Improvements

HwStatusBox (and other sections shown on the left panel of the MainWindow) are intended
to provide at-a-glance information about hardware status, program status, experiment
progress, etc. It is important to surface key information and hide inactive/redundant
information in order to keep the display compact, but it is also desirable for the user
to be able to access more detailed status data via well-considered tooltips. Then,
the user should be able to easily open the associated HwDialog (or other associated
dialog) directly from the status panel without having to navigate menus. Consider using
a "configure" icon from src/resources/icons as clickable target.

What is **not** in scope: direct control of hardware from a status box. Control is
consolidated in the Control widgets embedded in the HwDialog. 

## Notes on Implementation

Apply principles above if the class is not mentioned here. Most other classes should
be straightforward to implement.

### Experiment information display

- Experiment number QSpinBox -> QLabel
- Add (possibly truncated) display of save path; tooltip shows full path
- Add clickable target that opens ApplicationConfigManager
- Consider moving to place next to progress bars?

### ClockDisplayBox

- QDoubleSpinBox -> QLabel for each clock
- Hide rows for clocks that are not configured
- Tooltip shows physical clock and output number
- Clickable target for opening RfConfigDialog
- Clickable target per clock for opening HwDialog for that clock

### PulseStatusBox

- Do not hide inactive channels; Led shows enabled/disabled status
- Rep rate/triggered display already correct.
- Mouseover on Led or name label shows popup with syncCh, delay, width, active level. If
  channel is Duty Cycle mode (yellow Led), show on/off values. Format should be
  tabular.
- Clickable target for opening HwDialog

### FlowStatusBox

- QDoubleSpinBox -> QLabel.
- Hide inactive channels (Led off). Consider moving to 2 channels per column for flow
  channels, with pressure always on its own line. Pressure never hidden.
- Mouseover on channel name label, numeric label, or Led shows setpoint (including pressure)
- Clickable target for opening HwDialog
