# Hardware Dialog Left Panel Navigation

## Current State

Clicking a hardware type item in the left panel (the `QListWidget` / tree of hardware types)
has no effect — the left panel is display-only. Selection does not update the middle panel
(radio button group) or right panel (profile list).

## Fix

Connect the left panel's `currentItemChanged` (or `itemClicked`) signal to a slot that calls
`selectHardwareType(type)` — the same function the middle panel radio buttons already call to
populate the right panel. The middle panel radio button for the clicked type should also be
checked programmatically so the two panels stay in sync.
