# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Small

None.

## Medium

### ZoomPanPlot worker-thread snapshot for `filterData`

`ZoomPanPlot::filterData` runs on a `QtConcurrent` worker and reads
`QwtPlot::itemList()` (which walks an internal `QwtPlotDict`) and per-curve
state directly from the shared plot. `QwtPlot` does not document this access
path as thread-safe, and the worker has no protection against the UI thread
calling `attach()` / `detach()` on plot items mid-filter — the destructor and
`resetPlot()` already serialize via `waitForFilterComplete()`, but external
callers (e.g., `FtmwViewWidget` rebuilding curves on data load) do not.

The robust fix is for the UI thread to snapshot the work the worker needs —
a list of `(curve, xAxis, yAxis, canvasMap, autoscale-flag)` tuples — under
`p_mutex` before `QtConcurrent::run` is invoked, then have the worker iterate
the snapshot rather than calling `itemList()`. Curve attach/detach also needs
to take `p_mutex` so the snapshot and the live list stay coherent — the
cleanest path is overriding `QwtPlot`-level protected hooks (which are not
virtual in the public API) or wrapping curve construction/destruction in a
`ZoomPanPlot`-managed registry.

The shipped partial fix is `resetPlot()` calling `waitForFilterComplete()`
before `detachItems()`, which closes the most common failure path. External
classes that mutate the item list while a filter pass may be in flight are
the remaining risk.

## Large

None.

## Pre-Release

### [Documentation Revision](documentation-revision.md)

The sphinx/breathe documentation is outdated and needs to be updated for the
`cmakemigration` branch. The goals are:

- Improve the readme and program summary for the documentation landing page and Github README.
- Update the changelog with major feature developments; establish best practices for keeping it updated
- Prepare a feature summary/migration guide for version 1.x -> 2.0.0.
- Update the user guide to provide a walkthrough of major program features and use-cases
- Maintain a hardware catalog of C++ drivers/capabilities
- Create a developer's guide to explain the overall code structure, conventions, major
data classes, and guides for adding new hardware and implementations
- Provide an API reference for the most important classes for developers. Specifically,
these should be classes like SettingsStorage, HardwareObject, etc that are used
throughout the code. These classes should have Doxygen-style annotations in headers for
autogeneration with breathe.

### [Packaging and Binary Generation (Github Actions)](packaging-and-ci.md)

CMake-side work is complete: `cmake/Packaging.cmake` produces release-only DEB,
RPM, DMG, NSIS, and TGZ/ZIP packages via CPack; `cmake/QtDeployment.cmake` wires
`windeployqt`/`macdeployqt` as install hooks; and `.github/workflows/release.yml`
defines five jobs (linux-deb, linux-rpm, linux-appimage, macos-dmg, windows-nsis)
triggered by `workflow_dispatch` and `release: published`. **Remaining work is
testing and verification**: dispatch the workflow per-platform, iterate on
first-run issues (likely candidates: linuxdeploy library paths, macdeployqt's
Qwt resolution, windeployqt's runtime DLLs), and confirm the resulting packages
launch on clean VMs. See `packaging-and-ci.md` for the full strategy reference,
file map, and acceptance criteria.
