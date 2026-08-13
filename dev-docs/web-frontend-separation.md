# Web frontend separation — core engine + protocol + web client

Status: exploratory (planning only). No code committed. This document
scopes what a migration away from the QWidget/Qwt UI toward a
server–client architecture with a web (HTML/JS) frontend would require
of the existing C++ codebase. It is a map of the coupling that would
have to be re-cut, not a committed plan or a schedule.

## Motivation

UI work in the current stack is slow and inflexible for reasons that
are intrinsic to the QWidget/Qwt/C++ toolchain, not to Blackchirp:
every layout tweak is imperative C++, every change is an edit–compile–
relaunch cycle, and the visual vocabulary is bounded by what QWidget +
Qwt render. The appeal of a web frontend is **development velocity and
UI polish** — declarative layout, hot reload, mature styling — and the
freedom to redesign the experience rather than preserve the current
one. Exact reproduction of the existing UI (the dockable inner
`QMainWindow`, the per-curve appearance/preset editors, the context-menu
system) is explicitly *not* a goal; those are among the parts most
worth rethinking.

Performance is not a motivation. The current plotting stack is already
fast on large records for the right reason — server-side min/max
decimation (`ZoomPanPlot::_kickoffFilterPass` →
`BCEvenSpacedCurveBase::_filter`, closed-form for uniformly-spaced
FID/FT data). A web client relying on the *same* decimation would not
render faster; it would render the same decimated point sets. The plot
math is an asset to carry over, not a bottleneck to escape.

The target that fits these goals is a **headless core engine + web
client** split (an embedded HTTP/WebSocket server in the C++ core, a
browser or thin webview as the frontend), not an embedded QtWebEngine
view. The embedded-Chromium route keeps the frontend coupled to the Qt
build/release cadence and delivers the least of the velocity gain; the
server route decouples the two halves fully and yields remote access to
a running acquisition as a side effect. Its cost is that Blackchirp
becomes a client–server application, with the lifecycle, connection-
state, and offline-packaging concerns that implies.

## Current architecture — what already resembles a server

Blackchirp is already a producer/consumer system in which background
worker threads own the live state and the UI stays in sync with them.
The separation is real; it is simply held together by in-process
mechanisms that do not survive a network boundary.

### The build already layers along the seam

The CMake tree builds independently-linkable libraries:

- `BlackchirpData` (`cmake/BlackchirpData.cmake`) — the data layer:
  `Experiment`, configs, storage, analysis (`FtWorker`, `Ft`). Shared
  by both executables, depends on no hardware and no GUI.
- `BlackchirpHardware` — device drivers; main app only.
- `BlackchirpGui` / `BlackchirpViewerGui` — full GUI vs. a lightweight
  hardware-free GUI.
- `BlackchirpApplication` — the acquisition executable.

Two executables ship from this tree: `blackchirp` (acquisition) and
`blackchirp-viewer`. **The viewer is already, in effect, a read-only
client.** It links `BlackchirpData` + `BlackchirpViewerGui` with no
hardware layer and reconstructs the entire FTMW view from a saved
experiment on disk (`Experiment`'s disk-loading constructor, by
experiment number). It proves the data + presentation layers detach
cleanly from acquisition and hardware. A server–client split is
substantially "make the viewer's data-loading path consume a live
experiment over a socket instead of a finished one from disk."

### The threading model

`MainWindow` owns `d_threadObjectList`
(`QList<QPair<QThread*,QObject*>>`, `gui/mainwindow.h:118`) and moves
`LogHandler`, `HardwareManager`, and `AcquisitionManager` each onto its
own `QThread`. The GUI thread never touches acquisition state directly.

`AcquisitionManager` (`acquisition/acquisitionmanager.h`) lives on
`AcquisitionManagerThread`, owns the in-progress
`std::shared_ptr<Experiment>` (`ps_currentExperiment`), and drives the
acquisition loop. Its header already documents the discipline that a
protocol would formalize:

- All public slots and signal emissions run on the AM thread; other
  threads must use queued signal/slot connections or
  `QMetaObject::invokeMethod`.
- Waveform processing is dispatched to the thread pool via
  `QtConcurrent::run` and returned to the AM thread through a
  `QFutureWatcher`, so all state mutations stay thread-confined.
- The **sole cross-thread data contract** between the processing worker
  and the manager is a plain struct, `FtmwProcessingResult`
  (`acquisition/acquisitionmanager.h:23`) — already a serializable
  message in everything but name.

This is a message-passing architecture with hard thread boundaries. The
refactor does not introduce concurrency discipline; it replaces the
*transport* under boundaries that already exist.

## The three coupling mechanisms and how each must change

The user-identified crux is correct: UI↔core coupling rests on (1)
shared pointers, (2) signals/slots, and (3) cross-thread operations.
Each maps to a different piece of protocol work.

### 1. Shared `std::shared_ptr` to live state → serialized snapshots + deltas

This is the load-bearing coupling and the hardest to sever. The GUI and
the core do not copy data; they **share the same objects by pointer**:

- `std::shared_ptr<Experiment>` is a registered metatype
  (`qRegisterMetaType<std::shared_ptr<Experiment>>()`, `main.cpp:232`)
  and is handed from `MainWindow::experimentInitialized` to
  `AcquisitionManager::beginExperiment` across the thread boundary. Both
  sides then reference the same `Experiment`.
- `FtmwViewWidget` holds `std::shared_ptr<FidStorageBase> ps_fidStorage`
  and `std::shared_ptr<OverlayStorage> ps_overlayStorage`
  (`gui/widget/ftmwviewwidget.h:174-175`) — **the same storage objects
  the AcquisitionManager writes into.** The view reads FIDs directly out
  of the writer's storage.

A network client cannot hold these pointers. The refactor's central
task is to replace "share the object" with "ship a representation of the
object." Concretely:

- **Config/metadata (`Experiment` header, `FtmwConfig`, `LifConfig`,
  hardware configs).** Low-frequency, small. `Experiment` is a
  `HeaderStorage` tree that already serializes to a semicolon-delimited
  CSV schema; the same tree serializes to JSON for the wire. The client
  holds a mirror it never mutates directly — it sends *commands* to
  mutate and receives the updated snapshot back.
- **Bulk sample data (FIDs, FTs).** High-volume; must never cross the
  wire raw. The server owns `FidStorageBase` and runs decimation
  server-side (see §3), shipping only decimated series plus the metadata
  needed to label axes. Raw records stay on the acquisition machine.
- **Overlay storage.** `OverlayStorage` already separates persistent
  (disk-backed, async `QtConcurrent` I/O with completion signals) from
  in-memory preview overlays. Overlays become server-side resources the
  client references by id.

The design question this forces: **snapshot vs. delta granularity.**
Live FTMW acquisition accumulates into an averaged FID; the natural push
is a decimated FT delta on each processing batch, not a whole-experiment
snapshot. The `FtmwProcessingResult` boundary is the place that already
knows when a batch completed and is the natural emit point for a
"processed" event.

### 2. Cross-thread queued signals/slots → protocol event messages

The AM→GUI signals are already an event stream; they are named,
typed, and use registered metatypes. Each becomes a server→client
message:

| Signal (`AcquisitionManager`) | Payload today | Wire event |
|---|---|---|
| `ftmwUpdateProgress(int perMil)` | scalar | progress tick |
| `auxData(AuxDataMap, QDateTime)` | small map | aux/tracking point |
| `lifPointUpdate()` | (poke) | LIF point ready |
| `lifShotAcquired(int perMil)` | scalar | LIF progress |
| `experimentComplete()` | (poke) | experiment ended |
| `backupComplete()` | (poke) | backup finished |
| `logMessage` / `statusMessage` | string(+code) | log/status line |

The "poke" signals (`lifPointUpdate`, `experimentComplete`) are already
notifications that tell the GUI to go re-read shared storage — exactly
the pattern that becomes "server pushes an event; client requests (or is
pushed) the new decimated data." The client→server direction mirrors the
public *slots* (`beginExperiment`, `pause`, `resume`, `abort`,
`requestBackup`) and the `MainWindow` command surface (start experiment,
edit processing settings). These are a small, enumerable command set.

The work here is mechanical but broad: define the message schema, and
replace direct `connect(...)` wiring with encode-on-emit / decode-on-
receive at the socket. The number of distinct events is modest.

### 3. Timer-polled shared storage + client-side decimation → server-side decimation, pushed deltas

`FtmwViewWidget` refreshes the live view on a `timerEvent` poll
(`d_liveTimerId`), reading the current FID list out of shared storage,
running the FFT (`FtWorker`), and pushing the result into the plot,
whose `ZoomPanPlot` then decimates for the canvas width. Two things move:

- **Decimation moves server-side and becomes width-parameterized.** The
  `_filter(width, scaleMap)` logic is framework-agnostic array math; it
  runs in the core against a client-reported canvas width and visible
  x-range, and only the ~2×width-point result is serialized. This is the
  single most important performance decision and it keeps the current
  fast path intact.
- **The poll becomes a push (or a pull the client controls).** Instead
  of the view polling storage on a timer, the server emits a decimated
  delta when a processing batch completes (§1). Zoom/pan on the client
  becomes a request carrying the new x-range + width, answered with a
  freshly decimated series — the same computation `ZoomPanPlot` does on
  wheel/drag today, relocated across the boundary.

`FtWorker` (GSL FFT, windowing/apodization/zero-pad/sideband settings)
runs unchanged server-side; the processing-settings panel becomes a
command that reconfigures it and triggers a reprocess.

## What stays unchanged

- **Acquisition, hardware drivers, LIF hardware** (~35k lines under
  `hardware/`) — entirely below the seam; untouched.
- **The data layer** (`Experiment`, configs, `FidStorageBase` family,
  `AuxDataStorage`, `OverlayStorage`) — its *consumers* change, but the
  classes and the on-disk CSV schema stay. Serialization gains a JSON
  projection alongside the existing CSV one.
- **`FtWorker` / `Ft`** — the FFT/processing engine; relocated, not
  rewritten.
- **The decimation algorithm** — carried over verbatim as server-side
  code.
- **`SettingsStorage` / `BC::Key` persistence** — the write-protected
  QSettings-group model still governs server-side config ownership.
  Per-user *UI* preferences (curve appearance, presets, plot display
  settings) migrate to client-side state; only acquisition/hardware
  config stays server-authoritative. This split is a return to a
  boundary that already worked here: UI settings previously lived in
  `QSettings::UserScope` and core settings in `QSettings::SystemScope`,
  which behaved correctly on a multi-user system. The server–client
  split re-draws exactly that line, with the network boundary standing
  in for the scope boundary — settled, not open.

## What gets deleted rather than ported

Under the no-fidelity constraint, much of the ~20k-line plotting/UI
layer is not reproduced:

- `gui/plot/*` Qwt wrappers (`ZoomPanPlot`, `FidPlot`, `FtPlot`,
  `MainFtPlot`, curve classes) except the decimation math.
- `CurveAppearanceWidget` + `CurveAppearancePresetManager` +
  `PresetSaveDialog` (~1.8k lines) — replaced by declarative client
  components + a small JSON preferences blob.
- The dock/toolbar shell inside `FtmwViewWidget`
  (`redistributeDockSpace()`, toggle-action wiring) — replaced by
  whatever layout the web UI adopts.
- The right-click context-menu framework and per-curve menus.

The overlay system (`gui/overlay`, ~8.5k lines) is the one large
featureful block that must be re-expressed rather than dropped, but its
bulk is dialog/widget plumbing over `OverlayStorage`, which stays.

## Refactor in broad strokes

Ordered to keep the app working throughout; each phase is independently
valuable and the split can be abandoned after any of them.

1. **Cut the Qwt/data seam first (valuable regardless of destination).**
   Introduce a framework-neutral plot/series interface so `FtWorker`,
   the storage classes, and the decimation filter no longer name Qwt
   types in their public signatures. Qwt types currently leak into
   public method signatures across ~24 files (and into non-GUI code such
   as `data/experiment/overlaytypes.cpp`); this is the precondition that
   makes a swap tractable and is worth doing even if the migration
   stops here.
2. **Define the wire schema.** Enumerate the event set (§2) and command
   set, and give `Experiment`/configs a JSON projection beside the CSV
   one. Specify the decimated-series message (§3).
3. **Stand up the embedded server in the core**, off to the side of the
   existing GUI. `AcquisitionManager`'s existing signals/slots are the
   adapter points — encode on emit, decode into the existing slots.
4. **Vertical spike: the viewer as the client–server playground.** Make
   `blackchirp-viewer` the first client. It reads a finished experiment
   from disk, so it needs no hardware, acquisition loop, or experiment
   setup — yet it exercises the whole transport/decimation/render stack
   *and* the higher-frequency of the two real-time loops (see below).
   Point the viewer's plots at the server over a real socket (both ends
   on localhost is fine — the question is whether serialization +
   round-trip latency survives an interactive drag, so the transport
   must be real, not an in-process shortcut). Because the viewer already
   reprocesses FIDs when `FtWorker` settings change, the spike also
   exercises a mutate-and-reprocess *command* round-trip for free, not
   just read-only rendering.
5. **Expand panel by panel**, keeping the QWidget UI runnable in
   parallel until the web client reaches parity on the panels that
   matter. The live acquisition view is the follow-on target; by then
   the transport, serialization, decimation hand-off, and the
   interactive loop are already proven, leaving only the live-push path
   (§below) to validate.

### Why the viewer is the right first spike

Two distinct high-rate loops exist in the target system, with very
different demands:

- **Pan/zoom refresh** (client → server → client) fires at pointer-move
  rate — a drag emits ~60–120 events/s, each nominally wanting a freshly
  decimated series for the new x-range and canvas width, each round-trip
  inside the user's felt-latency budget. This is the loop that decides
  whether the whole premise works, and the viewer exercises it fully
  against static data.
- **Live push** (acquisition → client) is bounded by the far slower
  processing-batch rate. Forgiving by comparison, and *not* covered by
  the viewer (its data is static on disk).

So a passing viewer spike is necessary-but-not-sufficient to de-risk the
live acquisition view — but it retires the harder interactive loop plus
the entire transport/serialization/decimation/render stack, leaving the
live-push path as a smaller, well-scoped follow-on rather than the whole
bet.

**Key design decision the spike forces:** re-decimate server-side on
every pan event, or ship a decimated *superset* once and re-filter on
the client during the drag, round-tripping only on zoom-out or when the
pointer leaves the cached window? The current in-process code
re-decimates locally on every wheel/drag because it is free; across a
socket that likely does not hold, and a client-side cache with a defined
invalidation policy is the probable answer. The spike is where this is
measured and decided rather than guessed.

## Open questions / risks

- **Offline/self-contained packaging.** Today a student runs one native
  binary. A server + assets + browser story must stay trivial to launch
  on a lab instrument with no network. This is the concern most worth
  resolving before committing — it is a deployment problem, not a coding
  one, and it does not surface in a spike.
- **Snapshot/delta granularity and back-pressure** on the live path at
  full acquisition rate. The spike must measure this, not assume it.
- **Two languages / two build systems** permanently. Mitigated by the
  fact that "UI in C++" is already two mental models; the contributor
  pool for a web frontend (students who know TS/JS) is arguably larger
  than for Qwt.
- **Connection-state UX** — the client and a running acquisition
  disagreeing about state (reconnect, multiple clients, who may issue
  `abort`). New surface with no analogue in the current single-process
  app.
- **Authoritative config ownership** (settled). The server is the
  single writer for acquisition/hardware config; view preferences live
  client-side. This mirrors the previous `QSettings` UserScope/SystemScope
  split, which worked on multi-user systems. The remaining detail is
  reconciling multiple simultaneous clients' *view* preferences, which
  is a client-state problem, not a server one.

## Related

- `dev-docs/devel-roadmap.md` — existing roadmap; this effort is larger
  than anything currently listed there and would slot under "Large" if
  promoted from exploratory.

## Appendix A — Qwt/data seam interface draft

This is the concrete elaboration of refactor phase 1. It is landable on
its own, ahead of any server/web decision, and leaves the QWidget app
behaving identically. Sketches are illustrative, not compile-ready.

### The problem it solves

`BlackchirpPlotCurveBase` (`gui/plot/blackchirpplotcurve.h`) fuses three
concerns into one `QwtPlotCurve` subclass:

1. **Data model** — what the curve holds (point cloud; or evenly-spaced
   y-vector with `xFirst`/`spacing`; or an `Ft`; or a FID).
2. **Decimation** — `_filter(w, QwtScaleMap)`, the min/max pixel-column
   compression, including the closed-form even-spaced fast path.
3. **Rendering + appearance** — it *is* a `QwtPlotCurve`, draws itself,
   and stores appearance as Qwt enums (`QwtPlotCurve::CurveStyle`,
   `QwtSymbol::Style`, `QwtPlot::Axis`).

Concerns 1 and 2 have no legitimate reason to name a Qwt type. The only
Qwt coupling in the decimation is `QwtScaleMap`, used in exactly two
ways — `transform(x)` (data→pixel) and `invTransform(pixel)`
(pixel→data), both pure linear affine for these axes. The seam extracts
1 and 2 into `data/presentation/` (the Qwt-free, GUI-free shared
`BlackchirpData` library — where `curveappearance.h` already lives), and
leaves the Qwt curve as a thin rendering adapter.

### A.1 Geometry — replaces `QwtScaleMap`

```cpp
// data/presentation/plotgeometry.h  — no Qwt, no widgets
namespace BC::Plot {

/// Linear affine map between scale (data) and paint (pixel) coordinates.
/// Reproduces QwtScaleMap's transform/invTransform for linear axes,
/// which is all the FID/FT/tracking plots use.
struct AxisMap {
    double s1{0.0}, s2{1.0};   ///< scale-coordinate bounds (data units)
    double p1{0.0}, p2{1.0};   ///< paint-coordinate bounds (pixels)

    double transform(double s) const
    { return p1 + (s - s1) * (p2 - p1) / (s2 - s1); }
    double invTransform(double p) const
    { return s1 + (p - p1) * (s2 - s1) / (p2 - p1); }
};

} // namespace BC::Plot
```

On the QWidget side the adapter builds an `AxisMap` from the live
`QwtScaleMap` (`{map.s1(), map.s2(), map.p1(), map.p2()}`). On the server
side it is built from the client-reported visible x-range and canvas
width. Same struct, two sources — this is precisely the value that the
pan/zoom refresh loop ships from client to server.

### A.2 Appearance — neutral enums + a plain struct

Grow the existing `data/presentation/curveappearance.h` (today only
key-strings) into typed, Qwt-free appearance:

```cpp
namespace BC::Plot {

// Underlying values intentionally match the current Qwt enum integers
// so overlay-metadata CSVs and QSettings ints written by the existing
// build keep their meaning. The web client maps by NAME, not number.
enum class CurveStyle  : int { NoCurve=-1, Lines=0, Sticks=1, Steps=2, Dots=3 };
enum class MarkerStyle : int { NoMarker=-1, Ellipse=0, Rect=1, Diamond=2,
                               Triangle=3, Cross=8, XCross=9, HLine=10,
                               VLine=11, Star=12 /* … */ };
enum class LineStyle   : int { NoPen=0, Solid=1, Dash=2, Dot=3,
                               DashDot=4, DashDotDot=5 };     // == Qt::PenStyle
enum class PlotAxis    : int { YLeft=0, YRight=1, XBottom=2, XTop=3 }; // == QwtPlot::Axis

struct CurveAppearance {
    QString  color{"#000000"};   // hex; QColor on the widget side, CSS on the web side
    CurveStyle  curveStyle {CurveStyle::Lines};
    LineStyle   lineStyle  {LineStyle::Solid};
    double      thickness  {1.0};
    MarkerStyle marker     {MarkerStyle::NoMarker};
    int         markerSize {5};
    PlotAxis    axisX      {PlotAxis::XBottom};
    PlotAxis    axisY      {PlotAxis::YLeft};
    bool        visible    {true};
    bool        autoscale  {true};
    int         plotIndex  {-1};
};

} // namespace BC::Plot
```

`CurveAppearance` is the single object that: (a) the adapter reads to
configure a `QwtPlotCurve`; (b) `CurveStorageInterface` serializes to
QSettings/overlay-CSV (the `CurveKey` strings already exist); and (c) the
server serializes to JSON for the client. One struct, three sinks. The
matched enum integers are the backward-compat lever — persisted overlay
metadata stays valid with no migration.

**Long-term ownership (why this struct is transitional).** Appearance is
two different things with two different owners, and the typed
`CurveAppearance` is a monolith-era convenience that spans both:

- *Live/ephemeral curves* (live FID/FT, tracking) — appearance is never
  persisted and is pure presentation. Long-term the server does not model
  it at all; the client owns it outright.
- *Persisted artifacts* (overlays; styling saved with a stored
  experiment) — the color-coding is information the user wants to travel
  with the experiment (reopened on another machine, by another user, in
  another browser). It must be stored server-side, but the server's
  long-term role is **store without interpret**: an opaque,
  client-authored presentation blob attached to a data resource,
  round-tripped byte-for-byte with no knowledge of what `curveStyle` or
  `Ellipse` mean.

The clean line is therefore: the server owns data + identity + *opaque
storage* of per-artifact presentation; the client owns all presentation
*semantics* (enums, rendering, defaults, global style preferences). This
refines the UserScope/SystemScope split — global default styling is a
user preference (client-side), but appearance saved with a specific
overlay is an attribute of a shared data artifact, not a preference, so
it is server-persisted yet client-defined.

Consequently the *typed* `CurveAppearance` migrates to the client (a TS
type) as the web frontend takes over; the C++ struct survives only as
long as the QWidget frontend does, and the server side thins to an opaque
JSON attribute. Once the client owns the types, the matched-enum-integer
alignment (above) is no longer needed server-side — it exists purely to
carry the QWidget era across the seam. One migration consequence: overlay
metadata, today a Blackchirp-specific CSV the C++ parses, would evolve
toward persisting the client's presentation JSON verbatim. The
semicolon-CSV schema is shared across all three trees (`AGENTS.md`), so
that format change is a cross-tree coordination point, not a local edit.

### A.3 Data model — canonical data + decimation, no Qwt

```cpp
// data/presentation/plotseriesdata.h  — no Qwt, no widgets
namespace BC::Plot {

/// Owns a curve's canonical (undecimated) data and knows how to reduce
/// it to a per-pixel-column min/max set for a given canvas width + map.
class PlotSeriesData {
public:
    virtual ~PlotSeriesData() = default;

    /// Full, undecimated points — for CSV export and autoscale.
    virtual QVector<QPointF> data() const = 0;
    /// Data-space bounding rect; invalid (w<0/h<0) when empty.
    virtual QRectF boundingRect() const = 0;
    /// ~2*width points via min/max column compression. Thread-safe:
    /// implementations snapshot under their own lock, then compute lock-free.
    virtual QVector<QPointF> decimate(int width, const AxisMap &map) const = 0;
};

/// Arbitrary (x,y) point cloud (tracking/aux traces).
class PointCloudSeriesData : public PlotSeriesData {
public:
    void setData(QVector<QPointF> d, double yMin, double yMax);
    void appendPoint(QPointF p);
    // … overrides call the free function decimatePointCloud() (A.4)
};

/// Evenly-spaced y-vector: the shared model behind FID, FT, and generic
/// uniform traces. Sources differ only in how they fill (xFirst,spacing,y).
class EvenSpacedSeriesData : public PlotSeriesData {
public:
    void setData(QVector<double> y, double xFirst, double spacing,
                 double yMin, double yMax);
    // … overrides call decimateEvenSpaced() (A.4)
};

} // namespace BC::Plot
```

The three current even-spaced curve types (`BlackchirpFTCurve`,
`BlackchirpFIDCurve`, `BlackchirpEvenSpacedCurve`) collapse to
`EvenSpacedSeriesData` fed from three sources. `FtWorker` gains a small
method that emits `(xFirst, spacing, y, min, max)` directly, which is
also exactly the FT payload the server serializes — the plot model and
the wire model become the same object.

### A.4 Decimation — the crown jewel, extracted verbatim

The two `_filter` bodies (`blackchirpplotcurve.cpp:324` and `:430`) move
here nearly unchanged; the only edit is `QwtScaleMap` → `AxisMap` (the
`transform`/`invTransform` calls are identical). This is the Qwt-free,
server-shareable core:

```cpp
// data/presentation/plotdecimate.h
namespace BC::Plot {
QVector<QPointF> decimatePointCloud(const QVector<QPointF> &data,
                                    int width, const AxisMap &map);
QVector<QPointF> decimateEvenSpaced(double xFirst, double spacing,
                                    const QVector<double> &y,
                                    int width, const AxisMap &map);
}
```

`tests/tst_zoompanplotthreadsafety` already exercises the filter path;
add a direct unit test pinning `decimate*` output against the current
`_filter` output on fixture data, so the extraction is provably
behavior-preserving before the Qwt class is touched.

### A.5 The adapter — what remains Qwt-side

`BlackchirpPlotCurveBase` stops owning data and decimation logic and
becomes a thin `QwtPlotCurve` that holds a `PlotSeriesData*` and a
`CurveAppearance`:

```cpp
class BlackchirpPlotCurveBase : public QwtPlotCurve {
    std::unique_ptr<BC::Plot::PlotSeriesData> d_series;
    BC::Plot::CurveAppearance d_appearance;
public:
    void filter(int w, const QwtScaleMap map) {          // called by ZoomPanPlot
        BC::Plot::AxisMap m{map.s1(), map.s2(), map.p1(), map.p2()};
        setSamples(d_series->decimate(w, m));            // neutral core does the work
    }
    void applyAppearance();   // translate d_appearance -> pen/symbol/axes (enum casts)
    QRectF boundingRect() const override { return d_series->boundingRect(); }
    // setColor/setCurveStyle/... mutate d_appearance, persist, applyAppearance()
};
```

Enum translation is a `static_cast<int>` in both directions because A.2's
underlying values match Qwt's. `ZoomPanPlot`, `CurveFactory`,
`CurveAppearanceWidget`, and the overlay code keep working against this
adapter; their Qwt-typed *public* signatures can then be migrated to the
`BC::Plot` enums incrementally (a follow-on mechanical pass), because the
values are identical the whole time.

### A.6 What the wire gets for free

Once A.1–A.4 exist, the server's FT message is `CurveAppearance` (JSON
by field name) + the output of `decimateEvenSpaced` (a `QVector<QPointF>`
→ a flat `[x,y,…]` array). No Qwt, no widgets, no new data modeling —
the seam and the protocol payload are the same types. The viewer spike
(phase 4) consumes exactly these.

### A.7 Landing order (app stays green throughout)

1. Add `plotgeometry.h`, expand `curveappearance.h`, add `plotdecimate.h`
   with the extracted `_filter` bodies; unit-test `decimate*` against the
   current output.
2. Add `PlotSeriesData` + the two concrete classes; unit-test.
3. Reimplement the curve subclasses' `_filter`/data storage in terms of
   `PlotSeriesData` (delete the duplicated logic). Behavior unchanged;
   `tst_zoompanplotthreadsafety` must stay green.
4. Introduce `CurveAppearance` into the curve/adapter and route
   `CurveStorageInterface` through it. Overlay CSVs still load (matched
   enum ints).
5. (Follow-on, optional before the spike) migrate the Qwt-typed public
   signatures across `ZoomPanPlot`/`CurveFactory`/overlay to `BC::Plot`
   enums.

Steps 1–4 are the seam; the app is Qwt-native and identical after each.
Only step 5 changes call sites, and only cosmetically.
