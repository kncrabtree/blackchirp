# Logging and Debug Message Cleanup

## Goal
Review and rationalize all logging output before the Blackchirp 2.0.0 release. Every
message should be intentionally categorized: shown to the user, sent only to the debug
log, or removed entirely. This is a pre-release polish task — one of the last things
before beginning documentation revision.

## Background
During development of the cmakemigration branch, diagnostic output accumulated in two
forms: `qDebug()` calls (which bypass the log system) and `emit logMessage()` calls
(which go to the UI log tab). The recent addition of debug logging to the application
configuration gives us a proper channel for diagnostic output, but most messages haven't
been reclassified to use it.

### Current State (survey results)

**qDebug():** 41 calls across 8 files
- Hardware registration/initialization tracing (19 calls in hardware/core/)
- Overlay system diagnostics (18 calls in data/processing/ and gui/overlay/)
- Vendor library loading (2 calls in hardware/library/)
- None of these go through the log system

**emit logMessage():** ~445 calls across 46 files
- ~380 Error (~85%) — but many are configuration verification traces, not true errors
- ~31 Normal (~7%) — many are internal lifecycle tracing, not user-facing status
- ~25 Warning (~6%) — generally appropriate
- ~6 Debug (~1%) — drastically underused
- 0 Highlight — never used

## Principles

### What users should see (Normal / Warning / Error)
- Connection success/failure outcomes
- Experiment progress milestones (start, completion, abort)
- Hardware state changes the user initiated or needs to act on
- Errors that require user intervention or indicate data loss risk

### What goes to debug log only (Debug)
- Hardware lifecycle tracing (creation, thread assignment, destruction)
- Configuration loading/syncing progress
- Protocol-level command/response details
- Parameter verification traces (digitizer scale parsing, etc.)
- Registration and initialization diagnostics

### What gets removed
- Development-time scaffolding (`qDebug` calls added during feature development)
- Redundant messages (e.g., logging the same event at multiple levels)
- Messages that duplicate information already visible in the UI

### qDebug() policy going forward
All `qDebug()` calls should be replaced with `emit logMessage(..., LogHandler::Debug)` where
the object has access to the signal, or removed if the message is no longer useful. This
ensures all diagnostic output flows through the unified log system and respects the user's
debug logging preference. Direct `qDebug()` may remain acceptable only in static/non-QObject
contexts where logMessage is unavailable (e.g., static registration functions), but these
should be minimized.

## Work Areas (by priority/volume)

### 1. FTMW Digitizer Files (~285 logMessage calls, 7 files)
**Largest volume by far.** These files contain extensive command/response verification that
was written as Error but is really diagnostic tracing.

Files: `mso72004c.cpp`, `dpo71254b.cpp`, `mso64b.cpp`, `dsa71604c.cpp`, `dsov204a.cpp`,
`dsox92004a.cpp`, `m4i2220x8.cpp`

**Approach:** Per-message triage. Categories:
- **Keep as Error:** Failures that prevent acquisition (can't configure, can't read waveform)
- **Downgrade to Debug:** Response parsing details, parameter comparison traces, hex dumps
- **Downgrade to Warning:** Parameter mismatches that are corrected automatically

This is the most labor-intensive area and will require discussion on individual messages
since the line between "real error" and "diagnostic trace" isn't always obvious in
digitizer configuration.

### 2. HardwareManager (~74 logMessage calls)
**31 Normal messages** that are mostly initialization lifecycle tracing:
- "Loading hardware configuration from runtime profiles..."
- "Started thread for hardware: [key]"
- "Hardware created and initialized..."
- "Updating ClockManager with N clock(s)"

**Approach:** Bulk reclassification. Most Normal → Debug. Keep Error and Warning as-is.
User-facing Normal messages to retain: connection test results emitted to the log tab.

### 3. qDebug() Elimination (41 calls, 8 files)
- `runtimehardwareconfig.cpp` (8): Initialization/sync tracing → Debug logMessage or remove
- `hardwareregistration.cpp` (4): Registration enumeration → remove (startup-only)
- `hardwareregistry.cpp` (4): Instance creation tracing → Debug logMessage or remove
- `hardwareprofilemanager.cpp` (1): System profile creation → remove or Debug
- `overlaymanagerwidget.cpp` (15): Widget state diagnostics → Debug logMessage or remove
- `overlaystorage.cpp` (4): File operation diagnostics → Debug logMessage or remove
- `overlayprocessmanager.cpp` (3): Operation state → Debug logMessage or remove
- `labjacklibrary.cpp` (2): Library loading → Debug logMessage

**Challenge:** Several of these are in non-HardwareObject classes that don't have direct
access to `emit logMessage()`. Options:
- Route through `LogHandler` directly (if it supports static/global access)
- Use `qDebug()` with a Qt message handler that routes to LogHandler
- Accept `qDebug()` in static contexts but document the policy

### 4. Communication Protocol Files (~15 calls)
- `communicationprotocol.cpp`: Error messages are appropriate (connection failures)
- `tcpinstrument.cpp`: 2 Normal messages about socket state → Debug
- `gpibinstrument.cpp`: 2 Debug messages already correctly categorized

### 5. Optional Hardware Implementations (~110 calls)
- Flow controllers, pulse generators, AWGs, pressure/temp controllers
- Most are Error on communication failures — likely appropriate
- Quick pass to verify no diagnostic traces masquerading as errors

### 6. LIF Components (~36 calls)
- Laser and digitizer operation errors — likely appropriate
- Quick pass to verify severity levels

### 7. LogHandler::Highlight Usage
Currently used for experiment start (`mainwindow.cpp:717`) and normal experiment
completion (`experiment.cpp:403,409`). These go through direct `p_lh->logMessage()`
calls rather than `emit logMessage()`, which is why they were missed in the signal-based
survey. Consider whether other milestone events warrant Highlight.

## Execution Strategy
This is best done as a series of focused passes rather than one massive commit:

1. **qDebug elimination pass** — straightforward mechanical replacement/removal
2. **HardwareManager reclassification** — bulk Normal → Debug, review individually
3. **Digitizer file triage** — per-file, per-message review (most discussion needed)
4. **Quick pass on remaining files** — verify optional hardware, LIF, comm protocols
5. **Final review** — read through the log output of a typical startup + experiment cycle
   to verify the user sees a clean, informative log without noise

## LogHandler Redesign: Thread-Safe Global Instance

The current LogHandler design routes messages through Qt signal chains — hardware objects
`emit logMessage()`, which propagates through HardwareManager to LogHandler. This creates
signal duplication and complex connection management, especially across threads.

A better design: make LogHandler a **thread-safe global singleton** (like RuntimeHardwareConfig
or HardwareRegistry). Any code — QObject or not, any thread — could call:
```cpp
LogHandler::instance().log("message", LogHandler::Debug);
// or a convenience macro/function:
bcLog("message", LogHandler::Debug);
```

**Benefits:**
- Eliminates the `emit logMessage()` → signal chain → LogHandler relay pattern
- Non-QObject and static contexts (registration, factories) can log directly
- Removes the need for `qDebug()` entirely
- Simplifies HardwareManager connection setup (no logMessage forwarding)
- Thread safety via internal mutex or queued dispatch to the UI thread

**Considerations:**
- LogHandler still needs to update the UI log widget, which must happen on the main thread.
  The singleton could use `QMetaObject::invokeMethod` with `Qt::QueuedConnection` internally
  to dispatch to the UI, or use a lock-free queue polled by a timer.
- Existing `emit logMessage()` signals on HardwareObject could be kept temporarily for
  backward compatibility and removed incrementally, or removed in one pass as part of this
  cleanup.
- The `d_startLogMessage` / `d_endLogMessageCode` pattern in Experiment already calls
  LogHandler directly — this would become the standard pattern.

**Recommendation:** Do the LogHandler redesign as the first step of this cleanup task.
Once global logging is available, the qDebug elimination and logMessage reclassification
passes become straightforward mechanical work.

## Other Open Questions
- Are there digitizer error messages that hardware vendors or support staff rely on
  seeing? If so, those should stay at Error even if they look diagnostic.
- Should we add a log level filter to the UI log tab (e.g., show/hide Debug, Warning)?
  This is out of scope for this task but would complement it.
