# Digitizer Data Flow Optimization

**Selected approach**: Option A1 (Shared Ring Buffer) with optional
producer-side pre-accumulation.

## Problem Statement

The current FTMW waveform data flow uses per-shot Qt signal emission
through two relay hops (Digitizer -> HardwareManager -> AcquisitionManager).
While this works today (~20k FIDs/sec with firmware block averaging), the
architecture has scaling concerns:

- **Unbounded event queue**: If the consumer (AcquisitionManager) blocks or
  falls behind, the Qt event queue grows without limit. There is no
  backpressure mechanism — the digitizer keeps emitting signals regardless
  of whether the consumer has processed previous ones.
- **Per-shot event loop overhead**: Every waveform triggers two queued signal
  emissions and two event dispatches. Although QByteArray uses copy-on-write
  (so no deep copies occur — just atomic ref count increments), the
  QMetaCallEvent allocations and event loop dispatch overhead are unnecessary
  when the data could be consumed more directly.
- **HardwareManager passthrough**: The relay through HardwareManager adds an
  event dispatch with zero value — it exists because AcquisitionManager cannot
  access the digitizer object directly (practical constraint, not design choice).
- **Python digitizer concern**: Future Python digitizers using JSON IPC would
  add serialization overhead; single-shot FIFO instruments without firmware
  averaging could hit throughput limits.

**Goal**: Maximize FIDs/sec — be limited by hardware and experiment design,
not software architecture. The solution should not preclude efficient Python
digitizer support in the future.

## Current Architecture

```
Digitizer Thread              HW Manager Thread           Acq Manager Thread
┌──────────────┐              ┌─────────────────┐         ┌──────────────────┐
│ readWaveform │              │                 │         │                  │
│   (timer or  │─signal──────>│ ftmwScopeShot   │─signal─>│ processFtmwShot  │
│   socket)    │  QByteArray  │ Acquired()      │  (COW   │   addFids()      │
│              │  (COW ref)   │ (passthrough)   │   ref)  │   parseWaveform()│
│ emitShot()   │              │                 │         │   accumulate     │
└──────────────┘              └─────────────────┘         └──────────────────┘
```

### Per-waveform work (CPU path, no CUDA):
1. Digitizer copies raw data from hardware buffer into QByteArray
2. Queued signal #1: COW ref count increment + QMetaCallEvent (digi -> HWManager)
3. Queued signal #2: COW ref count increment + QMetaCallEvent (HWManager -> AcqManager)
4. `parseWaveform()`: byte-by-byte unpack into QVector<qint64> Fid objects
5. `FidStorageBase::addFids()`: element-wise addition into running accumulator
6. `advance()` checks: completion, autosave (every 60s to CSV)

Note: QByteArray uses implicit sharing (copy-on-write), so the signal hops
do NOT cause deep copies. The per-shot overhead is event queue management,
not data copying. The real concerns are unbounded queue growth, unnecessary
event dispatches, and the inability to batch or apply backpressure.

### Key observations:
- Waveform sizes vary widely (small to several MB)
- The M4i2220x8 already has a DMA ring buffer; data is copied out byte-by-byte
- FidStorageBase already uses a QMutex for thread safety (used by GUI reads)
- Chirp scoring / phase correction require parsed FIDs before accumulation,
  but these features are low priority and not in active use
- The CUDA path (GpuAverager) is rare and handles its own parse+accumulate
- LIF data (QVector<qint8>) is smaller and less frequent; signal-based is fine

## Selected Design: Option A1 — Shared Ring Buffer

### Concept

Replace the two signal hops with a bounded SPSC (single-producer,
single-consumer) ring buffer shared between the digitizer thread and the
AcquisitionManager thread. The digitizer writes waveform data into the
buffer; the AcquisitionManager drains and processes entries.

The base class (FtmwScope) handles all buffer management. Digitizer
implementations call `emitShot()` exactly as they do today; the base class
writes to the buffer instead of emitting a signal. Implementations may
optionally perform pre-accumulation before calling `emitShot()` if they
choose to, but the default path requires no implementation changes.

### Architecture

```
Digitizer Thread                              Acq Manager Thread
┌──────────────────┐  ┌───────────────────┐   ┌──────────────────┐
│ readWaveform()   │  │   WaveformBuffer  │   │ drainBuffer()    │
│                  │  │  (bounded SPSC)   │   │                  │
│ emitShot(data)   │─>│  ┌─┬─┬─┬─┬─┬─┐   │──>│ for each entry:  │
│  [base class     │  │  │ │ │ │ │ │ │   │   │  parseWaveform() │
│   writes to buf] │  │  └─┴─┴─┴─┴─┴─┘   │   │  addFids()       │
│                  │  │  drop-newest on   │   │  advance()       │
│ optional:        │  │  overflow         │   │                  │
│  pre-accumulate  │  └───────────────────┘   │ emit progress    │
│  N shots first   │     ↑           ↑        │ once per drain   │
└──────────────────┘     │           │        └──────────────────┘
                    owned by      ref passed
                    FtmwScope     via Experiment
                                  at setup time
```

### Why this approach

1. **Bounded memory with backpressure**: Fixed-size ring buffer (default ~10
   slots) prevents unbounded queue growth. Drop-newest policy discards
   incoming data when the consumer falls behind, which is the correct
   precursor to dynamic pre-accumulation (Phase 6): when the buffer is
   full, the producer will eventually accumulate shots locally instead
   of discarding them.

2. **Removes HardwareManager from the data path**: The buffer reference is
   passed through the Experiment object during setup. HardwareManager
   continues to handle lifecycle signals but never touches waveform data.

3. **Preserves separation of concerns**: The digitizer produces raw data;
   the AcquisitionManager still handles parsing, chirp scoring, and
   accumulation. No FtmwConfig knowledge needed in the digitizer thread.

4. **Natural adaptive batching**: The consumer drains all available entries
   per wake-up cycle, automatically adapting to producer/consumer speed
   differences without an explicit batch size parameter.

5. **Enables optional pre-accumulation**: If a digitizer implementation
   wants to pre-sum N waveforms before writing to the buffer, it can do so
   and annotate the entry with a shot count. The base class provides this
   as an opt-in mechanism, not a requirement.

6. **Dynamic pre-accumulation on backpressure**: If the buffer is full, the
   producer (in the base class) can automatically begin accumulating incoming
   shots until a slot becomes available, then write the accumulated result.
   This provides graceful degradation under load rather than data loss.

7. **Clean Python digitizer path**: Future Python digitizers write to the
   same buffer interface. For maximum performance, the Python trampoline
   could use shared memory (e.g., `multiprocessing.shared_memory`) to write
   raw waveform bytes directly into the buffer without JSON serialization.

8. **Base class handles complexity**: All buffer management, backpressure,
   and optional pre-accumulation logic lives in FtmwScope. Digitizer
   implementations call `emitShot()` as before; no implementation changes
   needed for the basic path.

### Alternatives considered

- **Option B (Pre-sum in digitizer thread)**: Attractive for throughput, and
  the digitizer is arguably the correct place for waveform parsing since it
  knows its own data format. However, moving parseWaveform + accumulation
  into the digitizer creates tight coupling to FtmwConfig internals (byte
  order, record layout, Fid templates, multi-record/block-average accounting).
  The selected design borrows the best part of Option B — optional
  pre-accumulation — without requiring it.

- **Option C (Batched signals)**: Least disruptive but least beneficial.
  Reduces event frequency but doesn't address unbounded queue growth or the
  HardwareManager relay. QByteArray COW means the copy savings are minimal.

- **Option D (Direct buffer access to FidStorageBase)**: Maximum performance
  but maximum coupling. Mutex contention with GUI reads and error handling
  complexity make this risky.

## Detailed Design

### WaveformBuffer class

A bounded, thread-safe SPSC ring buffer. Each slot holds a `WaveformEntry`:

```cpp
struct WaveformEntry {
    QByteArray data;       // raw waveform bytes
    quint64 shotCount;     // number of shots represented (1 for raw, N for pre-accumulated)
};
```

Key properties:
- **Fixed slot count**: Configured at creation (default 10 slots)
- **Pre-allocated QByteArrays**: Each slot's QByteArray is pre-reserved to the
  expected waveform size at experiment start, avoiding per-shot heap allocation
- **Drop-newest overflow**: When full, the incoming entry is discarded.
  This is race-free (producer never modifies the read index) and serves as
  the stepping stone toward dynamic pre-accumulation. A counter tracks
  dropped entries for logging/diagnostics.
- **Cross-platform notification**: QSemaphore-based signaling. Producer
  releases the semaphore after each write; consumer acquires with a timeout
  to allow periodic housekeeping even when no data arrives.
- **Thread safety**: Relies on SPSC discipline (one producer thread, one
  consumer thread). Internal synchronization via atomic read/write indices
  plus QSemaphore for notification.

### Buffer entry metadata

The `shotCount` field in each entry enables:
- **Raw waveforms**: shotCount = scopeConfig.shotIncrement() (1 for single-shot,
  N for firmware block averaging)
- **Pre-accumulated waveforms**: shotCount = sum of accumulated shot increments
- **Consumer accounting**: AcquisitionManager uses shotCount for progress
  tracking and FidStorageBase shot accumulation

For multi-record/block-average modes, `shotCount` must correctly reflect the
actual number of FIDs represented. The base class computes this from
`FtmwDigitizerConfig::d_blockAverage` and `d_numAverages` automatically.

### Segment boundary synchronization

Multi-segment acquisitions (LO Scan, DR Scan) require that no waveform data
from one segment leaks into the next. The current architecture uses
`setAcquisitionGated(true)` on the digitizer and `d_processingPaused` in
FtmwConfig to synchronize segment transitions.

With the ring buffer, segment boundaries require additional care:
- When `setAcquisitionGated(true)` is called, the producer stops writing to
  the buffer (existing behavior — `emitShot()` already returns early when gated)
- The consumer must **drain the buffer completely** before the segment advances
- A **flush marker** entry (sentinel with empty data) can be written to the
  buffer when gating begins, signaling the consumer that all preceding data
  belongs to the current segment
- After the consumer processes the flush marker, it signals readiness for the
  next segment (replacing the current `d_processingPaused` mechanism)

This ensures clean segment boundaries without data leakage, even if the
buffer contains multiple entries when a segment transition occurs.

### FtmwScope base class changes

```
Current emitShot():
  if gated → return
  if discardCount > 0 → decrement, return
  emit shotAcquired(data)       // Qt signal

New emitShot():
  if gated → return
  if discardCount > 0 → decrement, return
  if d_preAccumulating:
    accumulate data into d_accumBuffer
    d_accumShots += shotIncrement()
    if buffer has space OR d_accumShots >= threshold:
      write {d_accumBuffer, d_accumShots} to ring buffer
      reset accumulator
  else:
    if buffer is full:
      begin pre-accumulation (set d_preAccumulating = true)
      accumulate this shot
    else:
      write {data, shotIncrement()} to ring buffer
```

The pre-accumulation fallback means the base class handles backpressure
gracefully. Implementations don't need to know about it. Pre-accumulation
in the base class would use a simple raw-byte summation (treating the
QByteArray as an array of 1/2/4-byte integers based on `d_bytesPerPoint`),
which is exactly what `parseWaveform()` does but without constructing Fid
objects. This keeps the digitizer free of FtmwConfig coupling while still
enabling accumulation at the producer level.

Note: Pre-accumulation changes the data type of accumulated values (e.g.,
8-bit samples summed N times need wider storage). The accumulation buffer
would need to use a wider type (e.g., qint64 per sample) and the consumer
would need to know to skip `parseWaveform()` for pre-accumulated entries,
receiving the data as already-parsed qint64 values. This adds some
complexity to the entry format:

```cpp
struct WaveformEntry {
    QByteArray data;
    quint64 shotCount;
    bool preAccumulated;   // if true, data contains qint64 values, not raw bytes
};
```

### AcquisitionManager changes

Replace the `processFtmwScopeShot(QByteArray)` slot with a consumer loop:

```
drainBuffer():
  while buffer has entries:
    entry = buffer.read()
    if entry is flush marker:
      handle segment boundary
      break
    if entry.preAccumulated:
      merge qint64 data directly into FidStorageBase
    else:
      existing addFids(entry.data) pipeline
    totalShots += entry.shotCount
  emit ftmwUpdateProgress(...)
  checkComplete()
```

The consumer is triggered by:
- QSemaphore notification (primary path — low latency)
- Periodic timer (backup — ensures progress even if semaphore notification
  is missed, handles housekeeping like autosave checks)

### Connection setup

During experiment initialization:
1. FtmwScope creates the WaveformBuffer (sized based on waveform size)
2. The buffer pointer is stored in the Experiment object (or FtmwConfig)
3. AcquisitionManager retrieves the buffer pointer when beginning the experiment
4. HardwareManager is not involved in the data path

This leverages the existing pattern where the Experiment object is shared
between HardwareManager and AcquisitionManager during setup.

## Implementation Phases

### Phase 1: WaveformBuffer class **COMPLETE**
- Implement `WaveformBuffer` with SPSC ring buffer semantics
- `WaveformEntry` struct with data, shotCount, preAccumulated flag
- Drop-newest overflow with dropped-entry counter
- QSemaphore notification
- Pre-allocated QByteArray slots
- Comprehensive unit tests (correctness, threading, overflow, flush markers)

### Phase 2: FtmwScope integration **COMPLETE**
- Add `WaveformBuffer` member to FtmwScope base class
- Modify `emitShot()` to write to buffer instead of emitting signal
- Buffer created in `beginAcquisition()`, destroyed in `endAcquisition()`
- Retain `shotAcquired` signal as a lightweight no-data notification for
  shot counting / UI updates if needed (or remove if not used)
- No changes to any digitizer implementation classes — they still call
  `emitShot(data)` as before

### Phase 3: Experiment / FtmwConfig plumbing **COMPLETE**
- Added `WaveformBuffer*` non-owning pointer to FtmwConfig with getter/setter
- FtmwScope sets buffer pointer on FtmwConfig during `hwPrepareForExperiment()`
- AcquisitionManager accesses buffer via `exp->ftmwConfig()->waveformBuffer()`

### Phase 4: AcquisitionManager consumer **COMPLETE**
- Replaced `processFtmwScopeShot` with worker-thread architecture:
  - `drainFtmwBuffer()` (AM event loop, 20ms timer): reads entries from
    buffer (fast QByteArray moves), dispatches batch to QtConcurrent worker,
    pauses drain timer
  - Worker thread: calls `ftmw->addFids()` per entry (expensive parse +
    accumulate), checks atomic abort flag between entries
  - `onProcessingComplete()` (AM event loop): calls `advance()`, emits
    progress/signals, restarts drain timer
- AM event loop stays responsive during processing — abort, pause, aux data
  all execute immediately
- `finishAcquisition()` sets abort flag and waits for in-flight worker
  (bounded by one addFids call)
- Removed MainWindow signal connection for old data path

### Phase 5: HardwareManager cleanup **COMPLETE**
- Removed `ftmwScopeShotAcquired` signal from HardwareManager
- Removed relay connection in `setupHardwareSpecificConnectionsWithTracking()`

### Phase 6: Optional pre-accumulation in base class
- Implement backpressure-triggered pre-accumulation in `emitShot()`
- Wide accumulation buffer (qint64 per sample)
- Consumer path for pre-accumulated entries (skip parseWaveform)
- Configurable threshold or fully dynamic

### Phase 7: Testing and benchmarking
- Unit tests for WaveformBuffer (Phase 1)
- Integration test: VirtualFtmwScope -> buffer -> AcquisitionManager
- Throughput benchmark: old signal path vs new buffer path
- Regression tests: chirp scoring, phase correction, segment boundaries
- Multi-segment acquisition test (LO scan, DR scan)
- Peak-up mode test (rolling average semantics)
- Autosave timing verification

### Future: Python digitizer considerations (design only)
- Document how PythonFtmwScope would write to the WaveformBuffer
- Evaluate `multiprocessing.shared_memory` for zero-copy Python -> C++ transfer
- JSON IPC would handle commands only; waveform data via shared memory
- LifScope stays signal-based; no changes needed for PythonLifScope
