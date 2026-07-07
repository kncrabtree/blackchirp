# Chirp jitter monitor — sub-sample trigger-timing diagnostics

Status: proposed (planning only). Companion analysis-side design:
`ftmwpipeline` `dev-docs/planning/instrument-clock-declaration.md`
(timebase self-calibration; the pipeline is the intended consumer of
the data this feature records).

## Motivation

The averaged FID is exactly the true signal filtered by the empirical
characteristic function of the per-shot trigger-timing offsets: with
shot delays `dt_k`, the stored average's spectrum is
`S(f) * Phi(f)` where `Phi(f) = (1/N) * sum_k exp(-i*2*pi*f*dt_k)`.

Consequences, scaling with baseband frequency:

- **Band-dependent intensity attenuation.** Gaussian jitter of rms
  `sigma_t` attenuates by `exp(-2*pi^2*f^2*sigma_t^2)`: at
  `sigma_t = 5 ps` a 14.5 GHz baseband component loses ~10% amplitude
  while a 5 GHz component loses ~2% — an ~8% relative-intensity tilt
  across a 26–40 GHz CP band that propagates into rotational
  temperatures and abundance ratios. At 1 ps the effect is 0.4% and
  ignorable. Nobody currently knows which regime an acquisition is in.
- **Lineshape kernel.** The same `Phi(f)` is a convolution kernel on
  every line; for ultra-high-SNR fits it is one candidate contributor
  to the residual lineshape error.
- **Positions are immune.** Symmetric jitter gives a real `Phi(f)`
  (pure attenuation); a drifting mean delay gives a linear spectral
  phase (a time-origin shift). Neither moves a line's frequency.

If the `dt_k` series (or its accumulated `Phi(f)`) is recorded, the
filter is *exactly computable* and the analysis side can divide it out
of the spectrum — an exact deconvolution, not a statistical model. If
only the binary question matters ("are deep averages jitter-limited?"),
the aggregated rms answers it per acquisition.

The existing phase-correction machinery already computes nearly
everything required and throws the sub-sample part away.

## Current state

`FtmwConfig::preprocessChirp` (`src/data/experiment/ftmwconfig.cpp`),
active when `d_phaseCorrectionEnabled`:

- FOM = raw dot product of the new shot's chirp region against the
  running average's chirp at integer sample lags
  (`calculateFom`, Kahan-summed).
- Hill-climb over the lag from `d_currentShift`: move only when a
  neighbour beats the center by 1.15x; <= 5 lag steps per shot,
  |total| <= 50; shots whose FOM falls below 0.9x the previous are
  rejected; the winning integer shift translates the whole record at
  co-addition (`FidStorageBase::addFids`).
- `ChirpShift` / `ChirpPhaseScore` go to aux data each tick.

Properties relevant here:

- **Resolution is one sample (20 ps at 50 GSa/s).** Coherence at the
  top of a wideband receiver (1 rad at 14.5 GHz) requires ~10 ps, so
  the integer correction guards against gross trigger walk but is blind
  to the sub-sample jitter that actually attenuates the high band. On
  an instrument whose trigger chain is locked to the same reference as
  the AWG, the shift typically sits at 0 and the machinery measures
  nothing.
- The hill-climb already evaluates the FOM at `shift-1`, `shift`,
  `shift+1` — the three points of a parabolic sub-sample interpolation
  that is currently discarded.
- The FOM thresholds are amplitude-coupled (a 10% power sag reads as
  decoherence), and the GUI tooltip warns the chirp must not saturate
  the digitizer; on instruments that deliberately clip the chirp the
  thresholds are biased even though the correlation *peak position*
  (zero-crossing information) survives clipping.

## Design

Three independent pieces, smallest first; each is useful alone.

### 1. Sub-sample offset estimator (the core)

At the end of each shot's hill-climb, parabolically interpolate the
three FOMs already in hand:

```
delta = 0.5 * (fomDown - fomUp) / (fomDown - 2*fomCenter + fomUp)
```

`shift + delta` is the shot's chirp timing offset against the running
average, in samples, with sub-sample resolution. Notes:

- A correlation peak ~1–2 samples wide is not a parabola, so `delta`
  carries a stable systematic bias (fraction of a sample). For a
  *monitor* this is irrelevant: jitter is the spread, drift is the
  trend, and both survive a fixed bias. Document it; do not chase it.
- The reference is the running average, so the measured offset is
  shot-minus-mean-of-previous. For `Phi(f)` reconstruction this is the
  correct frame to first order (offline tooling can integrate the
  reference drift from the same series); record the raw values and
  keep the subtlety in the offline consumer.
- Clipped-chirp variant: a `sign()` (1-bit) correlation FOM is
  clipping-invariant by construction and cheaper than the raw product.
  Worth a setting if monitor users run clipped chirps (the current
  instrument does).

### 2. Monitor-only mode (decoupled from correction)

A new flag (e.g. `d_chirpMonitorEnabled`) that runs the FOM evaluation
and the sub-sample estimator on every shot *without* applying any
shift, without rejecting shots, and without the amplitude thresholds.
Users who distrust the correction (or whose chirp violates its
assumptions) still get the diagnostics. `d_phaseCorrectionEnabled`
implies the monitor; the monitor alone changes no data.

Cost: identical to phase correction (three Kahan dot products over the
chirp region per shot) — already demonstrated affordable in
acquisition.

### 3. Storage tiers

- **Tier A — aux-data aggregates (default when monitor on).**
  Accumulate since the last aux tick: mean, rms, min, max of
  `shift + delta`. Register as
  `Ftmw/ChirpJitterMean`, `Ftmw/ChirpJitterRms`, etc., alongside the
  existing `ChirpShift`/`ChirpPhaseScore` keys
  (`AcquisitionManager::auxDataTick`). Zero new storage machinery;
  answers the binary question and shows drift trends in the existing
  aux-data viewer.
- **Tier B — accumulated `Phi(f)` (deconvolution-ready, small).**
  Running sums `sum cos(2*pi*f*dt_k)`, `sum sin(2*pi*f*dt_k)` on a
  coarse fixed frequency grid (e.g. 64 points across the digitizer
  Nyquist), written at experiment completion as a small
  semicolon-delimited CSV (`jitterphi.csv`) through `BlackchirpCSV` —
  a few kB regardless of shot count. This is the artifact the
  analysis pipeline divides out.
- **Tier C — full per-shot series (optional, behind a setting).**
  `(shot, shift, delta, fom)` appended in batches; ~16 MB per 10^6
  shots. Only needed for forensics beyond `Phi(f)` (environmental
  correlation at full time resolution). Defer unless a concrete use
  appears; Tier A's per-tick aggregates already correlate against aux
  time series.

### Settings / UI / persistence

- `FtmwConfig`: new bool (+ BC::Store key beside `phase`/`chirp` in
  the `storeValues`/`retrieveValues` pair), accumulator members, and
  the estimator in `preprocessChirp` (or a sibling
  `monitorChirp` invoked from the same call site in
  `AcquisitionManager`).
- GUI: one checkbox beside the existing phase-correction /
  chirp-scoring boxes (`ExperimentTypePage`), tooltip stating it is
  diagnostic-only and clip-tolerant (with the sign-FOM option).
- Viewer: nothing required (aux-data plots cover Tier A); `jitterphi.csv`
  is for offline analysis.

## Implementation sketch

1. `ftmwconfig.h/cpp`: estimator + accumulators + settings keys;
   refactor `preprocessChirp` so monitor and correction share the FOM
   evaluation (correction implies monitor).
2. `acquisitionmanager.cpp`: register/emit the new aux keys when the
   monitor is enabled.
3. Storage: `jitterphi.csv` writer at experiment completion
   (`BlackchirpCSV`; same semicolon schema as the other per-experiment
   CSVs).
4. GUI checkbox + settings plumbing (`experimenttypepage.cpp`).
5. Tests: extend the virtual FTMW digitizer to inject a configurable
   per-shot delay (it already has phase hooks); assert the monitor's
   rms/mean recover the injected distribution and `jitterphi.csv`
   matches the analytic `Phi(f)`.
6. Python module: a small reader for `jitterphi.csv` (the analysis
   pipeline consumes it for spectrum deconvolution).
7. CHANGELOG + a short user-docs section (the existing phase-correction
   docs are the natural home).

Rough scope: ~150–300 LOC in the C++ tree plus tests; no hardware
contract changes; no per-shot storage by default.

## Trigger / priority

Natural trigger: the first time a deep average's high-band intensity
accuracy matters (relative intensities across the band for temperature
or abundance work), or before any campaign where the
`ftmwpipeline` timebase/deconvolution analysis would consume
`jitterphi.csv`. The estimator piece is also the cheapest way to settle
whether sub-sample trigger jitter contributes to the high-SNR lineshape
floor observed on the analysis side.
