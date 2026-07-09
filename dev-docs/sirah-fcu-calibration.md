# Sirah FCU calibration schemes — SHG / polynomial / spline

Implementation plan for replacing the `SirahFcu` tuning math. The
[roadmap entry](devel-roadmap.md#sirah-fcu-calibration-schemes) points
here. This builds on the [Sirah Cobra refresh](sirah-cobra-refresh.md),
which introduced `SirahFcu` as an `NHG` doubling stage but left its
tuning math as a placeholder.

## Implementation status

**Implemented on branch `feature/sirah-cobra-refresh`** (not yet merged;
GUI show/hide and CSV import want a manual click-through, and the whole
model wants co-tuning verification, once the FCU is on the bench). All
five sequencing steps below plus UI integration have landed:

- `FcuCalibration` value type (`data/lif/fcucalibration.*`) + unit tests.
- `SirahFcu` migrated to it; grating keys dropped, scheme/crystal/invert
  and the `polyCoeffs`/`splinePoints` arrays registered. (Surfaced and
  fixed a latent `REGISTER_HARDWARE_ARRAY` macro bug that broke any class
  registering more than one array.)
- Offline tools under `python/tools/` (`fcu_measure.py`, `fcu_fit.py`).
- Generic "Import CSV…" button on `HwArrayEditDialog`.
- Scheme-aware **gating** — an optional `gateKey`/`gateValue` on
  `HwSettingDef`/`HwArraySettingDef` shows only the active scheme's
  settings, following the `displayUnitKey` linkage precedent.
- User + reference docs.

## Problem

`SirahFcu::posToWavelength`/`wavelengthToPos` (`sirahfcu.cpp:175-199`)
are a byte-for-byte copy of `SirahCobra`'s **grating** diffraction law
(`wl = (sin(grazAng) + sin(phi))/grooves`). A doubling crystal has no
grooves and no grazing angle; the registered `stageGratingGroovesPerMm`
/ `stageGrazingAngleDeg` are placeholders the header itself flags as
unknown until calibration (`sirahfcu.h:41-44`). The sine-bar *mechanics*
(motor position ↔ crystal angle) carry over correctly; only the
angle ↔ wavelength law is wrong.

The doubling crystal is angle-tuned for Type-I second-harmonic phase
matching. Its tuning curve is a smooth, monotone, invertible function of
the fundamental wavelength — but it is set by crystal dispersion, not
diffraction.

## Confirmed model (Sirah Autotracker service manual §6, pp. 43-56)

The manual describes the same "Auto FCU" look-up-table tuning Blackchirp
needs. §6.5 (p. 49, Fig 6-7) lists exactly five calibration parameters:

- **Crystal cut angle** — orientation of the crystal's optic axis to its
  face; tabulated per crystal in Table 6-1 (p. 50).
- **Crystal temperature** — "in arbitrary units"; a dispersion knob (the
  temperature variable of a temperature-dependent Sellmeier, used as a
  free fit parameter, not a controlled temperature).
- **Linear offset / angle offset** — the sine-bar mechanism geometry.
- **Screw pitch** — the lead-screw pitch of the sine drive.

The manual's default fit holds *cut angle* and *screw pitch* fixed and
fits *temperature + linear offset + angle offset* — because cut angle
and angle offset are near-degenerate additive angles (adding a
crystal-face **refraction** term separates them). Crystal identity is
selected by the "Conversion Type / Tuning Method" (`SHG BBO`, `SHG KDP`,
…), which picks the Sellmeier equations; a `Parameter >> Invert` toggle
selects the ± branch of the phase-match relation. The manual also offers
**polynomial** fit functions and a crystal-agnostic **spline** mode as
alternatives (§6.6).

Type-I phase-match angle for a negative uniaxial crystal:

    sin^2(theta_pm) = [n_o(lam)^-2 - n_o(lam/2)^-2]
                    / [n_e(lam/2)^-2 - n_o(lam/2)^-2]

with the sine-bar + refraction forward map (fundamental λ → motor pos):

    theta_pm  = f(lam, crystal, temperature)         # phase matching
    alpha_int = theta_pm - cutAngle                  # optic-axis to face
    alpha_ext = asin(n * sin(alpha_int))             # Snell at the face
    x         = linOff - leverLen * sin(angOff - alpha_ext)
    pos       = (motorRes / screwPitch) * x

A prototype (`fcu_model.py`, `fcu_crystals.py`, session scratchpad)
validated this end to end: the 5-parameter fit recovers a synthetic
table below noise, the pos↔λ round-trip is exact, and best-effort
BBO (Eimerl 1987) + KDP (Zernike 1964) Sellmeier place every Table 6-1
crystal's phase-match angle in its tabulated neighborhood, all monotone.

## Scope decisions

- **Evaluation only.** Blackchirp does *not* fit. Coefficients/parameters
  are determined offline (an interactive Python tool: set position →
  hand-optimize the crystal → record encoder → fit → visualize) and
  imported. No fitting (GSL nonlinear least squares) in the acquisition
  app. This keeps the app's scope to *evaluate a pre-fit model in both
  directions*, which is all the driver needs.
- **Three schemes**, user-selectable per device:
  - *Physical* — best-effort BBO + KDP Type-I SHG (the FCU is `NHG`-only;
    no DFG/LiNbO₃ path for now). The functional form must be correct;
    exact agreement with Sirah's internal curve is a non-goal, since the
    user fits *this* model to *their* data. The offline fitter and the
    C++ evaluator share the same Sellmeier so imported parameters
    reproduce exactly.
  - *Polynomial* — import forward (λ→pos) and inverse (pos→λ) coefficient
    lists, so neither direction needs a root find.
  - *Spline* — import a `(λ, pos)` point table; GSL rebuilds two
    interpolating splines (one per direction). Sidesteps the scipy
    smoothing-B-spline `(t,c,k)` ↔ GSL interpolation-spline mismatch.
- **Single tuning curve.** The compensator is physically geared to the
  crystal angle stage, so only one calibration is needed.

## Design detail

### `FcuCalibration` value type

A free-standing, hardware-free value type (in `data/lif/`, sibling of
`LifConversion`) so it unit-tests in CI with no hardware. It holds the
active scheme + its coefficients and exposes:

    double wavelengthToPos(double lamNm) const;   // fundamental nm -> steps
    double posToWavelength(double pos) const;      // steps -> fundamental nm
    bool   isValid() const;                        // assembled + monotone

Assembled from a settings snapshot (like `LifConversion`), so both the
driver and any future config/GUI layer build it without touching a
threaded device. Per-scheme internals:

- *Physical* — best-effort `n_o`/`n_e` for `{BBO, KDP}`, the phase-match
  + sine-bar + refraction forward map above, `invert` sign flag; inverse
  by a bracketed 1-D root find (the curve is monotone over each crystal's
  range). `leverLength`/`motorResolution` are fixed mechanics.
- *Polynomial* — Horner evaluation of the two imported coefficient lists.
- *Spline* — two `gsl_interp_steffen` (monotone) splines over the
  imported points, sorted per axis.

### `SirahFcu` settings

Delete the grating keys (`sGrooves`, `sGrazingAngle`); the tuning law
leaves the driver. Add via the registry macros (not constructor
`setDefault`):

- `CalibrationScheme` `Q_ENUM_NS { Physical, Polynomial, Spline }` —
  `Important` scalar (auto-renders as a combobox, per the `Op` enum in
  `liffreqconversionstage.cpp:10-21`).
- *Physical* scalars — `CrystalType { BBO, KDP }` enum, `cutAngleDeg`,
  `temperature`, `linearOffsetMm`, `angleOffsetDeg`, `screwPitchMm`,
  `invert` bool. Few numbers, entered directly (mirrors the manual's
  Fig 6-7 dialog); `cutAngleDeg` default seeded from Table 6-1.
- *Polynomial* — forward + inverse coefficient arrays (CSV-imported).
- *Spline* — `(λ, pos)` point array (CSV-imported).
- Keep the existing motor/backlash keys (`sStart`, `sHigh`, `sRamp`,
  `sMax`, `sbls`) — those are motion control, not the tuning law.

`hwReadSettings()` builds the `FcuCalibration` from the active scheme's
settings and caches it; `posToWavelength`/`wavelengthToPos` become thin
delegations to it.

### Generic CSV import

No settings dialog in the tree imports CSV today. Add an
**"Import CSV…"** button to `HwArrayEditDialog` (`gui/dialog/`): read a
semicolon-delimited file whose columns match the array's sub-keys and
populate the table rows. Generic — every array setting benefits, not
just the FCU coefficient/point tables.

### Offline reference fitter

Clean the prototype into a documented recipe under `python/` so users
fit against the *same* Sellmeier Blackchirp evaluates (keeping the two
self-consistent), with the interactive tune/record/fit/visualize loop
the manual's calibration procedure describes.

## Sequencing

1. `FcuCalibration` value type + `tst_fcucalibration.cpp` (pure math, no
   hardware): round-trips for physical (BBO + KDP), polynomial, and
   spline; monotonicity; the Table 6-1 cut-angle sanity check.
2. `SirahFcu` settings migration — drop grating keys, add scheme +
   per-scheme settings, build/cached `FcuCalibration` in
   `hwReadSettings()`, delegate the tuning methods.
3. Generic "Import CSV…" button on `HwArrayEditDialog`.
4. Offline reference fitter into `python/`.

## Testing

- Unit (`tst_fcucalibration.cpp`, Qt-Test, CI, no hardware): pos↔λ
  round-trips within tolerance for each scheme; monotone/invertible over
  each crystal's fundamental band; physical phase-match angle near the
  Table 6-1 cut angle; malformed-input rejection (`isValid()` false).
- Manual, on the bench when the FCU is online: co-tune the grating +
  doubler across the OH band (~282 nm out / ~564 nm fundamental, BBO
  ~52-57°, well away from 90°), verifying the LIF axis reads the doubled
  wavelength and each scheme reproduces the recorded calibration.

## Open notes

- Scheme-dependent settings all render at once (no conditional
  visibility in `HwSettingsWidget`); acceptable, like the existing gated
  `harmonic`. A scheme-aware panel is a possible later polish.
- Near-90° crystals (deep-UV `SHG-205`/`THG-200` BBO at 77°) have steep
  curves where a low-order polynomial is a poor proxy; the physical model
  or a dense spline is the answer there. The FCU's day-1 BBO doubler is
  far from 90°.
- DFG/LiNbO₃ and multi-crystal (THG) chains are out of scope — the FCU is
  `NHG`-only. Adding them is a Sellmeier + process extension to
  `FcuCalibration`, not a structural change.
