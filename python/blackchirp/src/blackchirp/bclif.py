"""LIF data containers for the Blackchirp Python module.

The :class:`BCLIF` class loads a complete LIF acquisition (the
``lif/`` subdirectory of an experiment folder) and exposes scan-axis
metadata, per-point trace access, and aggregating helpers that mirror
Blackchirp's own LIF processing tab. :class:`BCLifTrace` is the
single-point counterpart to :class:`~blackchirp.BCFid`: it holds the
decoded waveform for one scan point and reproduces the C++
``LifTrace::processXY`` and ``LifTrace::integrate`` semantics so that
integrated yields match the GUI bit-for-bit.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal as spsig

_PROC_INT_KEYS = frozenset(
    {
        "LifGateStartPoint",
        "LifGateEndPoint",
        "RefGateStartPoint",
        "RefGateEndPoint",
        "SavGolWindow",
        "SavGolPoly",
    }
)
_PROC_FLOAT_KEYS = frozenset({"LowPassAlpha"})
_PROC_BOOL_KEYS = frozenset({"SavGolEnabled"})

_TIME_UNIT_SCALES = {
    "s": 1.0,
    "ms": 1.0e3,
    "us": 1.0e6,
    "μs": 1.0e6,
    "ns": 1.0e9,
}

# Laser-unit conversion factors. These MUST match the constants in the C++
# writer (``src/data/lif/lifunits.cpp``) so that a topology read here
# round-trips against the values Blackchirp serialized.
_CM1_PER_EV = 8065.543937
_EV_PER_CM1 = 1.239841984e-4
_GHZ_PER_CM1 = 2.99792458e10 / 1.0e9  # GHz per cm^-1 (≈ 29.9792458)
_NM_CM1 = 1.0e7  # vacuum wavelength: lambda_nm = _NM_CM1 / cm^-1

# Accepted spellings for the four laser units, mapped to a canonical key.
# Includes the exact ``header.csv`` labels emitted by C++ ``unitLabel`` (e.g.
# the non-ASCII ``cm⁻¹``) so ``unit=lif.laser_units`` works directly.
_UNIT_ALIASES = {
    "cm-1": "cm-1",
    "cm^-1": "cm-1",
    "cm−1": "cm-1",
    "cm⁻¹": "cm-1",
    "1/cm": "cm-1",
    "wavenumber": "cm-1",
    "wavenumbers": "cm-1",
    "nm": "nm",
    "nanometer": "nm",
    "nanometers": "nm",
    "ghz": "GHz",
    "ev": "eV",
}


def _normalize_unit(unit: str) -> str:
    """Map any accepted unit spelling to its canonical key.

    Args:
        unit: A laser-unit string (case-insensitive; accepts the
            ``header.csv`` labels including the non-ASCII ``cm⁻¹``).

    Returns:
        One of ``"cm-1"``, ``"nm"``, ``"GHz"``, ``"eV"``.

    Raises:
        ValueError: If ``unit`` is not a recognized laser unit.
    """

    key = str(unit).strip().lower()
    try:
        return _UNIT_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown laser unit {unit!r}; expected one of cm-1, nm, GHz, eV"
        ) from None


def _to_cm1(value, unit: str):
    """Convert ``value`` (in ``unit``) to vacuum wavenumber (cm^-1)."""

    u = _normalize_unit(unit)
    arr = np.asarray(value, dtype=float)
    if u == "cm-1":
        out = arr
    elif u == "nm":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(arr > 0, _NM_CM1 / arr, np.nan)
    elif u == "GHz":
        out = arr / _GHZ_PER_CM1
    else:  # eV
        out = arr * _CM1_PER_EV
    return out if arr.ndim else float(out)


def _from_cm1(cm1, unit: str):
    """Convert vacuum wavenumber ``cm1`` (cm^-1) to ``unit``."""

    u = _normalize_unit(unit)
    arr = np.asarray(cm1, dtype=float)
    if u == "cm-1":
        out = arr
    elif u == "nm":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(arr > 0, _NM_CM1 / arr, np.nan)
    elif u == "GHz":
        out = arr * _GHZ_PER_CM1
    else:  # eV
        out = arr * _EV_PER_CM1
    return out if arr.ndim else float(out)


def _apply_op(op: str, n, in_coeffs):
    """Combine input affine coefficients per a conversion op (cm^-1).

    Each beam is ``A * fundamental + B``; this returns the ``(A, B)`` of a
    node's output given the ``(A, B)`` of its ordered inputs.

    Args:
        op: ``"NHG"``, ``"SFG"``, or ``"DFG"``.
        n: Harmonic order for ``NHG`` (ignored otherwise).
        in_coeffs: List of ``(A, B)`` tuples, one per input.

    Returns:
        The output ``(A, B)`` tuple.

    Raises:
        ValueError: If ``op`` is not a recognized conversion op.
    """

    if op == "NHG":
        a0, b0 = in_coeffs[0]
        return (n * a0, n * b0)
    if op == "SFG":
        (a0, b0), (a1, b1) = in_coeffs[0], in_coeffs[1]
        return (a0 + a1, b0 + b1)
    if op == "DFG":
        (a0, b0), (a1, b1) = in_coeffs[0], in_coeffs[1]
        return (a0 - a1, b0 - b1)
    raise ValueError(f"Unknown conversion op {op!r}")


def _parse_proc_value(key: str, raw):
    """Coerce a ``processing.csv`` cell to a typed Python value.

    Strings and numeric values are both accepted so the loader is
    robust to the various dtypes pandas may infer from the on-disk
    CSV.
    """

    if key in _PROC_INT_KEYS:
        return int(raw)
    if key in _PROC_FLOAT_KEYS:
        return float(raw)
    if key in _PROC_BOOL_KEYS:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() == "true"
    return raw


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _trapezoid_sample_space(y: np.ndarray, start: int, end: int) -> float:
    """Trapezoidal integral in sample-index space, matching C++.

    Reproduces ``LifTrace::integrate`` exactly: gates are clamped with
    ``qBound`` semantics and the inner sum stops one segment short of
    the upper gate, so the upper-bound sample at index ``end`` itself
    is not included in the partial sum.
    """

    n = len(y)
    if n < 2:
        return 0.0
    ls = max(0, min(int(start), n - 2))
    le = max(ls + 1, min(int(end), n - 1))
    seg = y[ls:le]
    if seg.size < 2:
        return 0.0
    return float(0.5 * (seg[:-1].sum() + seg[1:].sum()))


def _decode_base36_column(series: pd.Series) -> np.ndarray:
    ic = np.frompyfunc(int, 2, 1)
    return ic(series.to_numpy(dtype="str"), 36).astype(np.int64)


class BCLifTrace:
    """Container for a single LIF scan-point trace.

    Reads ``lif/N.csv`` for one ``(lIndex, dIndex)`` pair, decodes the
    base-36-encoded accumulated samples to per-shot voltages, and
    exposes the smoothing and integration operations that
    :class:`BCLIF` uses to build axis slices and 2D images.

    Args:
        num: File number for the trace (``N`` in ``lif/N.csv``).
        path: Experiment folder path.
        params: Row from ``lifparams.csv`` corresponding to this point.
        sep: CSV delimiter for the experiment.
        proc: Default processing settings, parsed from
            ``lif/processing.csv``.

    Attributes:
        params (pd.Series): The matching ``lifparams.csv`` row.
        proc (dict): Default processing settings.
        shots (int): Number of shots accumulated at this point.
        lifsize (int): Number of LIF samples per shot.
        refsize (int): Number of reference samples per shot
            (``0`` if no reference channel was recorded).
        spacing (float): Sample spacing, in seconds.
        lifymult (float): LIF y-multiplier (volts per integer sample).
        refymult (float): Reference y-multiplier (volts per integer
            sample). ``0`` when no reference channel was recorded.
    """

    def __init__(
        self,
        num: int,
        path: str,
        params: pd.Series,
        sep: str,
        proc: dict,
    ):
        self._num = int(num)
        self.params = params.copy()
        self._sep = sep
        self.proc = proc

        self.shots = int(self.params["shots"])
        self.lifsize = int(self.params["lifsize"])
        self.refsize = int(self.params["refsize"])
        self.spacing = float(self.params["spacing"])
        self.lifymult = float(self.params["lifymult"])
        self.refymult = float(self.params["refymult"])

        df = pd.read_csv(
            os.path.join(path, f"lif/{self._num}.csv"),
            sep=sep,
            header=0,
            dtype="str",
            keep_default_na=False,
        )

        lif_raw = _decode_base36_column(df["lif"])
        self._lif_raw = lif_raw
        scale = self.lifymult / self.shots if self.shots else self.lifymult
        self._lif = lif_raw.astype(np.float64) * scale

        if "ref" in df.columns and self.refsize > 0:
            ref_raw = _decode_base36_column(df["ref"])
            self._ref_raw = ref_raw
            ref_scale = self.refymult / self.shots if self.shots else self.refymult
            self._ref = ref_raw.astype(np.float64) * ref_scale
        else:
            self._ref_raw = None
            self._ref = None

    def x(self, units: str = "s") -> np.ndarray:
        """Compute the time array for the trace.

        Args:
            units: One of ``"s"``, ``"ms"``, ``"us"`` (or ``"μs"``),
                ``"ns"``. ``"s"`` is the default; the other choices
                rescale the array for plotting convenience and have no
                effect on stored sample spacing.

        Returns:
            1D numpy array of sample times in the requested units.

        Raises:
            ValueError: If ``units`` is not a recognised time-unit
                string.
        """
        try:
            scale = _TIME_UNIT_SCALES[units]
        except KeyError as err:
            raise ValueError(
                f"Unknown time unit {units!r}; choose from "
                f"{sorted(_TIME_UNIT_SCALES)}"
            ) from err
        return np.arange(self.lifsize) * (self.spacing * scale)

    def lif(self) -> np.ndarray:
        """Return the per-shot LIF waveform, in volts."""
        return self._lif

    def ref(self) -> Optional[np.ndarray]:
        """Return the per-shot reference waveform, or ``None``.

        ``None`` is returned for single-channel acquisitions (those
        whose ``lifparams`` row has ``refsize == 0``).
        """
        return self._ref

    def has_ref(self) -> bool:
        """Indicate whether a reference channel was recorded."""
        return self._ref is not None

    def xy(self, units: str = "s"):
        """Return time and waveform arrays as a tuple.

        For with-reference traces, the tuple is ``(x, lif, ref)``;
        for single-channel traces, ``(x, lif)``. The ``units``
        argument controls the time axis only (see :meth:`x`).
        """
        if self.has_ref():
            return self.x(units), self._lif, self._ref
        return self.x(units), self._lif

    def _resolve_proc(
        self,
        *,
        lif_start=None,
        lif_end=None,
        ref_start=None,
        ref_end=None,
        low_pass_alpha=None,
        savgol_window=None,
        savgol_poly=None,
        savgol_enabled=None,
    ) -> dict:
        def _pick(value, key, cast, fallback):
            if value is not None:
                return cast(value)
            if key in self.proc:
                return cast(self.proc[key])
            return fallback

        return {
            "lif_start": _pick(lif_start, "LifGateStartPoint", int, 0),
            "lif_end": _pick(lif_end, "LifGateEndPoint", int, max(self.lifsize - 1, 0)),
            "ref_start": _pick(ref_start, "RefGateStartPoint", int, 0),
            "ref_end": _pick(ref_end, "RefGateEndPoint", int, max(self.refsize - 1, 0)),
            "low_pass_alpha": _pick(low_pass_alpha, "LowPassAlpha", float, 0.0),
            "savgol_window": _pick(savgol_window, "SavGolWindow", int, 11),
            "savgol_poly": _pick(savgol_poly, "SavGolPoly", int, 3),
            "savgol_enabled": (
                _coerce_bool(savgol_enabled)
                if savgol_enabled is not None
                else _coerce_bool(self.proc.get("SavGolEnabled", False))
            ),
        }

    def _process_y(
        self,
        y: np.ndarray,
        alpha: float,
        savgol_enabled: bool,
        sgw: int,
        sgp: int,
    ) -> np.ndarray:
        out = np.asarray(y, dtype=np.float64)
        if alpha > 1e-5:
            b = [1.0 - alpha]
            a = [1.0, -alpha]
            zi = np.array([alpha * out[0]])
            out, _ = spsig.lfilter(b, a, out, zi=zi)
        if savgol_enabled:
            out = spsig.savgol_filter(out, sgw, sgp)
        return out

    def smooth(
        self,
        low_pass=None,
        savgol=None,
        *,
        low_pass_alpha=None,
        savgol_window=None,
        savgol_poly=None,
    ) -> np.ndarray:
        """Apply the IIR-then-Sav-Gol filter chain to the LIF channel.

        Args:
            low_pass: Override IIR-filter use. ``None`` follows
                ``processing.csv`` (apply if ``LowPassAlpha`` > 0);
                ``True`` forces the IIR on, ``False`` forces it off,
                and a numeric value overrides the alpha coefficient
                directly.
            savgol: Override Savitzky-Golay use. ``None`` follows
                ``processing.csv`` (``SavGolEnabled``); ``True`` /
                ``False`` force on / off.
            low_pass_alpha: Override the IIR alpha coefficient.
            savgol_window: Override the Sav-Gol window length.
            savgol_poly: Override the Sav-Gol polynomial order.

        Returns:
            The smoothed LIF waveform as a 1D numpy array.
        """

        proc_alpha = (
            float(low_pass_alpha)
            if low_pass_alpha is not None
            else float(self.proc.get("LowPassAlpha", 0.0))
        )
        if low_pass is False:
            alpha = 0.0
        elif low_pass is True:
            alpha = proc_alpha
        elif low_pass is None:
            alpha = proc_alpha
        else:
            alpha = float(low_pass)

        sgw = (
            int(savgol_window)
            if savgol_window is not None
            else int(self.proc.get("SavGolWindow", 11))
        )
        sgp = (
            int(savgol_poly)
            if savgol_poly is not None
            else int(self.proc.get("SavGolPoly", 3))
        )
        if savgol is None:
            sg_enabled = _coerce_bool(self.proc.get("SavGolEnabled", False))
        else:
            sg_enabled = bool(savgol)

        return self._process_y(self._lif, alpha, sg_enabled, sgw, sgp)

    def integrate(
        self,
        lif_start=None,
        lif_end=None,
        ref_start=None,
        ref_end=None,
        **proc_overrides,
    ) -> float:
        """Integrate the LIF channel and return a (possibly ratioed) float.

        The trapezoidal sum is taken in **sample-index space**: gates
        are sample indices, not times, and ``dx`` is one sample. This
        matches Blackchirp's GUI display number bit-for-bit.

        **Units.** Without a reference channel, the result is in
        ``V·sample`` — equivalently, voltage integrated against an
        integer index. To convert to a time-domain integral, multiply
        by ``self.spacing`` (V·s) or ``self.spacing * 1e9`` (V·ns).
        With a reference channel, the result is the dimensionless
        ratio ``lif_integral / ref_integral``; the conversion factors
        cancel because both halves of the ratio are computed in the
        same sample-index space. If the reference integral is
        numerically zero the unratioed LIF integral (in V·sample) is
        returned instead.

        Args:
            lif_start: Override LIF gate start sample.
            lif_end: Override LIF gate end sample.
            ref_start: Override reference gate start sample.
            ref_end: Override reference gate end sample.
            **proc_overrides: Optional ``low_pass_alpha``,
                ``savgol_window``, ``savgol_poly``, ``savgol_enabled``
                overrides.

        Returns:
            Integrated value as a float.
        """

        cfg = self._resolve_proc(
            lif_start=lif_start,
            lif_end=lif_end,
            ref_start=ref_start,
            ref_end=ref_end,
            **proc_overrides,
        )

        y_lif = self._process_y(
            self._lif,
            cfg["low_pass_alpha"],
            cfg["savgol_enabled"],
            cfg["savgol_window"],
            cfg["savgol_poly"],
        )
        lif_int = _trapezoid_sample_space(y_lif, cfg["lif_start"], cfg["lif_end"])

        if not self.has_ref():
            return lif_int

        y_ref = self._process_y(
            self._ref,
            cfg["low_pass_alpha"],
            cfg["savgol_enabled"],
            cfg["savgol_window"],
            cfg["savgol_poly"],
        )
        ref_int = _trapezoid_sample_space(y_ref, cfg["ref_start"], cfg["ref_end"])

        if abs(ref_int) < 1e-12:
            return lif_int
        return lif_int / ref_int


class BCLIF:
    """Container for a complete LIF scan.

    Loads ``lif/lifparams.csv`` and ``lif/processing.csv`` and reads
    the scan-axis parameters (``DelayStart``, ``LaserStart``, etc.)
    from the supplied experiment header. Exposes per-point access via
    :meth:`get_trace` and aggregating helpers (:meth:`delay_slice`,
    :meth:`laser_slice`, :meth:`image`) that integrate every present
    point with a single processing-override surface.

    This class is not intended to be instantiated directly; it is
    constructed by :class:`~blackchirp.BCExperiment` whenever the
    experiment folder contains a ``lif/`` subdirectory.

    Args:
        path: Experiment folder path.
        sep: CSV delimiter for the experiment.
        header: Experiment header DataFrame (``BCExperiment.header``)
            used to read ``LifConfig`` scan-axis rows.

    Attributes:
        lifparams (pd.DataFrame): Contents of ``lif/lifparams.csv``.
        proc (dict): Contents of ``lif/processing.csv`` parsed to
            typed values (ints, floats, bools).
        delay_points (int): Number of delay-axis points.
        delay_start (float): First delay value, in ``delay_units``.
        delay_step (float): Delay-axis step, in ``delay_units``.
        delay_units (str): Units of ``delay_start`` / ``delay_step``.
        laser_points (int): Number of laser-axis points.
        laser_start (float): First laser value, in ``laser_units``.
        laser_step (float): Laser-axis step, in ``laser_units``.
        laser_units (str): Units of ``laser_start`` / ``laser_step``.
        has_ref (bool): ``True`` if any ``lifparams`` row has
            ``refsize > 0``.
        numtraces (int): Number of populated scan points
            (``len(lifparams)``).
        has_topology (bool): ``True`` if a non-identity frequency-conversion
            topology (``liftopology.csv``) was recorded for this experiment.
        stages (list[str]): Conversion-stage hardware keys in DAG order
            (empty when ``has_topology`` is ``False``).
        final_stage (str | None): Hardware key of the stage that produces
            the output (excitation) beam, or ``None`` for the identity case.
        laser_key (str | None): Hardware key of the tunable laser the
            topology was captured against, or ``None`` for the identity case.
    """

    def __init__(
        self,
        path: str,
        sep: str,
        header: pd.DataFrame,
        liftopology: Optional[pd.DataFrame] = None,
    ):
        self.path = path
        self._sep = sep
        self._topology = liftopology

        self.lifparams = pd.read_csv(
            os.path.join(self.path, "lif/lifparams.csv"),
            sep=self._sep,
            header=0,
            keep_default_na=False,
        )

        proc_df = pd.read_csv(
            os.path.join(self.path, "lif/processing.csv"),
            sep=self._sep,
            header=0,
            keep_default_na=False,
        )
        self.proc = {}
        for _, row in proc_df.iterrows():
            key = row["ObjKey"]
            self.proc[key] = _parse_proc_value(key, row["Value"])

        cfg = header.query("ObjKey == 'LifConfig'")

        def _value(value_key: str) -> str:
            rows = cfg.query(f"ValueKey == '{value_key}'")
            if rows.empty:
                raise KeyError(
                    f"header.csv is missing required LifConfig/{value_key} row"
                )
            return str(rows.Value.iloc[0])

        def _units(value_key: str) -> str:
            rows = cfg.query(f"ValueKey == '{value_key}'")
            if rows.empty:
                return ""
            unit = rows.Units.iloc[0]
            if unit is None or (isinstance(unit, float) and unit != unit):
                return ""
            return str(unit)

        self.delay_points = int(_value("DelayPoints"))
        self.delay_start = float(_value("DelayStart"))
        self.delay_step = float(_value("DelayStep"))
        self.delay_units = _units("DelayStart")
        self.laser_points = int(_value("LaserPoints"))
        self.laser_start = float(_value("LaserStart"))
        self.laser_step = float(_value("LaserStep"))
        self.laser_units = _units("LaserStart")

        self.numtraces = len(self.lifparams)
        self.has_ref = bool((self.lifparams["refsize"].astype(int) > 0).any())

        self._index = {}
        for row_idx, row in self.lifparams.iterrows():
            self._index[(int(row["lIndex"]), int(row["dIndex"]))] = int(row_idx)

        self._build_conversion()

    def _file_num(self, l_index: int, d_index: int) -> int:
        return d_index * self.laser_points + l_index

    def get_trace(self, l_index: int, d_index: int) -> BCLifTrace:
        """Load a single ``BCLifTrace`` from disk.

        Args:
            l_index: Laser-axis index.
            d_index: Delay-axis index.

        Returns:
            A freshly loaded :class:`BCLifTrace`.

        Raises:
            KeyError: If no row exists in ``lifparams.csv`` for the
                requested ``(l_index, d_index)`` pair (i.e. the scan
                point was not acquired).
        """

        key = (int(l_index), int(d_index))
        if key not in self._index:
            raise KeyError(
                f"No trace at (lIndex={l_index}, dIndex={d_index}) in lifparams.csv"
            )
        params = self.lifparams.iloc[self._index[key]]
        return BCLifTrace(
            self._file_num(int(l_index), int(d_index)),
            self.path,
            params,
            self._sep,
            self.proc,
        )

    def delay_axis(self) -> Tuple[np.ndarray, str]:
        """Return the delay-axis sample values and their units string."""
        arr = self.delay_start + np.arange(self.delay_points) * self.delay_step
        return arr, self.delay_units

    def laser_axis(self) -> Tuple[np.ndarray, str]:
        """Return the laser-axis sample values and their units string."""
        arr = self.laser_start + np.arange(self.laser_points) * self.laser_step
        return arr, self.laser_units

    def _build_conversion(self):
        """Parse ``liftopology.csv`` into an affine model of the DAG.

        Every beam in the frequency-conversion graph is an affine function
        of the single tunable grating fundamental (cm^-1),
        ``beam = A * fundamental + B``. This walks the DAG once (in
        dependency order) to recover ``(A, B)`` for each stage's inputs and
        output, so any beam can be evaluated from — or inverted to — the
        fundamental in closed form. Sets ``has_topology``, ``stages``,
        ``final_stage``, and ``laser_key``. The identity/no-conversion case
        (absent file) leaves ``stages`` empty and treats laser == output.
        """

        df = self._topology
        self.stages = []
        self.final_stage = None
        self.laser_key = None
        self.has_topology = df is not None and len(df) > 0
        self._out_coeff = {}
        self._in_coeff = {}
        self._op = {}
        self._n = {}
        self._final = {}
        if not self.has_topology:
            return

        stage_keys = {str(r["StageKey"]) for _, r in df.iterrows()}

        def classify(token):
            token = str(token)
            if token == "":
                return None
            if token.startswith("Fixed:"):
                return ("fixed", float(token[len("Fixed:") :]))
            if token in stage_keys:
                return ("stage", token)
            # Any other token is the tunable laser source, and the token
            # itself is the active laser's hardware key.
            self.laser_key = token
            return ("laser", None)

        nodes = []
        for _, r in df.iterrows():
            sk = str(r["StageKey"])
            harm = str(r["Harmonic"]).strip()
            is_final = str(r["IsFinal"]).strip().lower() == "true"
            inputs = [
                ref
                for ref in (classify(r["Input0"]), classify(r["Input1"]))
                if ref is not None
            ]
            nodes.append(
                {
                    "stageKey": sk,
                    "op": str(r["Op"]),
                    "n": int(harm) if harm else None,
                    "inputs": inputs,
                }
            )
            self.stages.append(sk)
            self._op[sk] = str(r["Op"])
            self._n[sk] = int(harm) if harm else None
            self._final[sk] = is_final
            if is_final:
                self.final_stage = sk

        # Resolve per-beam affine coefficients, deferring any node whose
        # stage inputs are not yet resolved until a later pass.
        pending = list(nodes)
        while pending:
            still = []
            progressed = False
            for node in pending:
                deps = [ref[1] for ref in node["inputs"] if ref[0] == "stage"]
                if not all(d in self._out_coeff for d in deps):
                    still.append(node)
                    continue
                in_coeffs = []
                for ref in node["inputs"]:
                    if ref[0] == "laser":
                        in_coeffs.append((1.0, 0.0))
                    elif ref[0] == "fixed":
                        in_coeffs.append((0.0, ref[1]))
                    else:
                        in_coeffs.append(self._out_coeff[ref[1]])
                self._in_coeff[node["stageKey"]] = in_coeffs
                self._out_coeff[node["stageKey"]] = _apply_op(
                    node["op"], node["n"], in_coeffs
                )
                progressed = True
            if not progressed:
                raise ValueError(
                    "liftopology.csv has an unresolvable stage reference or cycle"
                )
            pending = still

    def _coeff(self, at: str, side: str):
        """Return the ``(A, B)`` affine coefficients for one beam location."""

        if at == "laser":
            return (1.0, 0.0)
        if at == "final":
            if not self.has_topology:
                return (1.0, 0.0)
            at = self.final_stage
        if at not in self._out_coeff:
            raise KeyError(f"Unknown stage {at!r}; known stages: {self.stages}")
        if side == "output":
            return self._out_coeff[at]
        if side == "input":
            return self._in_coeff[at][0]
        raise ValueError(f"side must be 'input' or 'output', got {side!r}")

    def fundamental(self, value, at="final", side="output", unit="cm-1"):
        """Solve for the grating fundamental from a beam value at one location.

        Because every beam is affine in the tunable fundamental, this is an
        exact inversion (no numerics). Accepts a scalar or a NumPy array and
        matches the input shape on output.

        Args:
            value: Beam value(s) at the location, expressed in ``unit``.
            at: A stage's hardware key, ``"final"`` (the output beam), or
                ``"laser"`` (the fundamental itself).
            side: ``"output"`` or ``"input"`` of the stage (the primary,
                tunable-path input for a two-input mixing stage). Ignored
                for ``at="laser"``.
            unit: Unit of both ``value`` and the returned fundamental — one
                of ``cm-1``, ``nm``, ``GHz``, ``eV`` (or a ``header.csv``
                label such as ``cm⁻¹``).

        Returns:
            The grating fundamental in ``unit`` (scalar or array, matching
            ``value``). For an experiment with no conversion, the
            fundamental equals ``value`` at ``at`` in ``{"final", "laser"}``.

        Raises:
            ValueError: If the location does not depend on the tunable laser
                (e.g. a fixed mixing beam), so it cannot be inverted.
            KeyError: If ``at`` names an unknown stage.

        Example:
            >>> lif.fundamental(280.0, at="final", side="output", unit="nm")
            560.0
        """

        u = _normalize_unit(unit)
        a, b = self._coeff(at, side)
        if a == 0.0:
            raise ValueError(
                f"Location at={at!r} side={side!r} does not depend on the "
                "tunable laser; cannot solve for the fundamental"
            )
        return _from_cm1((_to_cm1(value, u) - b) / a, u)

    def at_stage(self, fundamental, at="final", side="output", unit="cm-1"):
        """Evaluate the beam at one location for a given grating fundamental.

        Inverse of :meth:`fundamental`. Accepts a scalar or a NumPy array of
        fundamentals and matches the input shape on output, so a whole laser
        sweep can be mapped in one call.

        Args:
            fundamental: Grating fundamental value(s) in ``unit``.
            at: A stage's hardware key, ``"final"``, or ``"laser"``.
            side: ``"output"`` or ``"input"`` (primary input) of the stage.
            unit: Unit of both ``fundamental`` and the result.

        Returns:
            The beam value(s) at ``at``/``side`` in ``unit`` (scalar or
            array, matching ``fundamental``).

        Raises:
            KeyError: If ``at`` names an unknown stage.

        Example:
            >>> lif.at_stage(560.0, at="final", side="output", unit="nm")
            280.0
        """

        u = _normalize_unit(unit)
        a, b = self._coeff(at, side)
        return _from_cm1(a * _to_cm1(fundamental, u) + b, u)

    def stage_frequencies(
        self,
        value=None,
        *,
        fundamental=None,
        at="final",
        side="output",
        unit="cm-1",
    ) -> pd.DataFrame:
        """Tabulate every stage's inputs and output for a single frequency.

        Give it either a known beam ``value`` at a location (``at``/``side``)
        or a ``fundamental`` directly; it solves for the fundamental and
        evaluates the whole graph. This is the single-frequency, descriptive
        counterpart to :meth:`fundamental`/:meth:`at_stage` — for a bulk
        sweep, call those with an array (an ``(N, M)`` grid is
        ``np.stack([lif.at_stage(f, at=s) for s in lif.stages], axis=1)``).

        Args:
            value: A known beam value at ``at``/``side``, in ``unit``.
                Mutually exclusive with ``fundamental``.
            fundamental: The grating fundamental in ``unit``. Mutually
                exclusive with ``value``.
            at: Location that ``value`` refers to (stage key, ``"final"``,
                or ``"laser"``).
            side: ``"output"`` or ``"input"`` that ``value`` refers to.
            unit: Unit for the input and every value in the returned frame.

        Returns:
            A :class:`pandas.DataFrame` indexed by ``"laser"`` followed by
            each stage's hardware key, with columns ``op``, ``isfinal``,
            ``input0``, ``input1`` (``NaN`` for single-input stages), and
            ``output``, all in ``unit``. For an experiment with no
            conversion, only the ``laser`` row is returned.

        Raises:
            ValueError: If neither or both of ``value``/``fundamental`` are
                given, or a non-scalar is passed.
            KeyError: If ``at`` names an unknown stage.
        """

        u = _normalize_unit(unit)
        if (value is None) == (fundamental is None):
            raise ValueError("provide exactly one of value= or fundamental=")
        source = value if value is not None else fundamental
        if np.ndim(source) != 0:
            raise ValueError(
                "stage_frequencies is single-frequency; use fundamental() or "
                "at_stage() for array inputs"
            )
        if fundamental is not None:
            f_cm1 = _to_cm1(fundamental, u)
        else:
            a, b = self._coeff(at, side)
            if a == 0.0:
                raise ValueError(
                    f"Location at={at!r} side={side!r} does not depend on the "
                    "tunable laser; cannot solve for the fundamental"
                )
            f_cm1 = (_to_cm1(value, u) - b) / a

        def evaluate(coeff):
            ca, cb = coeff
            return _from_cm1(ca * f_cm1 + cb, u)

        f_disp = _from_cm1(f_cm1, u)
        index = ["laser"]
        rows = [
            {
                "op": None,
                "isfinal": False,
                "input0": f_disp,
                "input1": np.nan,
                "output": f_disp,
            }
        ]
        for sk in self.stages:
            in_coeffs = self._in_coeff[sk]
            index.append(sk)
            rows.append(
                {
                    "op": self._op[sk],
                    "isfinal": self._final[sk],
                    "input0": evaluate(in_coeffs[0]),
                    "input1": (
                        evaluate(in_coeffs[1]) if len(in_coeffs) > 1 else np.nan
                    ),
                    "output": evaluate(self._out_coeff[sk]),
                }
            )
        return pd.DataFrame(
            rows,
            index=index,
            columns=["op", "isfinal", "input0", "input1", "output"],
        )

    def delay_slice(
        self,
        l_index: int,
        fill=np.nan,
        **proc,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate every present trace at one laser index.

        Args:
            l_index: Laser-axis index to slice.
            fill: Value substituted at delay indices that have no
                acquired trace (``np.nan`` by default; pass ``0.0`` to
                represent missing points as zero).
            **proc: Optional processing overrides, forwarded to
                :meth:`BCLifTrace.integrate`.

        Returns:
            Tuple ``(delay_axis, integrals)``. Length of both is
            ``delay_points``.
        """

        delays, _ = self.delay_axis()
        out = np.full(self.delay_points, fill, dtype=np.float64)
        for d in range(self.delay_points):
            if (int(l_index), d) in self._index:
                out[d] = self.get_trace(l_index, d).integrate(**proc)
        return delays, out

    def laser_slice(
        self,
        d_index: int,
        fill=np.nan,
        **proc,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate every present trace at one delay index.

        Args:
            d_index: Delay-axis index to slice.
            fill: Value substituted at laser indices that have no
                acquired trace.
            **proc: Optional processing overrides, forwarded to
                :meth:`BCLifTrace.integrate`.

        Returns:
            Tuple ``(laser_axis, integrals)``. Length of both is
            ``laser_points``.
        """

        lasers, _ = self.laser_axis()
        out = np.full(self.laser_points, fill, dtype=np.float64)
        for l in range(self.laser_points):
            if (l, int(d_index)) in self._index:
                out[l] = self.get_trace(l, d_index).integrate(**proc)
        return lasers, out

    def image(
        self,
        fill=np.nan,
        **proc,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate every present trace into a 2D delay × laser image.

        Args:
            fill: Value substituted at ``(d, l)`` positions that have
                no acquired trace.
            **proc: Optional processing overrides, forwarded to
                :meth:`BCLifTrace.integrate`.

        Returns:
            Tuple ``(delay_axis, laser_axis, integrals)`` where
            ``integrals`` has shape ``(delay_points, laser_points)``.
        """

        delays, _ = self.delay_axis()
        lasers, _ = self.laser_axis()
        out = np.full((self.delay_points, self.laser_points), fill, dtype=np.float64)
        for l, d in self._index:
            out[d, l] = self.get_trace(l, d).integrate(**proc)
        return delays, lasers, out
