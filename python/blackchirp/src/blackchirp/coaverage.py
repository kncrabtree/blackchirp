"""Multi-experiment FID coaveraging.

Two entry points are provided:

* :func:`coaverage_fids` — time-domain coaverage that returns a
  :class:`~blackchirp.BCFid`. Sums raw integer data shot-for-shot and
  rescales to voltage with the (shared) ``vmult`` and the total shot
  count. Supports optional phase correction via cross-correlation
  against a chosen reference window.
* :func:`coaverage_spectra` — shot-weighted magnitude-spectrum
  coaverage that returns ``(x, y)`` arrays. Each input FID is
  Fourier-transformed with the same processing kwargs and the
  resulting magnitude spectra are averaged with weights equal to the
  shot count of each FID.

Both entry points enforce strict compatibility between the input
FIDs: matching ``spacing``, ``size``, ``sideband``, ``probefreq``,
``vmult``, and frame count. Float-valued fields are compared with
exact equality, on the assumption that fidparams CSVs written by the
same software produce identical numeric strings; mismatches raise
``ValueError`` rather than being silently coerced.
"""

from __future__ import annotations

import copy
from typing import Sequence, Union

import numpy as np
import scipy.signal as spsig

from .bcfid import BCFid, _SIDEBAND_INT_MAP, _SIDEBAND_MAP
from ._enum_helpers import _resolve_enum


def _resolved_sideband(fid: BCFid) -> str:
    return _resolve_enum(
        fid.fidparams["sideband"], _SIDEBAND_MAP, int_map=_SIDEBAND_INT_MAP
    )


def _check_compatibility(fids: Sequence[BCFid]) -> None:
    """Raise ``ValueError`` unless every FID matches the first on the
    fields that govern coaverage compatibility."""
    if len(fids) == 0:
        raise ValueError("fids must be a non-empty sequence of BCFid objects")
    for i, f in enumerate(fids):
        if not isinstance(f, BCFid):
            raise TypeError(f"fids[{i}] is not a BCFid (got {type(f).__name__})")

    ref = fids[0]
    ref_size = int(ref.fidparams["size"])
    ref_spacing = float(ref.fidparams.spacing)
    ref_probe = float(ref.fidparams.probefreq)
    ref_vmult = float(ref.fidparams.vmult)
    ref_sideband = _resolved_sideband(ref)
    ref_frames = int(ref.frames)

    for i, f in enumerate(fids[1:], start=1):
        size = int(f.fidparams["size"])
        if size != ref_size:
            raise ValueError(f"FID size mismatch: fids[0]={ref_size}, fids[{i}]={size}")
        if int(f.frames) != ref_frames:
            raise ValueError(
                f"Frame-count mismatch: fids[0]={ref_frames}, fids[{i}]={int(f.frames)}"
            )
        if float(f.fidparams.spacing) != ref_spacing:
            raise ValueError(
                f"spacing mismatch: fids[0]={ref_spacing}, "
                f"fids[{i}]={float(f.fidparams.spacing)}"
            )
        if float(f.fidparams.probefreq) != ref_probe:
            raise ValueError(
                f"probefreq mismatch: fids[0]={ref_probe}, "
                f"fids[{i}]={float(f.fidparams.probefreq)}"
            )
        if float(f.fidparams.vmult) != ref_vmult:
            raise ValueError(
                f"vmult mismatch: fids[0]={ref_vmult}, "
                f"fids[{i}]={float(f.fidparams.vmult)}; refusing to combine "
                "FIDs with different voltage scaling"
            )
        sb = _resolved_sideband(f)
        if sb != ref_sideband:
            raise ValueError(
                f"sideband mismatch: fids[0]={ref_sideband!r}, fids[{i}]={sb!r}"
            )


def _resolve_reference(fids: Sequence[BCFid], reference: Union[int, str]) -> int:
    if isinstance(reference, str):
        if reference == "max_shots":
            shots = [int(f.fidparams.shots) for f in fids]
            return int(np.argmax(shots))
        raise ValueError(
            f"reference={reference!r}; only 'max_shots' is accepted as a "
            "string value"
        )
    if isinstance(reference, (int, np.integer)) and not isinstance(reference, bool):
        idx = int(reference)
        if idx < 0 or idx >= len(fids):
            raise ValueError(f"reference index {idx} out of range [0, {len(fids) - 1}]")
        return idx
    raise TypeError(
        f"reference must be an int or 'max_shots' (got {type(reference).__name__})"
    )


def _resolve_pc_window(
    fid: BCFid, pc_start_us: float, pc_end_us: float
) -> tuple[int, int]:
    spacing = float(fid.fidparams.spacing)
    size = int(fid.fidparams["size"])
    start = int(round(pc_start_us / 1e6 / spacing))
    end = int(round(pc_end_us / 1e6 / spacing))
    if start < 0 or end > size or start >= end:
        raise ValueError(
            f"phase-correction window [{pc_start_us}us, {pc_end_us}us] resolves "
            f"to sample indices [{start}, {end}] which is invalid for size={size}"
        )
    return start, end


def _xcorr_shift(reference: np.ndarray, target: np.ndarray) -> int:
    """Return the integer shift (samples) that aligns ``target`` to
    ``reference`` by maximising their cross-correlation.

    Both inputs are de-meaned before correlation so that a DC offset
    (which the raw FID integers always carry) does not swamp the
    signal-alignment peak with the rectangular overlap envelope."""
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    ref = ref - ref.mean()
    tgt = tgt - tgt.mean()
    corr = spsig.correlate(ref, tgt, mode="full", method="auto")
    return int(np.argmax(corr) - (len(ref) - 1))


def _add_with_shift(dest: np.ndarray, src: np.ndarray, shift: int) -> None:
    """In-place add ``src`` to ``dest`` after shifting ``src`` by
    ``shift`` samples along axis 0. Both arrays must have the same
    shape; samples that fall off either end of ``dest`` are dropped."""
    if shift == 0:
        dest += src
    elif shift < 0:
        dest[:shift] += src[-shift:]
    else:
        dest[shift:] += src[:-shift]


def coaverage_fids(
    fids: Sequence[BCFid],
    *,
    pc_start_us: float = None,
    pc_end_us: float = None,
    reference: Union[int, str] = "max_shots",
    per_frame_pc: bool = False,
) -> BCFid:
    """Coaverage of multiple FIDs in the time domain.

    The result is a :class:`BCFid` whose raw integer data is the
    sample-by-sample sum of every input's raw data and whose
    ``fidparams.shots`` is the sum of every input's shot count.
    Voltage data is recomputed as ``rawdata * vmult / total_shots``.

    All input FIDs must agree on ``spacing``, ``size``, ``sideband``,
    ``probefreq``, ``vmult``, and frame count; mismatches raise
    ``ValueError``. Coaverage of FIDs taken with different digitizer
    settings is not supported — preprocess inputs to a common
    representation before calling.

    Phase correction is optional: when both ``pc_start_us`` and
    ``pc_end_us`` are supplied, each non-reference FID is shifted
    along the time axis by the integer offset that maximises the
    cross-correlation between its windowed data and the windowed data
    of the reference FID. A single shift is applied to all frames of a
    given FID, which assumes the frames share a clock; pass
    ``per_frame_pc=True`` to compute and apply one shift per frame
    instead.

    Args:
        fids: Sequence of FIDs to coaverage. Already-loaded
            :class:`BCFid` objects only — paths are not accepted, so
            callers can choose backups, differential FIDs, or
            frame-averaged FIDs as inputs.
        pc_start_us: Start of the phase-correction window, in μs.
            If either bound is ``None``, phase correction is disabled
            and the FIDs are summed without alignment. The window
            should cover a signal-rich portion of the FID (e.g. the
            chirp ring-down) — cross-correlation is computed on
            de-meaned data, so a window that is mostly noise will
            return an unreliable shift.
        pc_end_us: End of the phase-correction window, in μs.
        reference: Index of the FID to use as the phase-correction
            reference, or the string ``"max_shots"`` (default) to
            pick the FID with the highest shot count. Ties resolve to
            the first match.
        per_frame_pc: If ``True``, compute a separate shift for each
            (FID, frame) pair instead of one shift per FID.

    Returns:
        A new :class:`BCFid` containing the coaveraged data. Inputs
        are not mutated.

    Raises:
        ValueError: If ``fids`` is empty, if any compatibility check
            fails, if exactly one of ``pc_start_us``/``pc_end_us`` is
            provided, if the resolved phase-correction window is
            outside ``[0, size)``, or if ``reference`` is an invalid
            index or string.
        TypeError: If any element of ``fids`` is not a ``BCFid`` or
            if ``reference`` is neither ``int`` nor ``str``.
    """
    _check_compatibility(fids)

    pc_enabled = pc_start_us is not None and pc_end_us is not None
    if (pc_start_us is None) ^ (pc_end_us is None):
        raise ValueError(
            "pc_start_us and pc_end_us must be provided together (both or neither)"
        )

    ref_idx = _resolve_reference(fids, reference)
    ref_fid = fids[ref_idx]

    pc_start = pc_end = 0
    if pc_enabled:
        pc_start, pc_end = _resolve_pc_window(ref_fid, pc_start_us, pc_end_us)

    out_raw = ref_fid._rawdata.astype(np.int64, copy=True)
    total_shots = int(ref_fid.fidparams.shots)

    for i, f in enumerate(fids):
        if i == ref_idx:
            continue
        src = f._rawdata
        if pc_enabled:
            if per_frame_pc:
                for frame in range(out_raw.shape[1]):
                    ref_seg = ref_fid._rawdata[pc_start:pc_end, frame]
                    tgt_seg = src[pc_start:pc_end, frame]
                    shift = _xcorr_shift(ref_seg, tgt_seg)
                    _add_with_shift(out_raw[:, frame], src[:, frame], shift)
            else:
                ref_seg = ref_fid._rawdata[pc_start:pc_end, 0]
                tgt_seg = src[pc_start:pc_end, 0]
                shift = _xcorr_shift(ref_seg, tgt_seg)
                _add_with_shift(out_raw, src, shift)
        else:
            out_raw += src
        total_shots += int(f.fidparams.shots)

    out = copy.deepcopy(ref_fid)
    out._rawdata = out_raw
    out.fidparams.loc["shots"] = total_shots
    out.shots = total_shots
    out.data = out._rawdata * out.fidparams.vmult / total_shots
    return out


def coaverage_spectra(
    fids: Sequence[BCFid],
    **ft_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Shot-weighted coaverage of magnitude spectra.

    Each input FID is Fourier-transformed via :meth:`BCFid.ft` with
    the supplied processing kwargs, and the magnitude spectra are
    combined as

    .. math::
        y = \\frac{\\sum_i s_i\\, |Y_i|}{\\sum_i s_i}

    where :math:`s_i` is the shot count of FID :math:`i`. The same
    compatibility checks as :func:`coaverage_fids` apply, ensuring
    every spectrum lands on the same frequency grid.

    Note that magnitude-spectrum coaveraging does not reduce the
    noise floor — the Rayleigh-distributed noise mean is invariant
    under averaging — only its fluctuation. Use
    :func:`coaverage_fids` when phase coherence between experiments
    is good enough to align in the time domain; reach for this
    function when phase drift defeats time-domain alignment.

    Args:
        fids: Sequence of FIDs to coaverage.
        **ft_kwargs: Forwarded to :meth:`BCFid.ft`. The same kwargs
            are applied to every FID so that all spectra share a
            frequency grid and processing.

    Returns:
        Frequency array (in the units selected by the ``freq_units``
        kwarg, default MHz) and the shot-weighted magnitude-spectrum
        array. The intensity array preserves the per-frame second
        axis from :meth:`BCFid.ft`.

    Raises:
        ValueError: If ``fids`` is empty, if any compatibility check
            fails, or if any FID's :meth:`ft` returns a frequency
            array that disagrees with the reference (which should not
            happen given the compatibility checks but is asserted
            defensively).
    """
    _check_compatibility(fids)

    total_shots = sum(int(f.fidparams.shots) for f in fids)
    if total_shots <= 0:
        raise ValueError("total shot count across input FIDs is zero")

    # BCFid.ft() may mutate self.data in place via the windowed view
    # used for the FidRemoveDC subtraction. Operate on copies so that
    # repeated FT calls during coaverage do not corrupt the caller's
    # FIDs or each other.
    x_ref, y_acc = copy.deepcopy(fids[0]).ft(**ft_kwargs)
    y_acc = y_acc * float(fids[0].fidparams.shots)
    for f in fids[1:]:
        x, y = copy.deepcopy(f).ft(**ft_kwargs)
        if x.shape != x_ref.shape or not np.allclose(x, x_ref):
            raise ValueError(
                "FT frequency grids disagree across inputs; cannot coaverage"
            )
        y_acc = y_acc + y * float(f.fidparams.shots)

    return x_ref, y_acc / total_shots
