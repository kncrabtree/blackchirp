"""Tests for ``coaverage_fids`` and ``coaverage_spectra``."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from blackchirp import BCFid, coaverage_fids, coaverage_spectra


def _clone(fid: BCFid) -> BCFid:
    return copy.deepcopy(fid)


def _set_shots(fid: BCFid, shots: int) -> None:
    """Update both the ``fidparams.shots`` cell and the cached
    ``BCFid.shots`` attribute, then refresh the voltage data array."""
    fid.fidparams.loc["shots"] = shots
    fid.shots = shots
    fid.data = fid._rawdata * fid.fidparams.vmult / shots


def test_two_identical_fids_double_shots(any_exp):
    """Coaverage of two copies of one FID produces 2x raw data and shots."""
    f = any_exp.ftmw.get_fid(0)
    out = coaverage_fids([f, _clone(f)])
    assert isinstance(out, BCFid)
    np.testing.assert_array_equal(out._rawdata, 2 * f._rawdata)
    assert int(out.fidparams.shots) == 2 * int(f.fidparams.shots)
    assert out.shots == 2 * int(f.fidparams.shots)
    expected_data = (2 * f._rawdata) * f.fidparams.vmult / (2 * int(f.fidparams.shots))
    np.testing.assert_allclose(out.data, expected_data)


def test_inputs_not_mutated(any_exp):
    """The input FIDs must not be modified by coaverage_fids."""
    f = any_exp.ftmw.get_fid(0)
    raw_before = f._rawdata.copy()
    shots_before = int(f.fidparams.shots)
    coaverage_fids([f, _clone(f), _clone(f)])
    np.testing.assert_array_equal(f._rawdata, raw_before)
    assert int(f.fidparams.shots) == shots_before


def test_single_input(any_exp):
    """A single-element list returns a deep copy with the same content."""
    f = any_exp.ftmw.get_fid(0)
    out = coaverage_fids([f])
    np.testing.assert_array_equal(out._rawdata, f._rawdata)
    assert out is not f
    assert int(out.fidparams.shots) == int(f.fidparams.shots)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        coaverage_fids([])


def test_non_bcfid_input_raises(any_exp):
    f = any_exp.ftmw.get_fid(0)
    with pytest.raises(TypeError):
        coaverage_fids([f, "not a fid"])


def test_size_mismatch_refused(any_exp):
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    g.fidparams.loc["size"] = int(g.fidparams["size"]) + 1
    with pytest.raises(ValueError, match="size mismatch"):
        coaverage_fids([f, g])


def test_spacing_mismatch_refused(any_exp):
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    g.fidparams.loc["spacing"] = float(g.fidparams.spacing) * 2
    with pytest.raises(ValueError, match="spacing mismatch"):
        coaverage_fids([f, g])


def test_probefreq_mismatch_refused(any_exp):
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    g.fidparams.loc["probefreq"] = float(g.fidparams.probefreq) + 1.0
    with pytest.raises(ValueError, match="probefreq mismatch"):
        coaverage_fids([f, g])


def test_vmult_mismatch_refused(any_exp):
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    g.fidparams.loc["vmult"] = float(g.fidparams.vmult) * 1.5
    with pytest.raises(ValueError, match="vmult mismatch"):
        coaverage_fids([f, g])


def test_sideband_mismatch_refused(any_exp):
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    sb = f.fidparams["sideband"]
    g.fidparams.loc["sideband"] = (
        "LowerSideband" if sb != "LowerSideband" else "UpperSideband"
    )
    with pytest.raises(ValueError, match="sideband mismatch"):
        coaverage_fids([f, g])


def test_frame_count_mismatch_refused(any_exp):
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    g._rawdata = np.concatenate([g._rawdata, g._rawdata[:, :1]], axis=1)
    g.frames = g._rawdata.shape[1]
    with pytest.raises(ValueError, match="Frame-count mismatch"):
        coaverage_fids([f, g])


def test_reference_max_shots_default(any_exp):
    """With no PC, summing is order-independent so 'max_shots' default
    should yield the same raw data as an explicit reference index."""
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    _set_shots(g, int(f.fidparams.shots) // 2 + 1)
    out_default = coaverage_fids([f, g])
    out_explicit = coaverage_fids([f, g], reference=0)
    np.testing.assert_array_equal(out_default._rawdata, out_explicit._rawdata)


def test_reference_max_shots_picks_largest(any_exp):
    """'max_shots' picks the largest-shot FID — verify by giving the
    second input a synthetic high shot count and an unusual rawdata
    pattern that survives as the seed of the sum."""
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    g._rawdata = g._rawdata + 7
    _set_shots(g, int(f.fidparams.shots) * 10)
    out = coaverage_fids([f, g])
    expected = f._rawdata + g._rawdata
    np.testing.assert_array_equal(out._rawdata, expected)


def test_reference_invalid_index(any_exp):
    f = any_exp.ftmw.get_fid(0)
    with pytest.raises(ValueError):
        coaverage_fids([f, _clone(f)], reference=5)


def test_reference_invalid_string_value(any_exp):
    f = any_exp.ftmw.get_fid(0)
    with pytest.raises(ValueError):
        coaverage_fids([f, _clone(f)], reference="lowest")


def test_reference_wrong_type(any_exp):
    f = any_exp.ftmw.get_fid(0)
    with pytest.raises(TypeError):
        coaverage_fids([f, _clone(f)], reference=1.5)


def _pc_window_us(fid: BCFid, n_samples: int = 4096) -> tuple[float, float]:
    """Return a (start_us, end_us) window covering the first
    ``n_samples`` of ``fid`` (or the whole FID if shorter).
    Keeping the window small keeps the cross-correlation fast on
    real fixture data."""
    size = int(fid.fidparams["size"])
    spacing = float(fid.fidparams.spacing)
    end = min(n_samples, size)
    return 0.0, end * spacing * 1e6


def test_pc_zero_shift_matches_no_pc(any_exp):
    """With identical inputs the cross-correlation gives shift=0, so the
    PC and no-PC paths must produce the same result."""
    f = any_exp.ftmw.get_fid(0)
    start_us, end_us = _pc_window_us(f)
    out_pc = coaverage_fids(
        [f, _clone(f)], pc_start_us=start_us, pc_end_us=end_us, reference=0
    )
    out_no = coaverage_fids([f, _clone(f)], reference=0)
    np.testing.assert_array_equal(out_pc._rawdata, out_no._rawdata)


def test_pc_recovers_known_shift(any_exp):
    """Manually shift one input by a known number of samples and verify
    that PC realigns it. The summed raw data should match 2x the
    reference, modulo the samples that fall off the end of the window."""
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    shift = 3
    g._rawdata = np.roll(g._rawdata, shift, axis=0)
    g._rawdata[:shift, :] = 0  # remove circular wrap

    start_us, end_us = _pc_window_us(f)
    out = coaverage_fids([f, g], pc_start_us=start_us, pc_end_us=end_us, reference=0)
    # After PC, the realigned target is shifted back by -shift; samples
    # at the tail get dropped (added nothing). Compare on the safe interior.
    safe = out._rawdata[:-shift, :]
    expected = 2 * f._rawdata[:-shift, :]
    np.testing.assert_array_equal(safe, expected)


def test_pc_window_one_sided_raises(any_exp):
    f = any_exp.ftmw.get_fid(0)
    with pytest.raises(ValueError, match="together"):
        coaverage_fids([f, _clone(f)], pc_start_us=0.0)


def test_pc_window_out_of_range(any_exp):
    f = any_exp.ftmw.get_fid(0)
    with pytest.raises(ValueError, match="phase-correction window"):
        coaverage_fids([f, _clone(f)], pc_start_us=-1.0, pc_end_us=1.0)


def test_per_frame_pc_runs(any_exp):
    """Smoke test: per_frame_pc=True path executes and produces a sane
    BCFid. Most fixtures are single-frame, so this exercises the loop
    even though the per-frame and per-fid shifts coincide."""
    f = any_exp.ftmw.get_fid(0)
    start_us, end_us = _pc_window_us(f)
    out = coaverage_fids(
        [f, _clone(f)],
        pc_start_us=start_us,
        pc_end_us=end_us,
        reference=0,
        per_frame_pc=True,
    )
    np.testing.assert_array_equal(out._rawdata, 2 * f._rawdata)


def test_coaverage_spectra_two_identical(any_exp):
    """Coaveraging two copies of a FID's spectrum yields the original."""
    f = any_exp.ftmw.get_fid(0)
    x_ref, y_ref = f.ft()
    x, y = coaverage_spectra([f, _clone(f)])
    np.testing.assert_array_equal(x, x_ref)
    np.testing.assert_allclose(y, y_ref)


def test_coaverage_spectra_shot_weighting(any_exp):
    """Two FIDs with synthetic spectra differing in shot count weight
    correctly: y = (s0*y0 + s1*y1) / (s0 + s1)."""
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    _set_shots(g, int(f.fidparams.shots) * 3)
    x0, y0 = f.ft()
    _, y1 = g.ft()
    s0 = int(f.fidparams.shots)
    s1 = int(g.fidparams.shots)
    expected = (s0 * y0 + s1 * y1) / (s0 + s1)
    x, y = coaverage_spectra([f, g])
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_allclose(y, expected)


def test_coaverage_spectra_passes_ft_kwargs(any_exp):
    """ft_kwargs should reach BCFid.ft. Verify by changing freq_units."""
    f = any_exp.ftmw.get_fid(0)
    x_mhz, _ = coaverage_spectra([f, _clone(f)])
    x_ghz, _ = coaverage_spectra([f, _clone(f)], freq_units="GHz")
    np.testing.assert_allclose(x_ghz, x_mhz * 1e-3)


def test_coaverage_spectra_compatibility_enforced(any_exp):
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    g.fidparams.loc["probefreq"] = float(g.fidparams.probefreq) + 1.0
    with pytest.raises(ValueError, match="probefreq mismatch"):
        coaverage_spectra([f, g])


def test_coaverage_spectra_zero_total_shots_raises(any_exp):
    f = any_exp.ftmw.get_fid(0)
    g = _clone(f)
    # _set_shots with shots=0 produces NaN data via division; suppress
    # the resulting divide-by-zero warning since we never reach the
    # path that would touch the data.
    with np.errstate(divide="ignore", invalid="ignore"):
        _set_shots(f, 0)
        _set_shots(g, 0)
    with pytest.raises(ValueError, match="zero"):
        coaverage_spectra([f, g])
