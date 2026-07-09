"""Tests for LIF frequency-conversion topology reading and DAG accessors.

The ``v2-lif-ref`` fixture carries a synthetic doubler ``liftopology.csv``
(one ``NHG`` stage, ``n=2``: output = 2 x fundamental in cm^-1). The
``v2-lif-noref`` fixture has no topology file and exercises the identity
(no-conversion) path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# --- Present, non-identity topology (v2-lif-ref: NHG doubler) ---------------


def test_liftopology_dataframe_loaded(v2_lif_ref_exp):
    topo = v2_lif_ref_exp.liftopology
    assert isinstance(topo, pd.DataFrame)
    assert list(topo.columns) == [
        "Index",
        "StageKey",
        "Op",
        "Harmonic",
        "IsFinal",
        "Input0",
        "Input1",
        "OutCoeffA",
        "OutCoeffB",
    ]
    assert len(topo) == 1
    assert topo.iloc[0]["Op"] == "NHG"


def test_topology_model_attributes(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    assert lif.has_topology is True
    assert lif.stages == ["LifFreqConversionStage.doubler"]
    assert lif.final_stage == "LifFreqConversionStage.doubler"
    assert lif.laser_key == "LifLaser.virtual"


def test_doubler_output_from_fundamental_cm1(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    # NHG n=2: output_cm1 = 2 * fundamental_cm1.
    assert lif.at_stage(
        15000.0, at="final", side="output", unit="cm-1"
    ) == pytest.approx(30000.0)
    assert lif.fundamental(
        30000.0, at="final", side="output", unit="cm-1"
    ) == pytest.approx(15000.0)


def test_fundamental_at_stage_round_trip_array(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    outputs = np.array([28000.0, 29000.0, 30000.0])
    fund = lif.fundamental(outputs, at="final", side="output", unit="cm-1")
    assert isinstance(fund, np.ndarray)
    assert fund.shape == outputs.shape
    np.testing.assert_allclose(fund, outputs / 2.0)
    np.testing.assert_allclose(
        lif.at_stage(fund, at="final", side="output", unit="cm-1"), outputs
    )


def test_scalar_returns_scalar(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    val = lif.at_stage(15000.0, unit="cm-1")
    assert isinstance(val, float)


def test_wavelength_doubling(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    # Doubling in cm^-1 halves the wavelength: a 280 nm output beam comes
    # from a 560 nm fundamental.
    assert lif.fundamental(
        280.0, at="final", side="output", unit="nm"
    ) == pytest.approx(560.0)
    assert lif.at_stage(560.0, at="final", side="output", unit="nm") == pytest.approx(
        280.0
    )


def test_non_ascii_cm1_label_accepted(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    # The header.csv unit label for wavenumber is the non-ASCII "cm⁻¹".
    assert lif.at_stage(15000.0, unit="cm⁻¹") == pytest.approx(
        lif.at_stage(15000.0, unit="cm-1")
    )


def test_stage_frequencies_table(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    df = lif.stage_frequencies(fundamental=15000.0, unit="cm-1")
    assert list(df.index) == ["laser", "LifFreqConversionStage.doubler"]
    assert list(df.columns) == ["op", "isfinal", "input0", "input1", "output"]
    # Laser row is the fundamental itself.
    assert df.loc["laser", "output"] == pytest.approx(15000.0)
    # Doubler: input is the fundamental, output is twice it, no second input.
    row = df.loc["LifFreqConversionStage.doubler"]
    assert row["op"] == "NHG"
    assert bool(row["isfinal"]) is True
    assert row["input0"] == pytest.approx(15000.0)
    assert np.isnan(row["input1"])
    assert row["output"] == pytest.approx(30000.0)


def test_stage_frequencies_from_value(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    df = lif.stage_frequencies(value=30000.0, at="final", side="output", unit="cm-1")
    assert df.loc["laser", "output"] == pytest.approx(15000.0)
    assert df.loc["LifFreqConversionStage.doubler", "output"] == pytest.approx(30000.0)


def test_stage_frequencies_rejects_array(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    with pytest.raises(ValueError):
        lif.stage_frequencies(fundamental=np.array([1.0, 2.0]))


def test_stage_frequencies_requires_exactly_one_source(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    with pytest.raises(ValueError):
        lif.stage_frequencies()
    with pytest.raises(ValueError):
        lif.stage_frequencies(value=1.0, fundamental=1.0)


def test_laser_axis_to_fundamental(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    axis, unit = lif.laser_axis()  # output beam axis, in nm
    fund = lif.fundamental(axis, at="final", side="output", unit=unit)
    # Fundamental wavelength is twice the output wavelength.
    np.testing.assert_allclose(fund, 2.0 * axis)


# --- Absent topology (v2-lif-noref: identity) -------------------------------


def test_identity_no_topology_file(v2_lif_noref_exp):
    assert v2_lif_noref_exp.liftopology is None
    lif = v2_lif_noref_exp.lif
    assert lif.has_topology is False
    assert lif.stages == []
    assert lif.final_stage is None


def test_identity_passthrough(v2_lif_noref_exp):
    lif = v2_lif_noref_exp.lif
    # With no conversion, output == fundamental.
    assert lif.at_stage(
        12345.0, at="final", side="output", unit="cm-1"
    ) == pytest.approx(12345.0)
    assert lif.fundamental(
        12345.0, at="final", side="output", unit="cm-1"
    ) == pytest.approx(12345.0)


def test_identity_stage_frequencies_laser_only(v2_lif_noref_exp):
    lif = v2_lif_noref_exp.lif
    df = lif.stage_frequencies(fundamental=12345.0, unit="cm-1")
    assert list(df.index) == ["laser"]
    assert df.loc["laser", "output"] == pytest.approx(12345.0)


def test_unknown_stage_raises(v2_lif_noref_exp):
    lif = v2_lif_noref_exp.lif
    with pytest.raises(KeyError):
        lif.at_stage(1.0, at="nonexistent.stage", side="output")


def test_unknown_unit_raises(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    with pytest.raises(ValueError):
        lif.at_stage(1.0, unit="furlongs")
