import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from arpes.fits.fit_models import AffineBroadenedFD, QuadraticModel
from arpes.fits.utilities import broadcast_model
from arpes.io import example_data
from arpes.utilities.conversion import convert_to_kspace
from arpes.utilities.conversion.forward import convert_through_angular_point


def load_energy_corrected():
    fmap = example_data.map.spectrum
    return fmap

    results = broadcast_model(AffineBroadenedFD, cut, "phi", parallelize=False)
    edge = QuadraticModel().guess_fit(results.F.p("fd_center")).eval(x=fmap.phi)
    return fmap.G.shift_by(edge, "eV")


def test_cut_momentum_conversion():
    """Validates that the core APIs are functioning."""
    kdata = convert_to_kspace(example_data.cut.spectrum, kp=np.linspace(-0.12, 0.12, 600))
    selected = kdata.values.ravel()[[0, 200, 800, 1500, 2800, 20000, 40000, 72000]]

    assert_array_almost_equal(
        np.nan_to_num(selected),
        np.array(
            [
                0,
                319.73139835,
                318.12917486,
                258.94653353,
                200.48829069,
                163.12937875,
                346.93136055,
                0,
            ]
        ),
        decimal=1,
    )


def test_cut_momentum_conversion_ranges():
    """Validates that the user can select momentum ranges."""

    data = example_data.cut.spectrum
    kdata = convert_to_kspace(data, kp=np.linspace(-0.12, 0.12, 80))

    expected_values = np.array(
        [
            192,
            157,
            157,
            183,
            173,
            173,
            177,
            165,
            171,
            159,
            160,
            154,
            155,
            153,
            146,
            139,
            139,
            138,
            127,
            125,
            121,
            117,
            118,
            113,
            125,
            145,
            147,
            141,
            147,
            147,
            146,
            143,
            143,
            145,
            131,
            147,
            136,
            133,
            145,
            139,
            136,
            138,
            128,
            133,
            126,
            136,
            135,
            139,
            141,
            147,
            143,
            144,
            155,
            151,
            159,
            140,
            150,
            120,
            121,
            125,
            127,
            130,
            138,
            140,
            149,
            144,
            155,
            151,
            154,
            165,
            165,
            166,
            172,
            168,
            167,
            177,
            177,
            171,
            168,
            160,
        ]
    )
    assert_array_almost_equal(kdata.argmax(dim="eV").values, expected_values)


def test_fermi_surface_conversion():
    """Validates that the kx-ky conversion code is behaving."""
    data = load_energy_corrected().S.fermi_surface

    kdata = convert_to_kspace(
        data,
        kx=np.linspace(-2.5, 1.5, 400),
        ky=np.linspace(-2, 2, 400),
    )

    kx_max = kdata.idxmax(dim="ky").max().item()
    ky_max = kdata.idxmax(dim="kx").max().item()

    assert ky_max == pytest.approx(0.4373433583959896)
    assert kx_max == pytest.approx(-0.02506265664160412)
    assert kdata.mean().item() == pytest.approx(613.848688084093)
    assert kdata.fillna(0).mean().item() == pytest.approx(415.7673895479573)


@pytest.mark.skip
def test_conversion_with_passthrough_axis():
    """Validates that passthrough is equivalent to individual slice conversion."""
    raise NotImplementedError


@pytest.mark.skip
def test_kz_conversion():
    """Validates the kz conversion code."""
    raise NotImplementedError


@pytest.mark.skip
def test_inner_potential():
    """Validates that the inner potential changes kz offset and kp range."""
    raise NotImplementedError


@pytest.mark.skip
def test_convert_angular_pair():
    """Validates that we correctly convert through high symmetry points and angle."""
    raise NotImplementedError


def test_convert_angular_point_and_angle():
    """Validates that we correctly convert through high symmetry points."""

    test_point = {
        "phi": -0.13,
        "theta": -0.1,
        "eV": 0.0,
    }
    data = load_energy_corrected()

    kdata = convert_through_angular_point(
        data,
        test_point,
        {"ky": np.linspace(-1, 1, 400)},
        {"kx": np.linspace(-0.02, 0.02, 10)},
    )

    max_values = np.array(
        [
            4141.82736603,
            4352.10441395,
            4528.14158708,
            4772.79036439,
            4967.80545468,
            5143.31935106,
            5389.48929973,
            5564.49516953,
            5963.14662042,
            6495.7520699,
            6865.54515501,
            7112.05589829,
            7796.47414459,
            8160.19371472,
            8525.13697199,
            8520.55263924,
            8266.86194778,
            7786.59602604,
            7151.34116069,
            6764.77021486,
            6381.08052888,
            6075.55168331,
            5922.88003222,
            5625.56194439,
            3077.88595448,
            117.28906073,
        ]
    )

    assert_array_almost_equal(
        kdata.sel(ky=slice(-0.7, 0)).isel(eV=slice(None, -20, 5)).max("ky").values,
        max_values,
    )
