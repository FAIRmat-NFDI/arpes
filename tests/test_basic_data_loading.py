import numpy as np
import pytest
import xarray as xr

from arpes.utilities.conversion import convert_to_kspace


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []

    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append(([x[1] for x in items]))

    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


class TestMetadata(object):
    """
    Tests metadata normalization conventions.
    """

    data = None

    scenarios = [
        # Lanzara Group "Main Chamber"
        (
            "main_chamber_load_cut",
            {
                "file": "basic/main_chamber_cut_0.fits",
                "expected": {
                    "scan_info": {
                        "time": "1:45:34 pm",
                        "date": "2/3/2016",
                        "type": None,
                        "spectrum_type": "cut",
                        "experimenter": None,
                        "sample": None,
                    },
                    "experiment_info": {
                        "temperature": None,
                        "temperature_cryotip": None,
                        "pressure": None,
                        "polarization": (None, None),
                        "photon_flux": None,
                        "photocurrent": None,
                        "probe": None,
                        "probe_detail": None,
                        "analyzer": "Specs PHOIBOS 150",
                        "analyzer_detail": {
                            "type": "hemispherical",
                            "radius": 150,
                            "name": "Specs PHOIBOS 150",
                            "parallel_deflectors": False,
                            "perpendicular_deflectors": False,
                        },
                    },
                    "analyzer_info": {
                        "lens_mode": None,
                        "lens_mode_name": "WideAngleMode:40V",
                        "acquisition_mode": None,
                        "pass_energy": None,
                        "slit_shape": None,
                        "slit_width": None,
                        "slit_number": None,
                        "lens_table": None,
                        "analyzer_type": "hemispherical",
                        "mcp_voltage": None,
                    },
                    "daq_info": {
                        "daq_type": None,
                        "region": None,
                        "region_name": None,
                        "prebinning": {"eV": 2, "phi": 1},
                        "trapezoidal_correction_strategy": None,
                        "dither_settings": None,
                        "sweep_settings": {
                            "low_energy": None,
                            "high_energy": None,
                            "n_sweeps": None,
                            "step": None,
                        },
                        "frames_per_slice": 500,
                        "frame_duration": None,
                        "center_energy": None,
                    },
                    "laser_info": {
                        "pump_wavelength": None,
                        "pump_energy": None,
                        "pump_fluence": None,
                        "pump_pulse_energy": None,
                        "pump_spot_size": (None, None),
                        "pump_profile": None,
                        "pump_linewidth": None,
                        "pump_temporal_width": None,
                        "pump_polarization": (None, None),
                        "probe_wavelength": None,
                        "probe_energy": 5.93,
                        "probe_fluence": None,
                        "probe_pulse_energy": None,
                        "probe_spot_size": (None, None),
                        "probe_profile": None,
                        "probe_linewidth": 0.015,
                        "probe_temporal_width": None,
                        "probe_polarization": (None, None),
                        "repetition_rate": None,
                    },
                    "sample_info": {
                        "id": None,
                        "name": None,
                        "source": None,
                        "reflectivity": None,
                    },
                },
            },
        ),
        (
            "maestro_load_cut",
            {
                "file": "basic/MAESTRO_12.fits",
                "expected": {
                    "scan_info": {
                        "time": "7:08:42 pm",
                        "date": "10/11/2018",
                        "type": "XY Scan",
                        "spectrum_type": "spem",
                        "experimenter": None,
                        "sample": None,
                    },
                    "experiment_info": {
                        "temperature": None,
                        "temperature_cryotip": None,
                        "pressure": None,
                        "polarization": (None, None),
                        "photon_flux": None,
                        "photocurrent": None,
                        "probe": None,
                        "probe_detail": None,
                        "analyzer": "R4000",
                        "analyzer_detail": {
                            "type": "hemispherical",
                            "radius": None,
                            "name": "Scienta R4000",
                            "parallel_deflectors": False,
                            "perpendicular_deflectors": True,
                        },
                    },
                    "analyzer_info": {
                        "lens_mode": None,
                        "lens_mode_name": "Angular30",
                        "acquisition_mode": None,
                        "pass_energy": 50,
                        "slit_shape": "curved",
                        "slit_width": 0.5,
                        "slit_number": 7,
                        "lens_table": None,
                        "analyzer_type": "hemispherical",
                        "mcp_voltage": None,
                    },
                    "beamline_info": {
                        "hv": pytest.approx(125, 1e-2),
                        "linewidth": None,
                        "beam_current": pytest.approx(500.44, 1e-2),
                        "photon_polarization": (None, None),
                        "repetition_rate": 5e8,
                        "entrance_slit": None,
                        "exit_slit": None,
                        "undulator_info": {
                            "harmonic": 1,
                            "type": "elliptically_polarized_undulator",
                            "gap": None,
                            "z": None,
                            "polarization": None,
                        },
                        "monochromator_info": {
                            "grating_lines_per_mm": 600,
                        },
                    },
                    "daq_info": {
                        "daq_type": "XY Scan",
                        "region": None,
                        "region_name": None,
                        "prebinning": {
                            "eV": 2,
                        },
                        "trapezoidal_correction_strategy": None,
                        "dither_settings": None,
                        "sweep_settings": {
                            "low_energy": None,
                            "high_energy": None,
                            "n_sweeps": None,
                            "step": None,
                        },
                        "frames_per_slice": 10,
                        "frame_duration": None,
                        "center_energy": 33.2,
                    },
                    "sample_info": {
                        "id": None,
                        "name": None,
                        "source": None,
                        "reflectivity": None,
                    },
                },
            },
        ),
    ]

    def test_load_file_and_basic_attributes(
        self, sandbox_configuration, file, expected
    ):
        data = sandbox_configuration.load(file)
        assert isinstance(data, xr.Dataset)

        for k, v in expected.items():
            metadata = getattr(data.S, k)
            assert k and (metadata == expected[k])


class TestBasicDataLoading(object):
    """
    Tests procedures/plugins for loading basic data.
    """

    data = None

    scenarios = [
        # Lanzara Group "Main Chamber"
        (
            "main_chamber_load_cut",
            {
                "file": "basic/main_chamber_cut_0.fits",
                "expected": {
                    "dims": ["phi", "eV"],
                    "coords": {
                        "phi": [0.22165, 0.63879, 0.001745],
                        "eV": [-0.42558, 0.130235, 0.0023256],
                        "alpha": 0,
                        "x": -0.770435,
                        "y": 34.75,
                        "z": -3.4e-5,
                        "chi": -6.25 * np.pi / 180,
                    },
                    "offset_coords": {
                        "phi": 0.22165 - 0.405,
                        "beta": 0,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "main_chamber_load_map",
            {
                "file": "basic/main_chamber_map_1.fits",
                "expected": {
                    "dims": ["beta", "phi", "eV"],
                    "coords": {
                        "beta": [-0.314159, 0.61086, 0.03426],
                        "phi": [0.15184, 0.862193, 0.001745],
                        "eV": [-1.369977, 0.431409, 0.0046189],
                        "x": 2.99,
                        "y": 41.017,
                        "z": -0.03104,
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0.15184 - 0.405,
                        "beta": -0.314159,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "main_chamber_load_multi_region",
            {
                "file": "basic/main_chamber_PHONY_2.fits",
                "expected": {
                    "dims": ["phi", "eV"],
                    "coords": {
                        "phi": [0.22165, 0.63879, 0.001745],
                        "eV": [-0.42558, 0.130235, 0.0023256],
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0.22165 - 0.405,
                        "beta": 0,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "main_chamber_load_single_cycle",
            {
                "file": "basic/main_chamber_PHONY_3.fits",
                "expected": {
                    "dims": ["phi", "eV"],
                    "coords": {
                        "phi": [0.22165, 0.63879, 0.001745],
                        "eV": [-0.42558, 0.130235, 0.0023256],
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0.22165 - 0.405,
                        "beta": 0,
                        "chi": 0,
                    },
                },
            },
        ),
        # Lanzara Group "Spin-ToF"
        # ('stof_load_edc', {
        #     'file': 4,
        #     'expected': {},
        # }),
        # ('stof_load_spin_edc', {
        #     'file': 5,
        #     'expected': {},
        # }),
        # ('stof_load_map', {
        #     'file': 6,
        #     'expected': {},
        # }),
        # ('stof_load_spin_map', {
        #     'file': 7,
        #     'expected': {},
        # }),
        # ALS Beamline 7 "MAESTRO"
        (
            "maestro_load_cut",
            {
                "file": "basic/MAESTRO_12.fits",
                "expected": {
                    "dims": ["y", "x", "eV"],
                    "coords": {
                        "y": [4.961, 5.7618, 0.04],
                        "x": [0.86896, 1.6689, 0.04],
                        "eV": [-35.478, -31.2837, 0.00805],
                        "z": -0.4,
                        "alpha": 0,
                    },
                    "offset_coords": {"phi": 0, "theta": 0, "chi": 0},
                },
            },
        ),
        (
            "maestro_load_xps",
            {
                "file": "basic/MAESTRO_13.fits",
                "expected": {
                    "dims": ["y", "x", "eV"],
                    "coords": {
                        "y": [-0.92712, -0.777122, 0.010714],
                        "x": [0.42983, 0.57983, 0.010714],
                        "eV": [32.389, 39.9296, 0.011272],
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0,
                        "theta": 10.008 * np.pi / 180,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "maestro_load_map",
            {
                "file": "basic/MAESTRO_PHONY_14.fits",
                "expected": {
                    "dims": ["y", "x", "eV"],
                    "coords": {
                        "y": [4.961, 5.7618, 0.04],
                        "x": [0.86896, 1.6689, 0.04],
                        "eV": [-35.478, -31.2837, 0.00805],
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0,
                        "theta": 0,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "maestro_load_hv",
            {
                "file": "basic/MAESTRO_PHONY_15.fits",
                "expected": {
                    "dims": ["y", "x", "eV"],
                    "coords": {
                        "y": [4.961, 5.7618, 0.04],
                        "x": [0.86896, 1.6689, 0.04],
                        "eV": [-35.478, -31.2837, 0.00805],
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0,
                        "theta": 0,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "maestro_load_multi_region",
            {
                "file": "basic/MAESTRO_16.fits",
                "expected": {
                    "dims": ["eV"],
                    "coords": {"eV": [-1.5, 0.50644, 0.0228], "alpha": 0},
                    "offset_coords": {
                        "phi": 0,
                        "theta": 10.062 * np.pi / 180,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "maestro_load_nano_arpes_hierarchical_manipulator",
            {
                "file": "basic/MAESTRO_nARPES_focus_17.fits",
                "expected": {
                    "dims": ["optics_insertion", "y", "eV"],
                    "coords": {
                        "eV": [-35.16, -28.796, 0.01095],
                        "optics_insertion": [-100, 100, 10],
                        "y": [935.67, 935.77, -0.005],
                        "alpha": np.pi / 2,
                    },
                    "offset_coords": {
                        "phi": -0.4,
                        "theta": 1.4935e-6,
                        "chi": 0,
                    },
                },
            },
        ),
    ]

    def test_load_file_and_basic_attributes(
        self, sandbox_configuration, file, expected
    ):
        data = sandbox_configuration.load(file)
        assert isinstance(data, xr.Dataset)

        # assert basic dataset attributes
        for attr in ["location"]:  # TODO add spectrum type requirement
            assert attr in data.attrs

        # assert that all necessary coordinates are present
        necessary_coords = {
            "phi",
            "psi",
            "alpha",
            "chi",
            "beta",
            "theta",
            "x",
            "y",
            "z",
            "hv",
        }
        for necessary_coord in necessary_coords:
            assert necessary_coord in data.coords

        # assert basic spectrum attributes
        for attr in ["hv", "location"]:
            if attr == "hv" and (
                data.S.spectrum.attrs.get("spectrum_type") == "hv_map"
                or len(data.S.spectra) > 1
            ):
                continue
            assert attr in data.S.spectrum.attrs

        # assert dimensions
        assert list(data.S.spectra[0].dims) == expected["dims"]

        # assert coordinate shape
        by_dims = data.S.spectra[0].dims
        ranges = [
            [
                pytest.approx(data.coords[d].min().item(), 1e-3),
                pytest.approx(data.coords[d].max().item(), 1e-3),
                pytest.approx(data.G.stride(generic_dim_names=False)[d], 1e-3),
            ]
            for d in by_dims
        ]

        assert list(zip(by_dims, ranges)) == list(
            zip(by_dims, [expected["coords"][d] for d in by_dims])
        )
        for k, v in expected["coords"].items():
            if isinstance(v, float):
                assert k and (pytest.approx(data.coords[k].item(), 1e-3) == v)

        def safefirst(x):
            try:
                x = x[0]
            except (TypeError, IndexError):
                pass

            try:
                x = x.item()
            except AttributeError:
                pass

            return x

        for k in expected["offset_coords"].keys():
            offset = safefirst(data.S.spectra[0].S.lookup_offset_coord(k))
            assert k and (pytest.approx(offset, 1e-3) == expected["offset_coords"][k])

        kspace_data = convert_to_kspace(data.S.spectra[0])
        assert isinstance(kspace_data, xr.DataArray)
