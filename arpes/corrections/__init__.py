import lmfit as lf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from arpes.fits import QuadraticModel, GStepBModel, broadcast_model
from arpes.utilities import apply_dataarray
from arpes.utilities.math import shift_by
from arpes.provenance import provenance

def exclude_hemisphere_axes(axes_list: list):
    hemisphere_axes = {'phi', 'eV'}
    return list(set(axes_list).difference(hemisphere_axes))

def exclude_hv_axes(axes_list: list):
    hv_axes = {'hv', 'eV'}
    return list(set(axes_list).difference(hv_axes))


def build_quadratic_fermi_edge_correction(arr: xr.DataArray, fit_limit=0.001, plot=False) -> lf.model.ModelResult:
    edge_fit = broadcast_model(GStepBModel, arr.sum(exclude_hemisphere_axes(arr.dims)).sel(eV=slice(-0.2, 0.1)), 'phi')

    quadratic_corr = QuadraticModel().guess_fit(
        apply_dataarray(edge_fit, np.vectorize(lambda x: x.params['center'].value)),
        weights=(apply_dataarray(edge_fit, np.vectorize(lambda x: x.params['center'].stderr)).values < fit_limit) * 1)
    if plot:
        apply_dataarray(edge_fit, np.vectorize(lambda x: x.params['center'].value)).plot()
        plt.plot(arr.coords['phi'], quadratic_corr.best_fit)

    return quadratic_corr

def build_photon_energy_fermi_edge_correction(arr: xr.DataArray, plot=False, energy_window=0.2):
    edge_fit = broadcast_model(GStepBModel, arr.sum(exclude_hv_axes(arr.dims)).sel(
        eV=slice(-energy_window, energy_window)), 'hv')

    return edge_fit

def apply_photon_energy_fermi_edge_correction(arr: xr.DataArray, correction=None, **kwargs):
    if correction is None:
        correction = build_photon_energy_fermi_edge_correction(arr, **kwargs)

    correction_values = apply_dataarray(correction, np.vectorize(lambda x: x.params['center'].value))
    arr.attrs['hv_correction'] = correction_values.values

    shift_amount = -correction_values / (arr.coords['eV'].values[1] - arr.coords['eV'].values[0])
    energy_axis = arr.dims.index('eV')
    hv_axis = arr.dims.index('hv')

    corrected_arr = xr.DataArray(
        shift_by(arr.values, shift_amount, axis=energy_axis, by_axis=hv_axis, order=1),
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    del corrected_arr.attrs['id']
    provenance(corrected_arr, arr, {
        'what': 'Shifted Fermi edge to align at 0 along hv axis',
        'by': 'apply_photon_energy_fermi_edge_correction',
        'correction': correction_values,
    })

    return corrected_arr

def apply_quadratic_fermi_edge_correction(arr: xr.DataArray, correction: lf.model.ModelResult=None):
    if correction is None:
        correction = build_quadratic_fermi_edge_correction(arr)

    arr.attrs['FE_Corr'] = correction.best_values

    delta_E = arr.coords['eV'].values[1] - arr.coords['eV'].values[0]
    energy_axis = arr.dims.index('eV')
    phi_axis = arr.dims.index('phi')
    shift_amount = -correction.best_fit / delta_E

    corrected_arr = xr.DataArray(
        shift_by(arr.values, shift_amount, axis=energy_axis, by_axis=phi_axis, order=1),
        arr.coords,
        arr.dims,
        attrs=arr.attrs
    )

    del corrected_arr.attrs['id']
    provenance(corrected_arr, arr, {
        'what': 'Shifted Fermi edge to align at 0',
        'by': 'apply_quadratic_fermi_edge_correction',
        'correction': correction.best_values,
    })

    return corrected_arr