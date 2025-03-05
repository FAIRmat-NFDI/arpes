from collections.abc import Sequence
from typing import Optional, Union

import h5py
import numpy as np
import xarray as xr
from pint import Quantity

from arpes.config import ureg
from arpes.endstations import SingleFileEndstation, add_endstation

__all__ = ["NeXusEndstation"]

nexus_translation_table = {
    "sample/transformations/trans_x": "x",
    "sample/transformations/trans_y": "y",
    "sample/transformations/trans_z": "z",
    "sample/transformations/sample_polar": "theta",
    "sample/transformations/offset_polar": "theta_offset",
    "sample/transformations/sample_tilt": "beta",
    "sample/transformations/offset_tilt": "beta_offset",
    "sample/transformations/sample_azimuth": "chi",
    "sample/transformations/offset_azimuth": "chi_offset",
    "instrument/beam_probe/incident_energy": "hv",
    "instrument/electronanalyser/work_function": "work_function",
    "instrument/electronanalyser/transformations/analyzer_rotation": "alpha",
    "instrument/electronanalyser/transformations/analyzer_elevation": "psi",
    "instrument/electronanalyser/transformations/analyzer_dispersion": "phi",
    "instrument/electronanalyser/energydispersion/kinetic_energy": "eV",
}


class NeXusEndstation(SingleFileEndstation):
    """An endstation for reading arpes data from a nexus file."""

    PRINCIPAL_NAME = "NXmpes"

    _TOLERATED_EXTENSIONS = {
        ".nxs",
    }

    def load_nexus_file(self, filepath: str, entry_name: str = "entry") -> xr.DataArray:
        """
        Loads an MPES NeXus file and creates a DataArray from it.

        Args:
            filepath (str): The path of the .nxs file.
            entry_name (str, optional):
                The name of the entry to process. Defaults to "entry".

        Raises:
            KeyError:
                Thrown if dependent axis are not found in the nexus file.

        Returns:
            xr.DataArray: The data read from the .nxs file.
        """

        def write_value(name: str, dataset: h5py.Dataset):
            if str(dataset.dtype) == "bool":
                attributes[name] = bool(dataset[()])
            elif dataset.dtype.kind in "iufc":
                attributes[name] = dataset[()]
                if "units" in dataset.attrs:
                    attributes[name] = attributes[name] * ureg(dataset.attrs["units"])
            elif dataset.dtype.kind in "O" and dataset.shape == ():
                attributes[name] = dataset[()].decode()

        def is_valid_metadata(name: str) -> bool:
            invalid_end_paths = ["depends_on"]
            invalid_start_paths = ["data", "process"]
            for invalid_path in invalid_start_paths:
                if name.startswith(invalid_path):
                    return False
            for invalid_path in invalid_end_paths:
                if name.endswith(invalid_path):
                    return False
            return True

        def parse_attrs(name: str, dataset: Union[h5py.Dataset, h5py.Group]):
            short_path = name.split("/", 1)[-1]
            if isinstance(dataset, h5py.Dataset) and is_valid_metadata(short_path):
                write_value(short_path, dataset)

        def translate_nxmpes_to_pyarpes(attributes: dict) -> dict:
            for key, newkey in nexus_translation_table.items():
                if key in attributes:
                    try:
                        if attributes[key].units == "degree":
                            attributes[newkey] = attributes[key].to(ureg.rad)
                        else:
                            attributes[newkey] = attributes[key]
                    except AttributeError:
                        attributes[newkey] = attributes[key]
                    # flip sign of offsets, as they are subtracted in pyARPES rather than added
                    if newkey.find("offset") > -1:
                        attributes[newkey] *= -1

            # remove axis arrays from static coordinates:
            for axis in self.ENSURE_COORDS_EXIST:
                if axis in attributes and (
                    isinstance(attributes[axis], (Sequence, np.ndarray))
                    or (
                        isinstance(attributes[axis], Quantity)
                        and (
                            isinstance(
                                attributes[axis].magnitude, (Sequence, np.ndarray)
                            )
                        )
                    )
                ):
                    if len(attributes[axis]) > 0:
                        attributes[axis] = attributes[axis][0]

            return attributes

        def load_nx_data(nxdata: h5py.Group, attributes: dict) -> xr.DataArray:
            axes = nxdata.attrs["axes"]

            # handle moving axes
            new_axes = []
            for axis in axes:
                if f"reference" not in nxdata[axis].attrs:
                    raise KeyError(f"Reference attribute not found for axis {axis}.")

                axis_reference: str = nxdata[axis].attrs["reference"]
                axis_reference_key = axis_reference.split("/", 2)[-1]
                new_axes.append(nexus_translation_table[axis_reference_key])
                if nexus_translation_table[axis_reference_key] in attributes:
                    attributes.pop(nexus_translation_table[axis_reference_key])

            coords = {}
            for axis, new_axis in zip(axes, new_axes):
                coords[new_axis] = nxdata[axis][:] * ureg(nxdata[axis].attrs["units"])
                if coords[new_axis].units == "degree":
                    coords[new_axis] = coords[new_axis].to(ureg.rad)
            data = nxdata[nxdata.attrs["signal"]][:]
            dims = new_axes

            dataset = xr.DataArray(data, coords=coords, dims=dims, attrs=attributes)

            return dataset

        data_path = f"/{entry_name}/data"
        with h5py.File(filepath, "r") as h5file:
            attributes = {}
            h5file.visititems(parse_attrs)
            attributes = translate_nxmpes_to_pyarpes(attributes)
            dataset = load_nx_data(h5file[data_path], attributes)
            return dataset

    def load_single_frame(
        self,
        frame_path: Optional[str] = None,
        scan_desc: Optional[dict] = None,
        **kwargs,
    ) -> xr.Dataset:
        if frame_path is None:
            return xr.Dataset()
        data = self.load_nexus_file(frame_path)
        return xr.Dataset({"spectrum": data}, attrs=data.attrs)


add_endstation(NeXusEndstation)
