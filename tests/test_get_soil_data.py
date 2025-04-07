"""
Module Name: test_get_soil_data.py
Description: Test get_soil_data functions for soilgrids building block.

Developed in the BioDT project by Thomas Banitz (UFZ) with contributions by Franziska Taubert (UFZ)
and Tuomas Rossi (CSC).

Copyright (C) 2024
- Helmholtz Centre for Environmental Research GmbH - UFZ, Germany
- CSC - IT Center for Science Ltd., Finland

Licensed under the EUPL, Version 1.2 or - as soon they will be approved
by the European Commission - subsequent versions of the EUPL (the "Licence").
You may not use this work except in compliance with the Licence.

You may obtain a copy of the Licence at:
https://joinup.ec.europa.eu/software/page/eupl

This project has received funding from the European Union's Horizon Europe Research and Innovation
Programme under grant agreement No 101057437 (BioDT project, https://doi.org/10.3030/101057437).
The authors acknowledge the EuroHPC Joint Undertaking and CSC - IT Center for Science Ltd., Finland
for awarding this project access to the EuroHPC supercomputer LUMI, hosted by CSC - IT Center for
Science Ltd., Finland and the LUMI consortium through a EuroHPC Development Access call.
"""

from pathlib import Path

import numpy as np
import pytest

from soilgrids.get_soil_data import (
    HIHYDROSOIL_SPECS,
    configure_soilgrids_request,
    construct_soil_data_file_name,
    download_soilgrids,
    get_hihydrosoil_data,
    get_hihydrosoil_map_file,
    get_soilgrids_data,
    shape_soildata_for_file,
)


def test_construct_soil_data_file_name():
    """Test construct_soil_data_file_name function."""
    coordinates = {"lat": 12.123456789, "lon": 99.9}
    file_name = construct_soil_data_file_name(coordinates)
    assert str(file_name).endswith(
        "soilDataFolder\\lat12.123457_lon99.900000__2020__soil.txt"
    )

    file_name = construct_soil_data_file_name(
        coordinates, folder="test_folder", data_format="xyz"
    )
    assert str(file_name).endswith(
        "test_folder\\lat12.123457_lon99.900000__2020__soil.xyz"
    )

    # Remove test folders if they were just created (i.e. they are empty)
    folders = ["soilDataFolder", "test_folder"]

    for folder in folders:
        folder = Path(folder)

        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def test_shape_soildata_for_file():
    """Test shape_soildata_for_file function."""
    soil_data_1D = np.array([1, 2, 3, 4])
    soil_data_2D = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])  # 4x3 array

    # 1D arrays must be one row, with one column per element
    assert shape_soildata_for_file(soil_data_1D).shape == (1, 4)

    # 2D arrays must be one row per entry, with one column per element
    shaped_data = shape_soildata_for_file(soil_data_2D)
    assert shaped_data.shape == (3, 4)

    # All rows in the shaped must be [1, 2, 3, 4]
    for index in range(3):
        assert all(shaped_data[index] == soil_data_1D)

    # Error for 3D array
    with pytest.raises(ValueError):
        shape_soildata_for_file(np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]]))


def test_configure_soilgrids_request():
    """Test configure_soilgrids_request function."""
    coordinates = {"lat": 12.123456789, "lon": 99.9}
    request = configure_soilgrids_request(coordinates)

    # Expected output
    expected_request = {
        "url": "https://rest.isric.org/soilgrids/v2.0/properties/query",
        "params": {
            "lon": coordinates["lon"],
            "lat": coordinates["lat"],
            "property": ["silt", "clay", "sand"],
            "depth": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"],
            "value": ["mean"],
        },
    }

    assert request == expected_request


def test_download_soilgrids():
    """Test download_soilgrids function."""
    coordinates = {"lat": 12.123456789, "lon": 99.9}
    request = configure_soilgrids_request(coordinates)
    json_data, time_stamp = download_soilgrids(request)

    # Check time stamp format is 'yyyy-mm-ddThh:mm:ss+00:00'
    assert time_stamp.endswith("+00:00")
    assert len(time_stamp) == 25
    assert len(time_stamp.split(":")) == 4
    assert len(time_stamp.split("T")) == 2

    # Check structure and some content of JSON data
    assert json_data is not None
    assert isinstance(json_data, dict)
    assert "properties" in json_data

    layer_names = []

    for layer in json_data["properties"]["layers"]:
        layer_names.append(layer["name"])

        assert "unit_measure" in layer
        assert len(layer["depths"]) == 6

        # Check for the first depth
        top_layer = layer["depths"][0]
        assert top_layer["label"] == "0-5cm"
        assert "mean" in top_layer["values"]

    layer_names.sort()
    assert layer_names == ["clay", "sand", "silt"]


def test_get_soilgrids_data():
    """Test get_soilgrids_data function."""
    coordinates = {"lat": 12.123456789, "lon": 99.9}
    request = configure_soilgrids_request(coordinates)
    json_data, _ = download_soilgrids(request)
    property_names = ["silt", "clay", "sand"]
    layer_count = 6
    property_data = get_soilgrids_data(json_data)

    assert np.shape(property_data) == (len(property_names), layer_count)

    # Check that values are valid and sum to 100% for each layer
    for index in range(layer_count):
        assert np.sum(property_data[:, index]) == pytest.approx(
            100.0, abs=1e-1
        )  # allow 0.1% error
        assert all(property_data[:, index] >= 0.0)
        assert all(property_data[:, index] <= 100.0)


def test_get_hihydrosoil_map_file(tmp_path, caplog):
    """Test get_hihydrosoil_map_file function."""
    property_name = "Ksat"
    depth = "0-5cm"
    opendap_folder = "http://opendap.biodt.eu/grasslands-pdt/soilMapsHiHydroSoil/"
    file_name = f"{property_name}_{depth}_M_250m.tif"
    expected_url = f"{opendap_folder}{file_name}"
    local_file = tmp_path / file_name
    local_file.touch()  # Create an empty file for testing

    # Test with local file
    assert get_hihydrosoil_map_file(property_name, depth, cache=tmp_path) == local_file
    assert "Trying to access via URL ..." not in caplog.text  # No URL access

    caplog.clear()

    # Test with missing file
    local_file.unlink()  # Remove the file
    with caplog.at_level("ERROR"):
        assert (
            get_hihydrosoil_map_file(property_name, depth, cache=tmp_path)
            == expected_url
        )
        assert f"Local file '{local_file}' not found!" in caplog.text

    caplog.clear()

    # Test without cache
    assert get_hihydrosoil_map_file(property_name, depth) == expected_url

    # Test with invalid property name or depth
    with caplog.at_level("ERROR"):
        assert get_hihydrosoil_map_file("invalid_property", depth) is None
        assert (
            f"File '{opendap_folder}invalid_property_{depth}_M_250m.tif' not found!"
            in caplog.text
        )

        assert get_hihydrosoil_map_file(property_name, "invalid_depth") is None
        assert (
            f"File '{opendap_folder}{property_name}_invalid_depth_M_250m.tif' not found!"
            in caplog.text
        )


def test_get_hihydrosoil_data():
    """Test get_hihydrosoil_data function."""
    coordinates = {"lat": 50.0, "lon": 10.0}
    hhs_data, query_protocol = get_hihydrosoil_data(coordinates)
    layer_count = 6

    # Check query protocol
    assert len(query_protocol) == len(HIHYDROSOIL_SPECS) * layer_count

    for query in query_protocol:
        assert len(query) == 2
        assert query[0].startswith(
            "http://opendap.biodt.eu/grasslands-pdt/soilMapsHiHydroSoil/"
        )

        # Check time stamp format is 'yyyy-mm-ddThh:mm:ss+00:00'
        assert query[1].endswith("+00:00")
        assert len(query[1]) == 25
        assert len(query[1].split(":")) == 4
        assert len(query[1].split("T")) == 2

    # Check data
    assert hhs_data.shape == (len(HIHYDROSOIL_SPECS), layer_count)
    for row in hhs_data:
        assert all(np.isnan(row) | np.isfinite(row))  # allow NaN
        assert all(row[np.isfinite(row)] >= 0.0)  # no negative values
