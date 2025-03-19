"""
Module Name: test_utils.py
Description: Test utility functions for soilgrids building block.

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

import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio

from soilgrids.utils import (
    check_url,
    extract_raster_value,
    list_to_file,
    reproject_coordinates,
)


def test_reproject_coordinates():
    """Test reproject_coordinates function."""
    # Input coordinates as lat lon pairs in EPSG:4326 (WGS 84) format
    lat_lon_pairs = [
        (52.52, 13.405),  # Berlin, Germany
        (60.1695, 24.9354),  # Helsinki, Finland
        (41.9028, 12.4964),  # Rome, Italy
    ]
    target_crs_list = [
        "EPSG:4326",  # WGS 84, same as input
        "EPSG:32633",  # WGS 84 / UTM zone 33N
        "EPSG:3035",  # ETRS89 / LAEA Europe
        "EPSG:3857",  # WGS 84 / Pseudo-Mercator
    ]

    for lat, lon in lat_lon_pairs:
        # Test if reprojecting to the same CRS as input returns the same coordinates
        assert reproject_coordinates(lat, lon, "EPSG:4326") == (lon, lat)

        # Test reprojecting to different target CRS, inlcuding the same as input
        for target_crs in target_crs_list:
            expected_east_north = pyproj.Transformer.from_crs(
                "EPSG:4326", target_crs, always_xy=True
            ).transform(lon, lat)
            generated_east_north = reproject_coordinates(lat, lon, target_crs)

            assert np.allclose(
                expected_east_north, generated_east_north, atol=0, rtol=1e-12
            )


def test_extract_raster_value(tmp_path):
    """Test extract_raster_value function."""
    # Create test raster file with EPSG:4326 CRS (same as input coordinates) and 2 bands
    raster_file = tmp_path / "test_raster.tif"

    with rasterio.open(
        raster_file,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=2,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=rasterio.Affine(0.1, 0, 13.0, 0, -0.1, 52.0),
        nodata=255,
    ) as dst:
        dst.write(
            np.array(
                [[col + row * 10 for col in range(10)] for row in range(10)],
                dtype=np.float32,
            ),
            1,
        )
        dst.write(
            np.array(
                [[row * 10 - col for col in range(10)] for row in range(10)],
                dtype=np.float32,
            ),
            2,
        )

    # Test with invalid band number
    with pytest.raises(ValueError):
        extract_raster_value(raster_file, {"lat": 52.0, "lon": 13.0}, band_number=3)

    # Test with coordinates out of range
    value, _ = extract_raster_value(raster_file, {"lat": 10.0, "lon": 80.0})
    assert value == 255

    # Test with valid coordinates and band number
    valid_coordinates = [
        {"lat": 52.0, "lon": 13.0},  # Top left corner
        {"lat": 51.5, "lon": 13.5},  # Center
        {"lat": 52.0, "lon": 13.9},  # Top right corner
        {"lat": 51.1, "lon": 13.0},  # Bottom left corner
        {"lat": 51.1, "lon": 13.9},  # Bottom right corner
    ]

    for band_number in [1, 2]:
        for coordinates in valid_coordinates:
            value, _ = extract_raster_value(
                raster_file, coordinates, band_number=band_number
            )
            expected_value = rasterio.open(raster_file).read(band_number)[
                round((coordinates["lat"] - 52.0) / -0.1),
                round((coordinates["lon"] - 13.0) / 0.1),
            ]
            assert value == expected_value

    # Test with example raster file from URL source that has Lambert Azimuthal Equal Area projection
    raster_file = "http://opendap.biodt.eu/grasslands-pdt/landCoverMaps/GER_Preidl/preidl-etal-RSE-2020_land-cover-classification-germany-2016.tif"
    expected_values = [
        ({"lat": 51.3373, "lon": 10.5367}, 10),  # GER, rapeseed
        ({"lat": 51.047796, "lon": 10.846754}, 15),  # GER, stone fruit
        ({"lat": 51.0581341, "lon": 10.8537121}, 19),  # GER, grassland
        ({"lat": 51.4429008, "lon": 12.3409231}, 21),  # GER, water
        ({"lat": 49.8366436, "lon": 18.1540575}, 255),  # CZ, out of map range
    ]

    for target in expected_values:
        value, _ = extract_raster_value(raster_file, target[0])
        assert value == target[1]


def test_check_url():
    """Test check_url function."""
    assert check_url("https://biodt.eu/") == "https://biodt.eu/"
    assert check_url("http://biodt.eu") == "https://biodt.eu/"  # redirected
    assert check_url("invalid_schema") is None
    assert check_url("http://example.com/invalid|character") is None
    assert (
        check_url("http://invalid_url") is None
    )  # takes some time due to retry attempts


def test_list_to_file(tmp_path):
    """Test writing a list to a file."""

    # Different test lists: tuples, strings, dictionaries
    lists_to_write = [
        [("a", "b", "c"), ("d", "e", "f")],
        ["abc", "def"],
        [
            {"col1": "a", "col2": "b", "col3": "c"},
            {
                "col2": "e",
                "col3": "f",
                "colx": "x",
                "col1": "d",
            },  # Order of keys is not guaranteed, extra keys may exist
        ],
    ]
    target_strings = {
        "txt": [("a\tb\tc", "d\te\tf"), ("abc", "def"), ("a\tb\tc", "d\te\tf")],
        "csv": [("a;b;c", "d;e;f"), ("abc", "def"), ("a;b;c", "d;e;f")],
    }
    column_names_list = [["col1", "col2", "col3"], ["col1"], ["col1", "col2", "col3"]]
    column_names_strings = {
        "txt": ["col1\tcol2\tcol3", "col1", "col1\tcol2\tcol3"],
        "csv": ["col1;col2;col3", "col1", "col1;col2;col3"],
    }

    def validate_file_content(content, target_strings, *, column_names_string=None):
        if column_names_string is None:
            for index, target_string in enumerate(target_strings):
                assert content[index].strip() == target_string
        else:
            assert content[0].strip() == column_names_string

            for index, target_string in enumerate(target_strings):
                assert content[index + 1].strip() == target_string

    for index, list_to_write in enumerate(lists_to_write):
        # Test with txt file and csv files
        for suffix in ["txt", "csv"]:
            file_path = tmp_path / f"test.{suffix}"
            list_to_file(list_to_write, file_path)  # Default is without column names

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.readlines()

                if isinstance(list_to_write[0], dict):
                    # Keys of first entry are used as column names for dictionaries
                    validate_file_content(
                        content,
                        target_strings[suffix][index],
                        column_names_string=column_names_strings[suffix][index],
                    )
                else:
                    validate_file_content(content, target_strings[suffix][index])

            # Add column names
            list_to_file(
                list_to_write, file_path, column_names=column_names_list[index]
            )

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.readlines()
                validate_file_content(
                    content,
                    target_strings[suffix][index],
                    column_names_string=column_names_strings[suffix][index],
                )

        # Test with xls files
        file_path = tmp_path / "test.xlsx"
        list_to_file(list_to_write, file_path)
        content = pd.read_excel(file_path)

        if isinstance(list_to_write[0], dict):
            # use first list as target also for dictionaries
            target_list = lists_to_write[0]
            assert np.all(content.columns == column_names_list[index])
        else:
            target_list = list_to_write

        for row_index, row in enumerate(target_list):
            assert np.all(content.values[row_index] == row)

        # Add column names
        list_to_file(list_to_write, file_path, column_names=column_names_list[index])
        content = pd.read_excel(file_path)
        assert np.all(content.columns == column_names_list[index])

        for row_index, row in enumerate(target_list):
            assert np.all(content.values[row_index] == row)

    # Test with invalid column names
    with pytest.raises(ValueError):
        list_to_file([("a", "b", "c")], file_path, column_names=["col1", "col2"])
