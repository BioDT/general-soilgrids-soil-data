"""
Module Name: utils.py
Description: Utility functions for soilgrids building block.

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

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyproj
import rasterio
import requests

from soilgrids.logger_config import logger


def reproject_coordinates(lat, lon, target_crs):
    """
    Reproject latitude and longitude coordinates to a target CRS.

    Parameters:
        lat (float): Latitude.
        lon (float): Longitude.
        target_crs (crs or str): Target CRS (as CRS object or str).

    Returns:
        tuple (float): Reprojected coordinates (easting, northing).
    """
    # Define the source CRS (EPSG:4326 - WGS 84, commonly used for lat/lon)
    src_crs = "EPSG:4326"

    # Create a transformer to convert from the source CRS to the target CRS
    # (always_xy: use lon/lat for source CRS and east/north for target CRS)
    transformer = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)

    # Reproject the coordinates (order is lon, lat!)
    east, north = transformer.transform(lon, lat)

    return east, north


def extract_raster_value(tif_file, coordinates, *, band_number=1, attempts=5, delay=2):
    """
    Extract value from raster file at specified coordinates.

    Parameters:
        tif_file (str): TIF file path or URL.
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        band_number (int): Band number for which the value shall be extracted (default is 1).
        attempts (int): Number of attempts to open the TIF file in case of errors (default is 5).
        delay (int): Number of seconds to wait between attempts (default is 2).

    Returns:
        tuple: Extracted value (None if extraction failed), and time stamp.
    """
    while attempts > 0:
        time_stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        try:
            with rasterio.open(tif_file) as src:
                # Check if band number exists in the raster file
                if band_number not in src.indexes:
                    try:
                        raise ValueError(
                            f"Band number {band_number} does not exist in the raster file {tif_file}."
                        )
                    except ValueError as e:
                        logger.error(e)
                        raise

                # Reproject coordinates (HiHydroSoil works with lat/lon too, but better to keep transformation in)
                east, north = reproject_coordinates(
                    coordinates["lat"],
                    coordinates["lon"],
                    src.crs,  # TIF file CRS
                )

                # Extract the value at the specified coordinates
                value = next(src.sample([(east, north)], indexes=band_number))

            return value[0], time_stamp
        except rasterio.errors.RasterioError as e:
            attempts -= 1
            logger.error(f"Reading TIF file failed (Error {e}).")

            if attempts > 0:
                logger.info(f" Retrying in {delay} seconds ...")
                time.sleep(delay)
            else:
                return None, time_stamp


def check_url(url, *, attempts=3, delay=2):
    """
    Check if a URL exists and, if so, return the URL (original or redirected).

    Parameters:
        url (str): URL to check.
        attempts (int): Number of attempts in case of connection errors or specific status codes (default is 3).
        delay (int): Number of seconds to wait between attempts (default is 2).

    Returns:
        str: URL if existing (original or redirected), None otherwise.
    """
    if not url:
        return None

    retry_status_codes = {502, 503, 504}

    while attempts > 0:
        try:
            response = requests.head(url, allow_redirects=True)

            if response.status_code == 200:
                return response.url
            elif response.status_code in retry_status_codes:
                attempts -= 1

                if attempts > 0:
                    time.sleep(delay)
            else:
                logger.error(
                    f"Invalid URL: {url} (Status code {response.status_code})."
                )
                return None
        except requests.exceptions.MissingSchema:
            logger.error(f"Invalid URL format: {url}")
            return None
        except requests.ConnectionError as e:
            attempts -= 1

            if attempts > 0:
                time.sleep(delay)
            else:
                logger.error(f"Connection error: {e}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    return None


def list_to_file(list_to_write, file_name, *, column_names=None):
    """
    Write a list to a text file (tab-separated) or csv file (;-separated) or an Excel file.

    Parameters:
        list_to_write (list): List of strings or tuples or dictionaries to be written to the file.
        file_name (str or Path): Path of output file (suffix determines file type).
        column_names (list): List of column names (strings) to write as header line (default is None).
    """
    # Convert string entries to single item tuples
    list_to_write = [
        (entry,) if isinstance(entry, str) else entry for entry in list_to_write
    ]

    # Check if list_to_write contains dictionaries
    if any(isinstance(entry, dict) for entry in list_to_write):
        if not all(isinstance(entry, dict) for entry in list_to_write):
            try:
                raise ValueError(
                    "All entries in the list must be either dictionaries or not dictionaries. Cannot write list with mixed types."
                )
            except ValueError as e:
                logger.error(e)
                raise

        # Get column names from dictionaries (keys of first dictionary) if not provided
        if not column_names:
            logger.warning(
                "No column names provided. Using keys from first dictionary in list to obtain column names."
            )
            column_names = list(list_to_write[0].keys())

        if not column_names:
            try:
                raise ValueError(
                    "No column names provided and no keys found in dictionaries to obtain column names. Cannot write list."
                )
            except ValueError as e:
                logger.error(e)
                raise

        # Convert dictionaries to lists of values based on column_names, empty string if key not found
        list_to_write = [
            [entry.get(col, "") for col in column_names] for entry in list_to_write
        ]
    else:
        # Check if all tuples in list have the same length as the column_names list
        if column_names and not all(
            len(entry) == len(column_names) for entry in list_to_write
        ):
            try:
                raise ValueError(
                    "All entries in the list must have the same length as the column names list."
                )
            except ValueError as e:
                logger.error(e)
                raise

    file_path = Path(file_name)
    file_suffix = file_path.suffix.lower()

    # Create data directory if missing
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    if file_suffix in [".txt", ".csv"]:
        with open(
            file_path, "w", newline="", encoding="utf-8", errors="replace"
        ) as file:
            writer = (
                csv.writer(file, delimiter="\t")
                if file_suffix == ".txt"
                else csv.writer(file, delimiter=";")
            )

            if column_names:
                header = column_names
                writer.writerow(header)  # Header row

            for entry in list_to_write:
                writer.writerow(entry)
    elif file_suffix == ".xlsx":
        df = pd.DataFrame(list_to_write, columns=column_names)
        df.to_excel(file_path, index=False)
    else:
        try:
            raise ValueError(
                "Unsupported file format. Supported formats are '.txt', '.csv' and '.xlsx'."
            )
        except ValueError as e:
            logger.error(e)
            raise

    logger.info(f"List written to file '{file_name}'.")
