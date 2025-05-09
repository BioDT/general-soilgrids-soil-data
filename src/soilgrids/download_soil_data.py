"""
Module Name: download_soil_data.py
Description: Functions for downloading and processing selected soil data at given location from
             SoilGrids and derived data sources (SoilGrids REST API, HiHydroSoil maps).

Developed in the BioDT project by Thomas Banitz (UFZ) with contributions by Franziska Taubert (UFZ),
Tuomas Rossi (CSC) and Taimur Haider Khan (UFZ).

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

Data sources:
    SoilGrids™ 2.0:
    - Poggio L., Sousa L.M., Batjes N.H., Heuvelink G.B., Kempen B., Ribeiro E., Rossiter D. (2021):
      SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty.
      SOIL 7: 217-240. https://doi.org/10.5194/soil-7-217-2021
    - Website: https://soilgrids.org/
    - Access via API: https://rest.isric.org/soilgrids/v2.0/docs

    HiHydroSoil v2.0:
    - Simons, G.W.H., R. Koster, P. Droogers. (2020):
      HiHydroSoil v2.0 - A high resolution soil map of global hydraulic properties.
      FutureWater Report 213.
    - Website: https://www.futurewater.eu/projects/hihydrosoil/
    - Access via TIF Maps, provided upon request to FutureWater
    - Redistributed with permission and without changes at:
      http://opendap.biodt.eu/grasslands-pdt/soilMapsHiHydroSoil/
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import numpy as np
import requests

from soilgrids import utils as ut
from soilgrids.logger_config import logger

# Define HiHydroSoil variable specifications, including:
#     hihydrosoil_name: HiHydroSoil variable name.
#     hihydrosoil_unit: HiHydroSoil unit.
#     map_to_float: Conversion factor from HiHydroSoil integer map value to actual float number.
#     hihydrosoil_to_grassmodel: Conversion factor from HiHydroSoil unit to grassland model unit.
#     grassmodel_unit: Grassland model unit.
#     grassmodel_name: Grassland model variable name, as used in final soil data file.
HIHYDROSOIL_SPECS = MappingProxyType(
    {
        "field capacity": {
            "hihydrosoil_name": "WCpF2",
            "hihydrosoil_unit": "m³/m³",
            "map_to_float": 1e-4,
            "hihydrosoil_to_grassmodel": 1e2,  # to %
            "grassmodel_unit": "V%",
            "grassmodel_name": "FC[V%]",
        },
        "permanent wilting point": {
            "hihydrosoil_name": "WCpF4.2",
            "hihydrosoil_unit": "m³/m³",
            "map_to_float": 1e-4,
            "hihydrosoil_to_grassmodel": 1e2,  # to %
            "grassmodel_unit": "V%",
            "grassmodel_name": "PWP[V%]",
        },
        "soil porosity": {
            "hihydrosoil_name": "WCsat",
            "hihydrosoil_unit": "m³/m³",
            "map_to_float": 1e-4,
            "hihydrosoil_to_grassmodel": 1e2,  # to %
            "grassmodel_unit": "V%",
            "grassmodel_name": "POR[V%]",
        },
        "saturated hydraulic conductivity": {
            "hihydrosoil_name": "Ksat",
            "hihydrosoil_unit": "cm/d",
            "map_to_float": 1e-4,
            "hihydrosoil_to_grassmodel": 1e1,  # cm to mm
            "grassmodel_unit": "mm/d",
            "grassmodel_name": "KS[mm/d]",
        },
    }
)


def construct_soil_data_file_name(
    coordinates, *, folder="soilDataFolder", data_format="txt"
):
    """
    Construct data file name.

    Parameters:
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        folder (str or Path): Folder where the data file will be stored (default is 'soilDataFolder').
        data_format (str): File suffix (default is 'txt').

    Returns:
        Path: Constructed data file name as a Path object.
    """
    # Get folder with path appropriate for different operating systems create folder if missing
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if "lat" in coordinates and "lon" in coordinates:
        formatted_lat = f"lat{coordinates['lat']:.6f}"
        formatted_lon = f"lon{coordinates['lon']:.6f}"
        file_start = f"{formatted_lat}_{formatted_lon}"
    else:
        try:
            raise ValueError(
                "Coordinates not correctly defined. Please provide as dictionary ({'lat': float, 'lon': float})!"
            )
        except ValueError as e:
            logger.error(e)
            raise

    file_name = folder / f"{file_start}__2020__soil.{data_format}"

    return file_name


def shape_soildata_for_file(array):
    """
    Reshape a 1D array to 2D or transpose a 2D array.

    Parameters:
        array (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Reshaped or transposed array.

    Raises:
        ValueError: If the input array is not 1D or 2D.
    """
    if array.ndim == 1:
        return array.reshape(1, -1)
    elif array.ndim == 2:
        return np.transpose(array)
    else:
        try:
            raise ValueError("Input array must be 1D or 2D.")
        except ValueError as e:
            logger.error(e)
            raise


def configure_soilgrids_request(
    coordinates, *, property_names=["silt", "clay", "sand"]
):
    """
    Configure a request for SoilGrids API based on given coordinates and properties.

    Parameters:
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        property_names (list): List of properties to download (default is ['silt', 'clay', 'sand']).

    Returns:
        dict: Request configuration including URL and parameters.
    """
    return {
        "url": "https://rest.isric.org/soilgrids/v2.0/properties/query",
        "params": {
            "lon": coordinates["lon"],
            "lat": coordinates["lat"],
            "property": property_names,
            "depth": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"],
            "value": ["mean"],
        },
    }

    # full options, Q0.5=median
    # "property": ["bdod", "cec", "cfvo", "clay", "nitrogen", "ocd", "ocs", "phh2o", "sand", "silt", "soc", "wv0010", "wv0033", "wv1500"],
    # "depth": ["0-5cm", "0-30cm" ????, "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"],
    # "value": ["Q0.05", "Q0.5", "Q0.95", "mean", "uncertainty"]


def download_soilgrids(request, *, attempts=6, delay_exponential=8, delay_linear=2):
    """
    Download data from SoilGrids REST API with retry functionality.

    Parameters:
        request (dict): Dictionary containing the request URL (key: 'url') and parameters (key: 'params').
        attempts (int): Total number of attempts (including the initial try). Default is 6.
        delay_exponential (int): Initial delay in seconds for request rate limit errors (default is 8).
        delay_linear (int): Delay in seconds for gateway errors and other failed requests (default is 2).

    Returns:
        tuple: JSON response data (dict) or None if download failed, and time stamp of the request.
    """
    logger.info(f"SoilGrids REST API download from {request['url']} ... ")
    status_codes_rate = {429}  # codes for retry with exponentially increasing delay
    status_codes_gateway = {502, 503, 504}  # codes for retry with fixed time delay

    while attempts > 0:
        attempts -= 1
        time_stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        try:
            response = requests.get(request["url"], params=request["params"])

            if response.status_code == 200:
                return response.json(), time_stamp
            elif response.status_code in status_codes_rate:
                logger.error(f"Request rate limited (Error {response.status_code}).")

                if attempts > 0:
                    logger.info(f"Retrying in {delay_exponential} seconds ...")
                    time.sleep(delay_exponential)
                    delay_exponential *= 2
            elif response.status_code in status_codes_gateway:
                logger.error(f"Request failed (Error {response.status_code}).")

                if attempts > 0:
                    logger.info(f"Retrying in {delay_linear} seconds ...")
                    time.sleep(delay_linear)
            else:
                raise Exception(
                    f"SoilGrids REST API download error: {response.reason} ({response.status_code})."
                )
        except requests.RequestException as e:
            logger.error(f"Request failed {e}.")

            if attempts > 0:
                logger.info(f"Retrying in {delay_linear} seconds ...")
                time.sleep(delay_linear)

    # After exhausting all attempts
    logger.error("Maximum number of attempts reached. Failed to download data.")
    return None, f"DOWNLOAD FAILED! {time_stamp}"


def get_soilgrids_data(soilgrids_data, *, property_names=["silt", "clay", "sand"]):
    """
    Extract property data and units from SoilGrids data.

    Parameters:
        soilgrids_data (dict): SoilGrids data containing property information.
        property_names (list): List of properties to extract data and units for (default is ['silt', 'clay', 'sand']).

    Returns:
        2D numpy.ndarray: Property data for various soil properties and depths (nan if no data found).
    """
    logger.info("Reading SoilGrids data ...")

    # handle case when soilgrids_data is None or empty
    if soilgrids_data is None or not soilgrids_data:
        logger.error(
            "No data found in SoilGrids response. Cannot extract property data."
        )
        return None
    else:
        # Initialize property_data array with zeros
        property_data = np.full(
            (
                len(property_names),
                len(soilgrids_data["properties"]["layers"][0]["depths"]),
            ),
            np.nan,
            dtype=float,
        )

        # Iterate through property_names
        for p_index, p_name in enumerate(property_names):
            # Find the corresponding property in soilgrids_data
            for prop in soilgrids_data["properties"]["layers"]:
                if prop["name"] == p_name:
                    p_units = prop["unit_measure"]["target_units"]

                    # Iterate through depths and fill the property_data array
                    for d_index, depth in enumerate(prop["depths"]):
                        if depth["values"]["mean"]:
                            property_data[p_index, d_index] = (
                                depth["values"]["mean"]
                                / prop["unit_measure"]["d_factor"]
                            )

                        logger.info(
                            f"Depth {depth['label']}, {p_name} "
                            f"mean: {property_data[p_index, d_index]} {p_units}"
                        )
                    break  # Stop searching once the correct property is found

        return property_data


def get_hihydrosoil_map_file(property_name, depth, *, cache=None):
    """
    Generate file path or URL for a HiHydroSoil map based on the provided property name and depth.

    Parameters:
        property_name (str): Name of the soil property (e.g. 'WCpF4.2' or 'Ksat').
        depth (str): Depth layer (one of '0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm').
        cache (Path): Path for local HiHydroSoil map directory (optional).

    Returns:
        pathlib.Path or URL: File path or URL to the HiHydroSoil map.
    """
    file_name = property_name + "_" + depth + "_M_250m.tif"

    if cache is not None:
        map_file = Path(cache) / file_name

        if map_file.is_file():
            return map_file
        else:
            logger.error(f"Local file '{map_file}' not found!")
            logger.info("Trying to access via URL ...")

    map_file = "http://opendap.biodt.eu/grasslands-pdt/soilMapsHiHydroSoil/" + file_name

    if ut.check_url(map_file):
        return map_file
    else:
        logger.error(f"File '{map_file}' not found!")

        return None


def get_hihydrosoil_data(coordinates, *, cache=None):
    """
    Read HiHydroSoil data for the given coordinates and return as array.

    Parameters:
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        cache (Path): Path for local HiHydroSoil map directory (optional).

    Returns:
        tuple:
        - Property data for various soil properties and depths (2D numpy.ndarray, nan if no data found),
        - List of query sources and time stamps.
    """
    logger.info("Reading HiHydroSoil data ...")
    hihydrosoil_depths = [
        "0-5cm",
        "5-15cm",
        "15-30cm",
        "30-60cm",
        "60-100cm",
        "100-200cm",
    ]

    # Initialize property_data array with zeros
    property_data = np.full(
        (len(HIHYDROSOIL_SPECS), len(hihydrosoil_depths)), np.nan, dtype=float
    )

    # Extract values from tif maps for each property and depth
    query_protocol = []

    for p_index, (p_name, p_specs) in enumerate(HIHYDROSOIL_SPECS.items()):
        for d_index, depth in enumerate(hihydrosoil_depths):
            map_file = get_hihydrosoil_map_file(
                p_specs["hihydrosoil_name"], depth, cache=cache
            )

            if map_file:
                # Extract and convert value
                logger.info(f"Reading from file '{map_file}' ...")
                value, time_stamp = ut.extract_raster_value(map_file, coordinates)
                query_protocol.append([map_file, time_stamp])

                if value != -9999:
                    property_data[p_index, d_index] = value * p_specs["map_to_float"]

            logger.info(
                f"Depth {depth}, {p_name}: "
                f"{property_data[p_index, d_index]:.4f} {p_specs['hihydrosoil_unit']}"
            )

    return property_data, query_protocol


def check_property_shapes(property_data, property_names, *, depths_required=6):
    """
    Check the shape of the property data array and raise an error if it does not match the expected dimensions.

    Parameters:
        property_data (numpy.ndarray): Array containing property data.
        property_names (list): List of property names.
        depths_required (int): Number of required depths (default is 6 for SoilGrids,
            use None for not checking layer number).

    Returns:
        None

    Raises:
        ValueError: If the shape of property_data does not match the expected dimensions.
    """
    if property_data.ndim == 1:
        property_count = 1
        layer_count = len(property_data)
    else:
        property_count = property_data.shape[0]
        layer_count = property_data.shape[1]

    if depths_required is not None and layer_count != depths_required:
        try:
            raise ValueError(
                f"Property data layers ({layer_count}) do not match the number of required depths ({depths_required})!"
            )
        except ValueError as e:
            logger.error(e)
            raise

    if property_count != len(property_names):
        try:
            raise ValueError(
                f"Property data shape ({property_count} rows) does not match the number of property names ({len(property_names)})!"
            )
        except ValueError as e:
            logger.error(e)
            raise


def map_depths_soilgrids_grassland_model(
    property_data,
    property_names,
    *,
    conversion_factor=1,
    conversion_units=None,
):
    """
    Map data from SoilGrids depths to grassland model depths.

    Parameters:
        property_data (numpy.ndarray): Array containing property data.
        property_names (list): List of property names.
        conversion_factor (float or array): Conversion factors to apply to the values (default is 1).
        conversion_units (list): List of units after conversion for each property (default is 'None').

    Returns:
        numpy.ndarray: Array containing mapped property values.
    """
    logger.info("Mapping data from SoilGrids depths to grassland model depths ...")

    # Define number of new depths, 0-200cm in 10cm steps
    new_depths_number = 20
    new_depths_step = 10

    # Define SoilGrids depths boundaries
    old_depths = np.array([[0, 5], [5, 15], [15, 30], [30, 60], [60, 100], [100, 200]])

    # Check correct shape of property_data
    check_property_shapes(
        property_data, property_names, depths_required=len(old_depths)
    )

    # Prepare conversion factors and units
    if isinstance(conversion_factor, float):
        conversion_factor = np.full((len(property_names),), conversion_factor)
    else:
        conversion_factor = np.array(conversion_factor)

    if conversion_units is None:
        conversion_units = [""] * len(property_names)

    # Initialize array to store mapped mean values
    if property_data.ndim == 1:
        data_to_map = property_data.copy().reshape(1, -1)
    else:
        data_to_map = property_data

    mapped_data = np.zeros((data_to_map.shape[0], new_depths_number), dtype=float)

    # Iterate over each 10cm interval
    for d_new in range(new_depths_number):
        start_depth = d_new * new_depths_step
        end_depth = (d_new + 1) * new_depths_step

        # Find the indices of SoilGrid depths within the new 10cm interval
        d_indices = np.where(
            (start_depth < old_depths[:, 1]) & (old_depths[:, 0] < end_depth)
        )[0]

        # For each property, calculate the mean of old values (1 or 2 values) for the new 10cm interval
        mapped_data[:, d_new] = (
            np.mean(data_to_map[:, d_indices], axis=1) * conversion_factor
        )
        log_message = f"Depth {start_depth}-{end_depth}cm"

        for p_index in range(len(property_names)):
            log_message += (
                f", {property_names[p_index]}: "
                f"{mapped_data[p_index, d_new]:.4f} {conversion_units[p_index]}"
            )

        logger.info(log_message)

    return mapped_data


def get_property_means(property_data, property_names, *, property_units=None):
    """
    Calculate property data means over all depths (equal weight for each depth).

    Parameters:
        property_data (numpy.ndarray): Array containing property data.
        property_names (list): List of property names.
        property_units (list): List of units for each property (default is 'None').

    Returns:
        numpy.ndarray: Array containing property means.
    """
    # Check correct shape of property_data, any number of layers allowed
    check_property_shapes(property_data, property_names, depths_required=None)

    logger.info("Averaging data over all depths ...")

    if property_data.ndim == 1:
        property_means = [np.mean(property_data)]
    else:
        property_means = np.mean(property_data, axis=1)

    if property_units is None:
        property_units = [""] * len(property_names)

    for p_index in range(len(property_names)):
        logger.info(
            f"Depth 0-200cm, {property_names[p_index]} "
            f"mean: {property_means[p_index]:.4f} {property_units[p_index]}"
        )

    return property_means


def soil_data_to_txt_file(
    coordinates,
    composition_data,
    hihydrosoil_data,
    *,
    data_query_protocol=None,
    file_name=None,
    composition_property_names=["silt", "clay", "sand"],
    # nitrogen_data,
):
    """
    Write SoilGrids and HiHydroSoil data to soil data TXT file in grassland model format.

    Parameters:
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        composition_data (numpy.ndarray): SoilGrids data array (if None, the following highly uncertain
            default values are used: Silt 40%, Clay 20%, Sand 40%).
        hihydrosoil_data (numpy.ndarray): HiHydroSoil data array.
        data_query_protocol (list): List of sources and time stamps from retrieving soil data (default is None).
        file_name (str or Path): File name to save soil data (default is None, default file name is used if not provided).
        composition_property_names (list): List of property names for SoilGrids data (default is ['silt', 'clay', 'sand']).

    Returns:
        None
    """
    # SoilGrids nitrogen part of the data in commits before 2024-09-30
    # SoilGrids composition part for all depths in commits before 2024-09-30

    logger.info("Writing soil compostion data to file ...")

    if composition_data is None:
        logger.error(
            "No data found in SoilGrids response. Assuming highly uncertain default composition values: Silt 40%, Clay 20%, Sand 40%."
        )
        composition_data_mean = np.array([0.4, 0.2, 0.4])
    else:
        # Prepare SoilGrids composition data in grassland model format
        composition_to_grassmodel = 1e-2  # % to proportions for all composition values
        composition_data_grassmodel = map_depths_soilgrids_grassland_model(
            composition_data,
            property_names=composition_property_names,
            conversion_factor=composition_to_grassmodel,
        )

        # Mean over all depths
        composition_data_mean = get_property_means(
            composition_data_grassmodel, composition_property_names
        )

    # Prepare HiHydroSoil data in grassland model format
    logger.info("Writing soil hydraulic properties data to file ...")

    hihydrosoil_property_names = list(HIHYDROSOIL_SPECS.keys())
    hihydrosoil_conversion_factor = [
        specs["hihydrosoil_to_grassmodel"] for specs in HIHYDROSOIL_SPECS.values()
    ]
    hihydrosoil_units_grassmodel = [
        specs["grassmodel_unit"] for specs in HIHYDROSOIL_SPECS.values()
    ]
    hihydrosoil_data_grassmodel = map_depths_soilgrids_grassland_model(
        hihydrosoil_data,
        property_names=hihydrosoil_property_names,
        conversion_factor=hihydrosoil_conversion_factor,
        conversion_units=hihydrosoil_units_grassmodel,
    )

    # Write collected soil data to TXT file
    if not file_name:
        file_name = construct_soil_data_file_name(
            coordinates, folder="soilDataPrepared"
        )

    # Create data directory if missing
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    # SoilGrids composition part
    composition_data_to_write = shape_soildata_for_file(composition_data_mean)
    composition_header = "\t".join(
        list(map(str.capitalize, composition_property_names))
    )
    np.savetxt(
        file_name,
        composition_data_to_write,
        delimiter="\t",
        fmt="%.4f",
        header=composition_header,
        comments="",
    )

    # HiHydroSoil part
    hihydrosoil_data_to_write = shape_soildata_for_file(hihydrosoil_data_grassmodel)
    grassmodel_depth_count = np.arange(1, 21).reshape(-1, 1)
    hihydrosoil_data_to_write = np.concatenate(
        (grassmodel_depth_count, hihydrosoil_data_to_write),
        axis=1,
    )
    grassmodel_names = [
        specs["grassmodel_name"] for specs in HIHYDROSOIL_SPECS.values()
    ]
    hihydrosoil_header = "\t".join(map(str, ["Layer"] + grassmodel_names))

    with open(file_name, "a", encoding="utf-8", errors="replace") as fh:
        fh.write("\n")
        np.savetxt(
            fh,
            hihydrosoil_data_to_write,
            delimiter="\t",
            fmt="%.4f",
            header=hihydrosoil_header,
            comments="",
        )

    logger.info(
        f"Processed soil data from SoilGrids and HiHydroSoil written to file '{file_name}'."
    )

    if data_query_protocol:
        file_name = file_name.with_name(
            file_name.stem + "__data_query_protocol" + file_name.suffix
        )
        ut.list_to_file(
            data_query_protocol,
            file_name,  # column_names=["data_source", "time_stamp"]
        )
