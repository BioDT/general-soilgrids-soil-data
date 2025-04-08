"""
Module Name: data_processing.py
Description: Function for obtaining selected soil data at given location from
             SoilGrids and derived data sources (SoilGrids REST API, HiHydroSoil maps).

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

from soilgrids import get_soil_data as gsd
from soilgrids.logger_config import logger


def data_processing(coordinates, *, file_name=None, hihydrosoil_cache=None):
    """
    Download data from SoilGrids and HiHydroSoil maps. Convert to .txt files.

    Parameters:
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        file_name (str or Path): File name to save soil data (default is None, default file name is used if not provided).
        hihydrosoil_cache (Path): Path for local HiHydroSoil map directory (optional).
    """
    # SoilGrids nitrogen part of the data in commits before 2024-09-30

    if "lat" in coordinates and "lon" in coordinates:
        logger.info(
            f"Preparing soil data for latitude: {coordinates['lat']}, longitude: {coordinates['lon']} ..."
        )
    else:
        try:
            raise ValueError(
                "Coordinates not correctly defined. Please provide as dictionary ({'lat': float, 'lon': float})!"
            )
        except ValueError as e:
            logger.error(e)
            raise

    # SoilGrids composition part of the data
    composition_request = gsd.configure_soilgrids_request(coordinates)
    composition_raw, time_stamp = gsd.download_soilgrids(composition_request)
    data_query_protocol = [[composition_request["url"], time_stamp]]
    composition_data = gsd.get_soilgrids_data(composition_raw)

    # HiHydroSoil part of the data
    hihydrosoil_data, hihydrosoil_queries = gsd.get_hihydrosoil_data(
        coordinates, cache=hihydrosoil_cache
    )
    data_query_protocol.extend(hihydrosoil_queries)

    gsd.soil_data_to_txt_file(
        coordinates,
        composition_data,
        hihydrosoil_data,
        data_query_protocol=data_query_protocol,
        file_name=file_name,
    )
