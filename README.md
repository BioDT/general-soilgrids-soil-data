# BioDT - Soilgrids Soil Data
Building block for downloading and processing selected soil data at given location from
   SoilGrids and derived data sources (Soilgrids REST API, HiHydroSoil maps).

## Installation
The current development version can be installed as:

    pip install git+https://github.com/BioDT/general-soilgrids-soil-data.git@main
    
## Usage
Request data for a location and produce .txt files in grassland model input data format:

```python
from soilgrids import data_processing

coordinates = {"lat": 51.123456, "lon": 11.987654}
data_processing(coordinates)
```

Full function signature: 

`data_processing(coordinates, *, file_name=None, hihydrosoil_cache=None)` 

Parameters:
- coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
- file_name (str or Path): File name to save soil data (default is None, default file name is used if not provided).
- hihydrosoil_cache (Path): Path for local HiHydroSoil map directory (default is None, online data is used if not provided).

## Developers
Developed in the BioDT project by Thomas Banitz (UFZ) with contributions by Franziska Taubert (UFZ), 
Tuomas Rossi (CSC) and Taimur Haider Khan (UFZ).

## Copyright
Copyright (C) 2024
- Helmholtz Centre for Environmental Research GmbH - UFZ, Germany
- CSC - IT Center for Science Ltd., Finland

Licensed under the EUPL, Version 1.2 or - as soon they will be approved
by the European Commission - subsequent versions of the EUPL (the "Licence").
You may not use this work except in compliance with the Licence.

You may obtain a copy of the Licence at:
https://joinup.ec.europa.eu/software/page/eupl

## Funding
This project has received funding from the European Union's Horizon Europe Research and Innovation
Programme under grant agreement No 101057437 (BioDT project, https://doi.org/10.3030/101057437).
The authors acknowledge the EuroHPC Joint Undertaking and CSC - IT Center for Science Ltd., Finland
for awarding this project access to the EuroHPC supercomputer LUMI, hosted by CSC - IT Center for
Science Ltd., Finland and the LUMI consortium through a EuroHPC Development Access call.

## Data sources
SoilGrids&#x2122; 2.0:
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