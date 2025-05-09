"""
Module Name: logger_config.py
Description: Logging configuration for soilgrids building block.

Developed in the BioDT project by Thomas Banitz (UFZ) with contributions by Franziska Taubert (UFZ)
and Tuomas Rossi (CSC).

Copyright (C) 2025
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

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Define log file path
log_file = Path("logs") / "soilgrids_app.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

# Create a logger specific to this package
logger = logging.getLogger("soilgrids")
logger.setLevel(logging.INFO)

# Create handlers
file_handler = TimedRotatingFileHandler(
    log_file, when="midnight", interval=1, backupCount=7
)  # rotate daily, keep 7 backups
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
