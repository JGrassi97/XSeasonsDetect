#!/bin/bash

# File input
OBSERVED="observed_data.nc"
MODEL="model_data.nc"

# File output
OBS_QUANTILES="observed_quantiles.nc"
MODEL_QUANTILES="model_quantiles.nc"
CORRECTED_MODEL="corrected_model_data.nc"

# Calcola i quantili dai dati osservati
cdo ydrunpctl,10,1,1 $OBSERVED $OBS_QUANTILES

# Calcola i quantili dai dati del modello
cdo ydrunpctl,10,1,1 $MODEL $MODEL_QUANTILES

# Applica il Quantile Mapping
cdo -L -remapdis,$OBS_QUANTILES $MODEL $CORRECTED_MODEL

echo "Quantile mapping completed. Corrected data saved in $CORRECTED_MODEL"
