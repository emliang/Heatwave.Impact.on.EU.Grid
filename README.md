# Heatwave Impact on European Electricity Grids

## Overview
This code repository contains the implementation and analysis for the study of heatwave effects on European electricity grids. As climate change increases the frequency, intensity, and duration of heatwaves, it is crucial to understand their impact on electricity grids to enhance societal security and resilience.

## Package Requirements
The code requires the following Python packages:
- NumPy
- SciPy
- Pandas
- Matplotlib
- PyPSA, PyPSA-Eur
- Pyomo
- PyPower
- Atlite
- IPOPT

## Datasets Requirements
The following table lists the key data sources used in this study:

| Data | Description |
|------|-------------|
| [PyPSA-Eur](https://pypsa-eur.readthedocs.io/) | Open-source dataset of European transmission network |
| [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview) | Historical hourly global climate data from ECMWF reanalysis |
| [C3S](https://cds.climate.copernicus.eu/datasets/sis-energy-derived-projections?tab=overview) | Future reference climate data from C3S Energy operational service |
| [ENTSO-E](https://www.entsoe.eu/data/power-stats/) | Historical hourly country-level power demand data |
| [Demand.ninja](https://demand.ninja/) | Future weather-dependent energy demand models |
| [Atlite](https://github.com/PyPSA/atlite) | Open-source model for renewable generation calculation |

## Methodology
The methodology employed in this study includes the following key components:

- **Future Heatwave Projection:** It generates projected heatwave events from 2025 to 2050 based on historical heatwaves from 2019 and 2022.
- **Future Demand Modeling:** It utilizes [Demand.ninja](https://demand.ninja/) to model power demand considering the weather condition and annual growth.
- **Renewable Generation:** It leverages existing frameworks from [Atlite](https://github.com/PyPSA/atlite) to calculate renewable generation given weather conditions.
- **Conductor Thermal Modeling:** It analyzes the impact of elevated temperatures on conductor physical properties and their effects on thermal limits.
- **Multi-Segment Modeling:** It models the transmission line at segmented levels to identify localized stress points and potential bottlenecks.
- **Optimal Power Flow Analysis:** It integrates these components to simulate the grid's response under thermal and demand stresses, revealing potential capacity bottlenecks and load shedding regions.

## Demos
The code contains the following demos for the Methodology:
- Heatwave generation based on morphing approach
- Demand calibration based on demand.ninja
- Heat balance equation calculation and visualization
- ACOPF/TD-ACOPF-quad/TD-ACOPF-iter solving by IPOPF + Pyomo
- A country-level analysis based on the Datasets 



