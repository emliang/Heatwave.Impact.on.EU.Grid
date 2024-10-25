# Heatwave Impact on European Electricity Grids

## Overview
This code repository contains the implementation and analysis for the study of heatwave effects on European electricity grids. As climate change increases the frequency, intensity, and duration of heatwaves, it is crucial to understand their impact on electricity grids to enhance societal security and resilience.

## Objective
The main objectives of this code are:
1. Investigate the effects of heatwaves on European electricity grids using comprehensive real-world datasets.
2. Develop a novel temperature-dependent segment-based approach for modeling temperature effects on grid operation limits, which is both efficient and accurate.
3. Evaluate the robustness of several European electricity grids for projected heatwave scenarios for the next 25 years.
4. Identify grid bottlenecks and national differences in grid resilience during heatwaves.

## Datasets
The table lists the key data sources including PyPSA-Eur for the European transmission network model, ERA5 and C3S climate reanalysis data, ENTSO-E historical power demand data, future demand projections from Demand.ninja, and the open-source Atlite library for calculating renewable generation potentials and time series.

<div style="overflow-x: auto;">
<table>
<thead>
<tr>
<th>Data</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="https://pypsa-eur.readthedocs.io/">PyPSA-Eur</a></td>
<td>Open-source dataset of European transmission network</td>
</tr>
<tr>
<td><a href="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview">ERA5</a></td>
<td>Historical hourly global climate data from ECMWF reanalysis</td>
</tr>
<tr>
<td><a href="https://cds.climate.copernicus.eu/datasets/sis-energy-derived-projections?tab=overview">C3S</a></td>
<td>Future reference climate data from C3S Energy operational service</td>
</tr>
<tr>
<td><a href="https://www.entsoe.eu/data/power-stats/">ENTSO-E</a></td>
<td>Historical hourly country-level power demand data</td>
</tr>
<tr>
<td><a href="https://demand.ninja/">Demand.ninja</a></td>
<td>Future weather-dependent energy demand models</td>
</tr>
<tr>
<td><a href="https://github.com/PyPSA/atlite">Atlite</a> </td>
<td>Open-source model for renewable generation calculation</td>
</tr>
</tbody>
</table>
</div>

## Methodology
<ul>
<li><strong>Future Heatwave Projection:</strong> It generates projected heatwave events from 2025 to 2050 based on historical heatwaves from 2019 and 2022.</li>
<li><strong>Future Demand Modeling:</strong> It utilizes <a href="https://demand.ninja/">Demand.ninja</a> [1] to model power demand considering the weather condition and annual growth.</li>
<li><strong>Renewable Generation:</strong> It leverages existing frameworks from <a href="https://github.com/PyPSA/atlite">Atlite</a> [2] to calculate renewable generation given weather conditions.</li>
<li><strong>Conductor Thermal Modeling:</strong> It analyzes the impact of elevated temperatures on conductor physical properties and their effects on thermal limits.</li>
<li><strong>Multi-Segment Modeling:</strong> It models the transmission line at segmented levels to identify localized stress points and potential bottlenecks.</li>
<li><strong>Optimal Power Flow Analysis:</strong> It integrates these components to simulate the grid's response under thermal and demand stresses, revealing potential capacity bottlenecks and load shedding regions.</li>

</ul>



## Demos
The key findings of this study include:
- The Spanish grid exhibits temperature-induced capacity bottlenecks that could jeopardize power supply during heatwaves.
- The German grid shows remarkable resilience to heatwaves.
- The study emphasizes the need for temperature-aware grid power flow analysis and long-range planning to ensure energy security in the face of climate-change induced future heatwaves.

