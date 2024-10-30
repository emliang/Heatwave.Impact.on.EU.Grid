from utils.data_config import *
from utils.network_process_utils import *
from utils.demand_calibration import *
from utils.opf_pyomo_utils import *

import os
import re
import time
import yaml
import pypsa
import atlite
import requests
import calendar
import functools
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xml.etree.ElementTree as ET

from pyproj import CRS
from yaml import Loader
from datetime import datetime
from pypsa.geo import haversine
from netCDF4 import Dataset, num2date, date2num
from entsoe.mappings import Area, NEIGHBOURS, lookup_area


EXTERNAL = "/Volumes/T9"
PYPSA = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/pypsa-eur'
RESOURCES = PYPSA + "/resources/"
DATA = PYPSA +  "/data/"
CUTOUT = PYPSA + "/cutouts/"


def weighted_average_weather(network, ncfile):
    elevation = xr.open_dataset(EXTERNAL + '/elevation.nc')
    """
    bus idx: [tem, wind, solar, angle]
    """
    bus_loc_list = network.buses[['x','y']].values
    x_da = xr.DataArray(bus_loc_list[:,0], dims=["bus"])
    y_da = xr.DataArray(bus_loc_list[:,1], dims=["bus"])
    bus_ncfile_slice = ncfile.sel(x= x_da, y= y_da, method="nearest")
    bus_ncfile_height = elevation.sel(lon = x_da, lat = y_da, method='nearest')
    network.buses['temperature'] = bus_ncfile_slice['temperature'].data.reshape(-1) - 273.15
    network.buses['wnd10m'] = bus_ncfile_slice['wnd10m'].data.reshape(-1) 
    network.buses['influx_toa'] = bus_ncfile_slice['influx_toa'].data.reshape(-1) 
    network.buses['wnd_azimuth'] = bus_ncfile_slice['wnd_azimuth'].data.reshape(-1) 
    network.buses['height'] = np.maximum(bus_ncfile_height.z.data.reshape(-1), 0)

    """
    segment_info: max_segment
    branch idx: [ [x,y,len], ..., [] ]
    """
    segment_info = network.segments
    nline = segment_info.shape[0]
    nseg = segment_info.shape[1]
    seg_location_list = segment_info[:,:,0:2] # branch * segments * (x, y)
    seg_mask = np.sign(segment_info[:, :, 2])
    seg_location_list = segment_info[:,:,0:2].reshape(-1,2)
    x_da = xr.DataArray(seg_location_list[:,0], dims=["seg"])
    y_da = xr.DataArray(seg_location_list[:,1], dims=["seg"])
    seg_ncfile_slice = ncfile.sel(x= x_da, y= y_da, method="nearest")
    seg_ncfile_height = elevation.sel(lon = x_da, lat = y_da, method='nearest')
    segment_wea = np.zeros(shape=[segment_info.shape[0], segment_info.shape[1], 5])
    segment_wea[:, :, 0] = (seg_ncfile_slice['temperature'].data.reshape(nline, nseg) - 273.15) * seg_mask 
    segment_wea[:, :, 1] = seg_ncfile_slice['wnd10m'].data.reshape(nline, nseg) * seg_mask
    segment_wea[:, :, 2] = seg_ncfile_slice['influx_toa'].data.reshape(nline, nseg) * seg_mask
    segment_wea[:, :, 3] = seg_ncfile_slice['wnd_azimuth'].data.reshape(nline, nseg) * seg_mask
    segment_wea[:, :, 4] = np.maximum(seg_ncfile_height.z.data.reshape(nline, nseg), 0) * seg_mask

    """
    Branch weather info
    """
    branch_wea = np.zeros(shape=[network.lines.shape[0], 4])
    seg_prop = network.segments[:, :, 2]
    branch_wea = (segment_wea * seg_prop.reshape(*seg_prop.shape[0:2],1)).sum(1)
    network.lines[['temperature', 'wnd10m', 'influx_toa', 'wnd_azimuth', 'height']] = branch_wea
    network.segment_info = segment_wea
    return network

def load_demand_online_api(date, country_code):
    # Define the API endpoint
    url = "https://web-api.tp.entsoe.eu/api"
    # Set the parameters for the GET request
    area_code = lookup_area(country_code).code
    start = f'{date.year}0{date.month}{date.day}{date.hour}00'
    end = f'{date.year}0{date.month}{date.day}{date.hour+1}00'
    params = {
        'securityToken': 'f796600c-b4a4-4551-b04d-1e3bc5902a0d',
        'documentType': 'A65',
        'processType': 'A16',
        'outBiddingZone_Domain': area_code,
        'periodStart': start,
        'periodEnd': end,
    }
    country_load_t = 0
    response = requests.get(url, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        xml_data = response.text
        root = ET.fromstring(xml_data)
        # Extracting elements from TimeSeries
        for ts in root.findall('.//{urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0}TimeSeries'):
            for point in ts.findall('.//{urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0}Point'):
                quantity = point.find('.//{urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0}quantity').text
                country_load_t = int(quantity)

def load_historical_load_info(date, code):
    if date.year <= 2020:
        demand_filename = './climate_data/entsoe/load_2015-2020.csv'
        load_demand = pd.read_csv(demand_filename, index_col=False)
        if code == 'GB':
            country_load = load_demand[f'{code}_GBN_load_actual_entsoe_transparency'].values + \
                        load_demand[f'{code}_NIR_load_actual_entsoe_transparency'].values
        else:   
            country_load = load_demand[f'{code}_load_actual_entsoe_transparency'].values
        # demand_filename = f'./climate_data/entsoe/MHLV_{2018}-{2019}.xlsx'
        # load_demand = pd.read_excel(demand_filename, index_col=False) 
        # country_load = load_demand[load_demand['CountryCode'] == code]['Value']
            
        # Convert to ISO 8601 format with 'Z' to denote UTC time
        timestamp = date.isoformat() + 'Z'
        country_load_t = country_load[load_demand['utc_timestamp'] == timestamp]
    else:
        demand_filename = f'./climate_data/entsoe/MHLV_{date.year}.csv'
        load_demand = pd.read_csv(demand_filename, index_col=False) 
        load_demand = load_demand[load_demand['CountryCode'] == code]
        country_load = load_demand['Value']
        date_column = pd.to_datetime(load_demand['DateUTC'], dayfirst=True)
        country_load_t = country_load[date_column == date].values[0]
    return country_load_t

def load_future_load_info(network, code, utc_timestamp, args):
    x_min, x_max = network.buses['x'].min() - 1,  network.buses['x'].max() + 1
    y_min, y_max = network.buses['y'].min() - 1,  network.buses['y'].max() + 1
    """Aggregate into daily weather data"""
    if utc_timestamp.day <3:
        last_monthdays =  calendar.monthrange(utc_timestamp.year, utc_timestamp.month-1)[1]
        date_s = datetime(utc_timestamp.year, utc_timestamp.month-1, last_monthdays+utc_timestamp.day-2, 0, 0, 0)
    else:
        date_s = datetime(utc_timestamp.year, utc_timestamp.month, utc_timestamp.day-2, 0, 0, 0)
    date_e = datetime(utc_timestamp.year, utc_timestamp.month, utc_timestamp.day, 23, 0, 0)
    ncfile = xr.open_dataset(EXTERNAL + f"/rcp45/Future_rcp45_July_data.nc")
    ncfile_slice = ncfile.sel(time=slice(date_s, date_e),x=slice(x_min, x_max ), y=slice(y_min, y_max ))

    ## daily average
    bus_loc_list = network.buses[['x', 'y']].values
    nbus = len(bus_loc_list)
    x_da = xr.DataArray(bus_loc_list[:,0], dims=["bus"])
    y_da = xr.DataArray(bus_loc_list[:,1], dims=["bus"])
    bus_ncfile_slice = ncfile_slice.sel(x= x_da, y= y_da, method="nearest")
    temp_data = bus_ncfile_slice['temperature'].data.reshape((-1,8,nbus)).mean(1) - 273.15 
    if args['heatwave_mode']:
        hot_index = network.hot_index
        delta_temp_ncfile = xr.open_dataset(EXTERNAL + f"/era5/Historical_delta_heatwave_July_data.nc")
        delta_temp_slice = delta_temp_ncfile.sel(time=slice(datetime(hot_index.year, hot_index.month, hot_index.day, 0), 
                                                            datetime(hot_index.year, hot_index.month, hot_index.day, 23)),
                                                 x=slice(x_min, x_max ), y=slice(y_min, y_max ))
        bus_ncfile_delta_slice = delta_temp_slice.sel(x= x_da, y= y_da, method="nearest")
        temp_data = temp_data + bus_ncfile_delta_slice['delta_temp'].data.mean(0).reshape(1,-1)
    else:
        if args['sensitivity_analysis_mode'] in [1,3]:
            temp_data = temp_data + args['artificial_delta_temp']
    wind_data = bus_ncfile_slice['wnd10m'].data.reshape((-1,8,nbus)).mean(1)
    solar_data = bus_ncfile_slice['influx_toa'].data.reshape((-1,8,nbus)).mean(1)

    demand_curve = np.load(f'code/models/demand_curve/{code}_demand_curve.npy' , allow_pickle=True).item()
    pop_ratio = network.buses['pop_ratio'].values.reshape(1,-1)
    bait = _bait(temp_data, wind_data, solar_data, demand_curve)[-1]
    bait = (bait * pop_ratio).sum()

    base_demand = demand_curve['base_demand'] + demand_curve['workday_demand']
    heating_threshold, cooling_threshold, heating_coef, cooling_coef = \
    demand_curve['heating_threshold'], demand_curve['cooling_threshold'], demand_curve['p_heating'], demand_curve['p_cooling']
    hour_loads_ratio = demand_curve['weekday_cooling_hour_ratio'][0]
    increase_temp = np.maximum(0, bait - cooling_threshold)
    decrease_temp = np.maximum(0, heating_threshold - bait)

    #  European power demand is expected to grow by 40 percent from 2020 to 2050, increasing from 3,500 TWh to 4,900 TWh.
    if args['load_growth']:
        base_demand = base_demand * (1.011) ** (utc_timestamp.year - 2020)
    total_demand = base_demand \
                   + cooling_coef * increase_temp  \
                   + heating_coef * decrease_temp
    total_demand = total_demand * 1e3 * 24

    # time_offset = get_timezone_difference(selected_country, utc_timestamp)
    # local_hour = utc_timestamp.hour + time_offset
    hour_demand = total_demand * hour_loads_ratio[utc_timestamp.hour]
    return hour_demand

def network_load_weather(network, date, selected_country, args):
    heatwave = args['heatwave_mode']
    heatwave_year = args['heatwave_year']
    if date.year >= 2023:
        ncfile = xr.open_dataset(EXTERNAL + f"/rcp45/Future_rcp45_July_data.nc")
        ncfile_slice = ncfile.sel(time =slice(datetime(date.year, date.month, date.day, date.hour//3*3),
                                            datetime(date.year, date.month, date.day, date.hour//3*3+3)))
        if heatwave:
            delta_temp_ncfile = xr.open_dataset(EXTERNAL + f"/era5/Historical_delta_heatwave_July_data.nc")
            network = network_load_heatwave(network, heatwave_year)
            hot_index = network.hot_index
            delta_temp_slice = delta_temp_ncfile.sel(
                time=datetime(hot_index.year, hot_index.month, hot_index.day, date.hour), method='nearest')
            delta_temp = delta_temp_slice['delta_temp'].data
            ncfile_slice['temperature'] =  ncfile_slice['temperature'] + np.expand_dims(delta_temp, 0)
        else:
            ncfile_slice['temperature'] =  ncfile_slice['temperature'] + args['artificial_delta_temp']
    else:
        ncfile = xr.open_dataset(EXTERNAL + f"/era5/Historical_era5_July_data.nc")
        ncfile_slice = ncfile.sel(time =slice(date, date))
        
    network = weighted_average_weather(network, ncfile_slice)
    weather_path = f'code/models/{selected_country}/weather'
    os.makedirs(weather_path, exist_ok=True)
    ncfile_slice.to_netcdf(weather_path + f'/{date}_{heatwave}_{heatwave_year}_weather_slice.nc', mode='w')
    return network

def network_load_heatwave(network, year=2022):
    his_era5_ncfile = xr.open_dataset(EXTERNAL + f"/era5/Historical_era5_July_data.nc")
    historical_heatwave = his_era5_ncfile.sel(time=slice(datetime(year,6,1),datetime(year,9,1)))
    """
    bus idx: [tem, wind, solar, angle]
    """
    bus_loc_list = network.buses[['x', 'y']].values
    x_da = xr.DataArray(bus_loc_list[:,0], dims=["bus"])
    y_da = xr.DataArray(bus_loc_list[:,1], dims=["bus"])
    historical_heatwave = historical_heatwave.sel(x = x_da, y = y_da, method='nearest')
    hot_index = historical_heatwave['temperature'].data.mean(1).argmax()
    network.hot_index = historical_heatwave.time.data[hot_index].astype('datetime64[D]').astype('O')
    return network

def network_load_demand(network, date, selected_country, args):
    """
    load data to ppc
    """
    utc_timestamp = date
    ### load demand and temperature in the date:
    if utc_timestamp.year<2023:
        try:
            country_load_t = load_historical_load_info(utc_timestamp, selected_country)
        except:
            country_load_t = None
            print('No data at this historical date')
    else:
        country_load_t = load_future_load_info(network, selected_country, utc_timestamp, args)
    print(f'enery demand {country_load_t} at {date}')
    if country_load_t is not None:
        network.buses['p_set'] = country_load_t * network.buses['pop_ratio'] 
        network.buses['q_set'] = network.buses['p_set'] * args['reactive_demand_ratio']
    else:
        network.buses['p_set'] = 0
        network.buses['q_set'] = 0
    return network

def network_load_renewable(network, date, selected_country, args, ratio=100):
    weather_path = f'code/models/{selected_country}/weather'
    heatwave = args['heatwave_mode']
    heatwave_year = args['heatwave_year']
    renewable_carriers = ['solar', 'onwind', 'offwind-ac', 'offwind-dc'] #'hydro'
    with open(PYPSA + "/config/config.yaml", "r") as f:
        config = yaml.load(f, Loader)
    for technology in renewable_carriers:
        params = config['renewable'][technology]
        correction_factor = params.get("correction_factor", 1.0)
        capacity_per_sqkm = params["capacity_per_sqkm"]
        p_nom_max_meth = params.get("potential", "conservative")
        if ratio == 100:
            regions_path = RESOURCES +  f"{selected_country}/regions_onshore_elec_s.geojson" \
                            if technology in ("onwind", "solar")  \
                            else RESOURCES + f"{selected_country}/regions_offshore_elec_s.geojson" 
        else:
            regions_path = RESOURCES +  f"{selected_country}/regions_onshore_elec_s_{ratio}.geojson" \
                            if technology in ("onwind", "solar")  \
                            else RESOURCES + f"{selected_country}/regions_offshore_elec_s_{ratio}.geojson" 
        corine_path = DATA + "bundle/corine/g250_clc06_V18_5.tif"
        shipdensity_path = DATA + "shipdensity_raster.tif"
        gebco_path = DATA + "bundle/GEBCO_2014_2D.nc"
        country_shapes= RESOURCES + f"{selected_country}/country_shapes.geojson"
        offshore_shapes= RESOURCES + f"{selected_country}/offshore_shapes.geojson"
        avai_path = weather_path + f'/{date.year<2024}_{technology}_avai_{ratio}.nc'
        profile_path = weather_path + f'/{date}_{heatwave}_{heatwave_year}_{technology}_profile_{ratio}.nc'
        regions = gpd.read_file(regions_path)
        regions = regions.set_index("name").rename_axis("bus")
        buses = regions.index

        ncfile_slice = xr.open_dataset(weather_path + f'/{date}_{heatwave}_{heatwave_year}_weather_slice.nc')
        cutout_path = weather_path + f'/{date}_{technology}_cutout.nc'
        cutout = atlite.Cutout(cutout_path, data = ncfile_slice)

        if os.path.exists(avai_path):
            availability = xr.open_dataarray(avai_path)
        else:
            res = params.get("excluder_resolution", 100)
            excluder = atlite.ExclusionContainer(crs=3035, res=res)
            if params["natura"]:
                natura_path = DATA + "natura.tiff"
                excluder.add_raster(natura_path, nodata=0, allow_no_overlap=True)
            corine = params.get("corine", {})
            if "grid_codes" in corine:
                codes = corine["grid_codes"]
                excluder.add_raster(corine_path, codes=codes, invert=True, crs=3035)
            if "distance" in corine and corine["distance"] > 0.0:
                codes = corine["distance_grid_codes"]
                buffer = corine["distance"]
                excluder.add_raster(corine_path, codes=codes, buffer=buffer, crs=3035)
            if "ship_threshold" in params:
                shipping_threshold = (
                    params["ship_threshold"] * 8760 * 6
                )  # approximation because 6 years of data which is hourly collected
                func = functools.partial(np.less, shipping_threshold)
                excluder.add_raster(
                    shipdensity_path, codes=func, crs=4326, allow_no_overlap=True
                )
            if params.get("max_depth"):
                # lambda not supported for atlite + multiprocessing
                # use named function np.greater with partially frozen argument instead
                # and exclude areas where: -max_depth > grid cell depth
                func = functools.partial(np.greater, -params["max_depth"])
                excluder.add_raster(gebco_path, codes=func, crs=4326, nodata=-1000)
            if "min_shore_distance" in params:
                buffer = params["min_shore_distance"]
                excluder.add_geometry(country_shapes, buffer=buffer)
            if "max_shore_distance" in params:
                buffer = params["max_shore_distance"]
                excluder.add_geometry(offshore_shapes, buffer=buffer, invert=True)
            kwargs = dict(nprocesses=20)
            availability = cutout.availabilitymatrix(regions, excluder, **kwargs)
            availability.to_netcdf(avai_path)



        resource = params["resource"] 
        func = getattr(cutout, resource.pop("method"))
        area = cutout.grid.to_crs(3035).area / 1e6
        area = xr.DataArray(
            area.values.reshape(cutout.shape), [cutout.coords["y"], cutout.coords["x"]]
        )
        potential = capacity_per_sqkm * availability.sum("bus") * area
        capacity_factor = correction_factor * func(capacity_factor=True, **resource)
        layout = capacity_factor * area * capacity_per_sqkm

        profile, capacities = func(
        matrix=availability.stack(spatial=["y", "x"]),
        layout=layout,
        index=buses,
        per_unit=True,
        return_capacity=True,
        **resource,
        )
        if p_nom_max_meth == "simple":
            p_nom_max = capacity_per_sqkm * availability @ area
        elif p_nom_max_meth == "conservative":
            max_cap_factor = capacity_factor.where(availability != 0).max(["x", "y"])
            p_nom_max = capacities / max_cap_factor

        # layoutmatrix = (layout * availability).stack(spatial=["y", "x"])
        # coords = cutout.grid[["x", "y"]]
        # bus_coords = regions[["x", "y"]]
        # average_distance = []
        # centre_of_mass = []
        # for bus in buses:
        #     row = layoutmatrix.sel(bus=bus).data
        #     nz_b = row != 0
        #     row = row[nz_b]
        #     co = coords[nz_b]
        #     distances = haversine(bus_coords.loc[bus], co)
        #     average_distance.append((distances * (row / row.sum())).sum())
        #     centre_of_mass.append(co.values.T @ (row / row.sum()))

        ds = xr.merge(
            [
                (correction_factor * profile).rename("profile"),
                capacities.rename("weight"),
                p_nom_max.rename("p_nom_max"),
                # potential.rename("potential"),
                # average_distance.rename("average_distance"),
            ]
        )

        # select only buses with some capacity and minimal capacity factor
        ds = ds.sel(
            bus=(
                (ds["profile"].mean("time") > params.get("min_p_max_pu", 0.0))
                & (ds["p_nom_max"] > params.get("min_p_nom_max", 0.0))
            )
        )

        if "clip_p_max_pu" in params:
            min_p_max_pu = params["clip_p_max_pu"]
            ds["profile"] = ds["profile"].where(ds["profile"] >= min_p_max_pu, 0)

        ds.to_netcdf(profile_path)

    """load renewable profile"""
    heatwave = args['heatwave_mode']
    solar_profile_path = weather_path + f'/{date}_{heatwave}_{heatwave_year}_solar_profile_{ratio}.nc'
    solar_profile = xr.open_dataset(solar_profile_path)
    onwind_profile_path = weather_path + f'/{date}_{heatwave}_{heatwave_year}_onwind_profile_{ratio}.nc'
    onwind_profile = xr.open_dataset(onwind_profile_path)
    offwind_ac_profile_path = weather_path + f'/{date}_{heatwave}_{heatwave_year}_offwind-ac_profile_{ratio}.nc'
    offwind_ac_profile = xr.open_dataset(offwind_ac_profile_path)
    offwind_dc_profile_path = weather_path + f'/{date}_{heatwave}_{heatwave_year}_offwind-dc_profile_{ratio}.nc'
    offwind_dc_profile = xr.open_dataset(offwind_dc_profile_path)
    for gen_id in network.generators.index:
        gen_item = network.generators.loc[gen_id]
        bus_id = gen_item['bus']
        tech = gen_item['carrier']
        if tech == 'solar' and bus_id in solar_profile.bus:
            network.generators.loc[gen_id, 'p_max_pu'] = solar_profile.sel(bus = bus_id).profile.data
        elif tech == 'onwind' and bus_id in onwind_profile.bus:
            network.generators.loc[gen_id, 'p_max_pu'] = onwind_profile.sel(bus = bus_id).profile.data
        elif tech == 'offwind-ac' and bus_id in offwind_ac_profile.bus:
            network.generators.loc[gen_id, 'p_max_pu'] = offwind_ac_profile.sel(bus = bus_id).profile.data
        elif tech == 'offwind-dc' and bus_id in offwind_dc_profile.bus:
            network.generators.loc[gen_id, 'p_max_pu'] = offwind_dc_profile.sel(bus = bus_id).profile.data
        elif tech in network.generators_t['p_max_pu'].columns:
            date_tech = datetime(2013, date.month, date.day, date.hour)
            network.generators.loc[gen_id, 'p_max_pu'] =  network.generators_t['p_max_pu'].loc[date_tech, gen_id] 
        else:
            pass
    return network

def cal_max_seg_temp(network, selected_country, conductor, weather, utc_timestamp, args, ratio):
    network = network_load_weather(network, utc_timestamp, selected_country,  args)
    network = network_load_demand(network, utc_timestamp, selected_country, args)
    network = network_load_renewable(network, utc_timestamp, selected_country, args, ratio)
    ppc = pypsa_pypower(network, args)
    """
    running TD ACOPF case
    """
    ppc_expr = copy.deepcopy(ppc)
    segment_wea = copy.deepcopy(network.segment_info)
    xe = network.buses.loc[network.lines['bus1'], ['x','y']].values - \
         network.buses.loc[network.lines['bus0'], ['x','y']].values
    conductor_angle = (90 - np.rad2deg(np.arctan2(xe[:, 1], xe[:, 0]))) % 360
    conductor['conductor_angle'] = np.reshape(conductor_angle, (-1,1))
    base_results= None
    if os.path.exists(f'code/models/critical_date/{selected_country}_critical_base_results.npy'):
        base_results = np.load(f'code/models/critical_date/{selected_country}_critical_base_results.npy',allow_pickle=True)
    result = pyomo_solve_ac(ppc, initial_value=base_results, solver_name='ipopt')

    weather_s = copy.deepcopy(weather)
    conductor_s = copy.deepcopy(conductor)
    weather_s['air_temperature'] = segment_wea[:, :, 0]
    weather_s['wind_speed'] = segment_wea[:, :, 1]
    weather_s['solar_heat_intensity'] = segment_wea[:, :, 2]
    conductor_s['elevation'] = segment_wea[:, :, 4]
    BaseI = ppc['baseI']
    num_parallel = network.lines['num_parallel'].values
    num_parallel[num_parallel==0] = 1
    Branch_status = np.sign(ppc_expr['branch'][:, idx_brch.BR_STATUS] * ppc_expr['branch'][:, idx_brch.RATE_A])
    current = result['I_pu'] * BaseI * 1000 / num_parallel * Branch_status
    seg_mask = np.sign(segment_wea[:, :, 0])
    con_temp = heat_banlance_equation(np.expand_dims(current, 1), conductor_s, weather_s) * seg_mask
    result['con_temp'] = con_temp
    print(f'{network.buses.shape[0]} at {utc_timestamp} with !!!!!!!!! highest conductor temp: {con_temp.max()}')
    return con_temp.max()

def identify_critical_scenarios(network, selected_country, conductor, weather, year, month, args, ratio):
    hot_temp = []
    hot_date = []
    for day in range(1, 31):
        for hour in [14]:  # utc time stamp
            date = datetime(year, month, day, hour, 0, 0) 
            max_seg_temp = cal_max_seg_temp(network, selected_country, conductor, weather, date, args, ratio)
            hot_date.append(date)
            hot_temp.append(max_seg_temp)
    return hot_date, hot_temp

def find_critical_scenarios(conductor, weather, args, year_list):
    """
    Find the most critical dates
    """
    ratio = 50
    # if selected_country in ['ES', 'FR',]:
    #     ratio = 25
    # elif selected_country in ['DE', 'GB', 'IT']:
    #     ratio = 50
    # else:
    #     ratio = 75
    if ratio < 100:
        network_filename = RESOURCES + f"{selected_country}/networks/elec_s_{ratio}.nc"  
    else:
        network_filename = RESOURCES + f"{selected_country}/networks/elec_s.nc"      
    network = pypsa.Network(network_filename)
    network.buses['pop_ratio'] = 0 
    pop = network.loads_t['p_set'].iloc[12]
    network.buses.loc[network.loads_t['p_set'].columns, 'pop_ratio'] = pop/pop.sum()
    network = divide_segments(network)
    hot_date_list = {}
    for year in year_list:
        month = 7
        hot_date, hot_temp = identify_critical_scenarios(network, selected_country, conductor, weather, year, month, args, ratio)
        print(year, np.max(hot_temp))
        hot_date_list[f'{year}_{month}'] = [hot_date, hot_temp]
        np.save(f'code/models/critical_date/{selected_country}_critical_scenarios.npy', hot_date_list, allow_pickle=True)

def load_critical_scenarios(selected_country):
    his_year_list = [2019, 2022] 
    Fut_year_list = [range(2025,2030), range(2030,2035), range(2035,2040), range(2040,2045), range(2045, 2050)]
    hot_date_list = np.load(f'code/models/critical_date/{selected_country}_critical_scenarios.npy', allow_pickle=True)
    hot_date_list = hot_date_list.item()
    hot_day_list = []
    hot_temp_list = []
    for year in his_year_list:
        [hot_date, hot_temp] = hot_date_list[f'{year}_{7}']
        hot_date = np.array(hot_date)
        hot_temp = np.array(hot_temp)
        hot_index = np.argsort(hot_temp)[-2:]
        hot_day_list.append(hot_date[hot_index])
        hot_temp_list.append(hot_temp[hot_index])
    for year_list in Fut_year_list:
        fut_hot_day_list = []
        fut_hot_temp_list = []
        for year in year_list:
            [hot_date, hot_temp] = hot_date_list[f'{year}_{7}']
            weekday_index = np.array([1 if (datetime(year, 7, i+1).weekday() < 5) else 0 for i in range(31)])
            hot_date = np.array(hot_date)[weekday_index]
            hot_temp = np.array(hot_temp)[weekday_index]
            hot_index = np.argmax(hot_temp)
            fut_hot_day_list.append(hot_date[hot_index])
            fut_hot_temp_list.append(hot_temp[hot_index])   
        hot_index = np.argsort(fut_hot_temp_list)[-2:]
        fut_hot_day_list = np.array(fut_hot_day_list)
        fut_hot_temp_list = np.array(fut_hot_temp_list)
        hot_day_list.append(fut_hot_day_list[hot_index])
        hot_temp_list.append(fut_hot_temp_list[hot_index]) 
    hot_day_list = np.concatenate(hot_day_list)
    return hot_day_list

def run_td_acopf_eur(ppc, base_result, tdpf_analysis, base_resistance, base_seg_resistance, conductor, weather, num_iter=10, tol=1e-3):
    num_bundle = ppc['branch'][:, -3] 
    num_bundle[num_bundle==0] = 1
    ref_temp = conductor['ref_temperature']
    resistance_ratio = conductor['resistance_ratio']
    baseMva = ppc['baseMVA']
    baseKV = ppc['bus'][0, idx_bus.BASE_KV]
    BaseI = baseMva / baseKV
    prev_pg = 0
    if base_result is not None:
        I_pu = base_result['I_pu']
        I = I_pu*BaseI*1000
        per_I = I/num_bundle
        con_temp = heat_banlance_equation(np.expand_dims(per_I, 1), conductor, weather) 
        if 'seg' in tdpf_analysis:
            td_seg_resistance = td_resistance(base_seg_resistance, con_temp, ref_temp, resistance_ratio) 
            ppc['branch'][:, idx_brch.BR_R] = np.sum(td_seg_resistance, -1)
        else:
            td_branch_resistance = td_resistance(base_resistance, con_temp, ref_temp, resistance_ratio) 
            ppc['branch'][:, idx_brch.BR_R] = np.squeeze(td_branch_resistance, 1)
    for iter in range(num_iter):
        result = pyomo_solve_ac(ppc, initial_value=base_result, ex_gen=False, tem_cons=True, angle_cons=False)
        I_pu = result['I_pu']
        I = I_pu*BaseI*1000
        per_I = I/num_bundle
        con_temp = 0.5 * con_temp + 0.5 * heat_banlance_equation(np.expand_dims(per_I, 1), conductor, weather) 
        # con_temp = heat_banlance_equation(np.expand_dims(per_I, 1), conductor, weather) 
        if 'seg' in tdpf_analysis:
            td_seg_resistance = td_resistance(base_seg_resistance, con_temp, ref_temp, resistance_ratio) 
            ppc['branch'][:, idx_brch.BR_R] = np.sum(td_seg_resistance, -1)
        else:
            td_branch_resistance = td_resistance(base_resistance, con_temp, ref_temp, resistance_ratio) 
            ppc['branch'][:, idx_brch.BR_R] = np.squeeze(td_branch_resistance, 1)
                
        if iter>0 and np.max((np.abs(result['PG'] - prev_pg))) < tol:
            break
        else:
            print(f'{iter}-th inter with obj: ', result['obj'], end='\r')
            base_result = result
            prev_pg = base_result['PG']
    return ppc, result

def ACOPF_analysis_eur(network, args, ppc_expr, conductor, result, result_row, base_cost, utc_timestamp, tdpf_analysis='post_1'):
    renewable_mode = args['renewable_mode']
    safe_threshold = args['safe_threshold']
    heatwave_mode = args['heatwave_mode']
    artificial_delta_temp = args['artificial_delta_temp']
    heatwave_year = args['heatwave_year']
    load_growth = args['load_growth']

    data_id = result_row['exper_id']
    baseMva = ppc_expr['baseMVA']
    baseKV = ppc_expr['baseKV']
    BaseI = ppc_expr['baseI']
    ## Pd,Qd,Pg,Qg
    Pg = result['PG']
    Pd = result['PD']
    cost = result['obj']
    ## Pij, Pji, Qij, Qji, Smax
    Branch_status = np.sign(ppc_expr['branch'][:, idx_brch.BR_STATUS] * ppc_expr['branch'][:, idx_brch.RATE_A])
    Gen_status = np.sign(ppc_expr['gen'][:, idx_gen.GEN_STATUS] * ppc_expr['gen'][:, idx_gen.PMAX])
    Smax = ppc_expr['branch'][:, idx_brch.RATE_A] / baseMva
    Imax = result['Imax'] / BaseI
    Pmax = ppc_expr['gen'][:, idx_gen.PMAX] / baseMva
    S_pu = result['S_pu']
    I_pu = result['I_pu']
    mis_match = result['eq_vio']
    p_mis_match = result['p_eq_vio']
    q_mis_match = result['q_eq_vio']
    Sratio = S_pu/(Smax+1e-8) * Branch_status
    Gratio = Pg / (Pmax + 1e-8) * Gen_status
    Iratio = I_pu/Imax * Branch_status
    seg_prop = ppc_expr['segment'][:, :, 2]
    seg_mask = np.sign(seg_prop)
    seg_temp = result['con_temp'] * seg_mask
    branch_temp = np.sum(seg_prop * seg_temp, axis=1) * Branch_status
    Tratio = seg_temp / conductor['max_temperature']
    network.lines['Sratio'] = Sratio
    network.lines['Iratio'] = Iratio
    network.lines['branch_temp'] = branch_temp
    network.generators['Gratio'] = Gratio
    network.buses['load_shedding'] = p_mis_match
    network.Tratio = Tratio
    # total_resistance = ppc_expr['branch'][:, idx_brch.BR_R]
    # Ploss = I_pu**2 * total_resistance



    record_path = f'code/models/{selected_country}/{utc_timestamp}/record'
    os.makedirs(record_path, exist_ok=True)
    network.export_to_netcdf(record_path + f'/{data_id}_network.nc')

    # network = pypsa.Network(record_path + f'/{data_id}_network.nc')
    # network_plot(network, result, selected_country, utc_timestamp, data_id, args)
    violin_plot(network, result, selected_country, utc_timestamp, data_id, args)

    result_row['load'] = np.sum(Pd) * baseMva * 3 / 1000 # GW total load (Three-phase ACOPF)

    # p_mis_match_ratio = p_mis_match / Pd
    # p_mis_match_ratio[np.isnan(p_mis_match_ratio)] = 0
    # p_mis_match_ratio[np.isinf(p_mis_match_ratio)] = 0

    result_row['node_load_shedding'] = np.sum(p_mis_match)/np.sum(Pd)
    result_row['node_mis_match_0001_num'] = np.sum((p_mis_match >= 0.001).astype(int)) # use 0.1% percent load sheding as threshold.
    result_row['node_mis_match_001_num'] = np.sum((p_mis_match >= 0.01).astype(int)) # use 0.1% percent load sheding as threshold.
    result_row['node_mis_match_01_num'] = np.sum((p_mis_match >= 0.1).astype(int)) # use 0.1% percent load sheding as threshold.
    # result_row['node_mis_match_ratio'] = result_row['node_mis_match_num'] /network.buses.shape[0] 
    # index = p_mis_match.argmax()
    # if result_row['node_load_shedding'] > 0.01:
    #     print(f'max mismatch ratio: {p_mis_match_ratio[index]*100:.4f}', Pd[index], p_mis_match[index], p_mis_match.max())
    #     print(1/0)
    result_row['cost_ratio'] = (cost - base_cost) / base_cost #* 100

    result_row['generator_utilization'] = np.sum(Gratio) / np.sum(Gen_status)
    result_row['branch_utilization'] = np.sum(Sratio) / np.sum(Branch_status) # * 100

    safe_thresh = 0.7
    result_row[f'{safe_thresh}_generator_num'] = np.sum((Gratio >= safe_thresh).astype(int))
    result_row[f'{safe_thresh}_flow_num'] = np.sum((Sratio >= safe_thresh).astype(int))
    result_row[f'{safe_thresh}_cur_num'] = np.sum((Iratio >= safe_thresh).astype(int))
    result_row[f'{safe_thresh}_temp_num'] = np.sum((Tratio >= safe_thresh).astype(int))

    result_row['tem_con_mean'] = np.sum(branch_temp) / np.sum(Branch_status)
    result_row['tem_con_max'] = np.max(seg_temp)
    result_row['speedup'] = result['speedup']

    safe_thresh = 0.8
    result_row[f'{safe_thresh}_generator_num'] = np.sum((Gratio >= safe_thresh).astype(int))
    result_row[f'{safe_thresh}_flow_num'] = np.sum((Sratio >= safe_thresh).astype(int))
    result_row[f'{safe_thresh}_cur_num'] = np.sum((Iratio >= safe_thresh).astype(int))
    result_row[f'{safe_thresh}_temp_num'] = np.sum((Tratio >= safe_thresh).astype(int))

    safe_thresh = 0.9
    result_row[f'{safe_thresh}_generator_num'] = np.sum((Gratio >= safe_thresh).astype(int))
    result_row[f'{safe_thresh}_flow_num'] = np.sum((Sratio >= safe_thresh).astype(int))
    result_row[f'{safe_thresh}_cur_num'] = np.sum((Iratio >= safe_thresh).astype(int))
    result_row[f'{safe_thresh}_temp_num'] = np.sum((Tratio >= safe_thresh).astype(int))

    result_row['generation'] = np.sum(Pg)
    result_row['generation_ratio'] = (np.sum(Pg)) / np.sum(Pd)
    result_row[f'{safe_thresh}_current_num'] = np.sum((Iratio >= safe_thresh).astype(int))

    # result_row['total_power_loss'] = np.sum(Ploss)
    # result_row['flow_mean'] = np.mean(S)
    # result_row['flow_max'] = np.max(S)
    # result_row['flow_limit_ratio_max'] = np.max(Sratio) # * 100
    # result_row['branch_loss_mean'] = np.mean(Ploss)
    # result_row['branch_loss_max'] = np.max(Ploss)
    # result_row['branch_loss_ratio_mean'] = np.sum(Ploss / (P+1e-5)) / np.sum(Branch_status) # * 100
    # result_row['branch_loss_ratio_max'] = np.max(Ploss / (P+1e-5)) # * 100

    return result_row

def TDACOPF_eur(df, network, selected_country, utc_timestamp, args, conductor, weather, expr_id,
                tdpf_analysis='base', warm_start=True):
    """
    load data to ppc
    """
    ppc = pypsa_pypower(network, args)
    renewable_mode = args['renewable_mode']
    safe_threshold = args['safe_threshold']
    artificial_delta_temp = args['artificial_delta_temp']
    heatwave_mode = args['heatwave_mode']
    heatwave_year = args['heatwave_year']
    load_growth = args['load_growth']
    nbus = len(ppc['bus'])
    baseMva = ppc['baseMVA']
    baseKV = ppc['baseKV']
    BaseI = ppc['baseI']
    num_parallel = network.lines['num_parallel'].values
    num_parallel[num_parallel==0] = 1
    base_resistance = np.expand_dims(ppc['branch'][:, idx_brch.BR_R], 1)
    base_seg_resistance = ppc['segment'][:, :, 2] * base_resistance

    """
    running TD ACOPF case
    """
    ppc_expr = copy.deepcopy(ppc)
    segment_wea = copy.deepcopy(network.segment_info)
    branch_wea = copy.deepcopy(network.lines[['temperature', 'wnd10m', 'influx_toa', 'wnd_azimuth', 'height']].values)
    bus_wea = copy.deepcopy(network.buses[['temperature', 'wnd10m', 'influx_toa', 'wnd_azimuth', 'height']].values)
    if 'seg' in tdpf_analysis:
        weather['air_temperature'] = segment_wea[:, :, 0] 
        weather['wind_speed'] = segment_wea[:, :, 1]
        weather['solar_heat_intensity'] = segment_wea[:, :, 2]
        weather['wind_angle'] = segment_wea[:, :, 3]
        conductor['elevation'] = segment_wea[:, :, 4]
    else:
        weather['air_temperature'] = branch_wea[:, [0]] 
        weather['wind_speed'] = branch_wea[:, [1]]
        weather['solar_heat_intensity'] = branch_wea[:, [2]]
        weather['wind_angle'] = branch_wea[:, [3]]
        conductor['elevation'] = branch_wea[:, [4]]
    xe = network.buses.loc[network.lines['bus1'], ['x','y']].values - \
         network.buses.loc[network.lines['bus0'], ['x','y']].values
    conductor_angle = (90 - np.rad2deg(np.arctan2(xe[:, 1], xe[:, 0]))) % 360 # from north (0), east (90), south (180), weast (270)
    conductor['conductor_angle'] = np.reshape(conductor_angle, (-1,1))
    air_temp = weather['air_temperature'] 
    ref_temp = conductor['ref_temperature']
    bus_temp = bus_wea[:, 0] 
    seg_prop = network.segments[:, :, 2]
    seg_mask = np.sign(network.segments[:, :, 2])
    if args['sensitivity_analysis_mode'] in [3]:
        weather_copy = copy.deepcopy(weather)
        weather_copy['air_temperature'] = weather_copy['air_temperature'] - args['artificial_delta_temp']
        Imax = maximum_allowable_current(conductor, weather_copy) * np.expand_dims(num_parallel, 1)
    else:
        Imax = maximum_allowable_current(conductor, weather) * np.expand_dims(num_parallel, 1)
    Imax = np.min(Imax, 1) # the minimum one of all segments
    ppc_expr['branch'][:, idx_brch.RATE_B] = Imax / 1000

    ppc_path = f'code/models/{selected_country}/ppc/renew_{renewable_mode}_heat_{heatwave_mode}_{artificial_delta_temp}_{heatwave_year}_load_growth_{load_growth}_{selected_country}_{nbus}'
    os.makedirs(ppc_path, exist_ok=True)
    
    """
    running acopf case         
    analysis_list = ['base', 'quadratic', 'post_1_tem', 'post_10_tem',
                             'quadratic_seg', 'post_1_tem_seg', 'post_10_tem_seg']
    """
    if tdpf_analysis in ['base']:
        ppc_base = copy.deepcopy(ppc)
        st = time.time()
        result = pyomo_solve_ac(ppc_base, solver_name='ipopt')
        et = time.time()
        base_runtime = et - st
        result['run_time'] = base_runtime
        base_cost = result['obj']
        result['speedup'] = 1

        weather_s = copy.deepcopy(weather)
        conductor_s = copy.deepcopy(conductor)
        weather_s['air_temperature'] = segment_wea[:, :, 0]
        weather_s['wind_speed'] = segment_wea[:, :, 1]
        weather_s['solar_heat_intensity'] = segment_wea[:, :, 2]
        conductor_s['elevation'] = segment_wea[:, :, 4]
        Branch_status = np.sign(ppc_expr['branch'][:, idx_brch.BR_STATUS] * ppc_expr['branch'][:, idx_brch.RATE_A])
        current = result['I_pu'] * BaseI * 1000 / num_parallel * Branch_status
        con_temp = heat_banlance_equation(np.expand_dims(current, 1), conductor_s, weather_s) * seg_mask
        result['con_temp'] = con_temp
        Imax = maximum_allowable_current(conductor_s, weather_s) * np.expand_dims(num_parallel, 1)
        Imax = np.min(Imax, 1) # the minimum one of all segments
        result['Imax'] = Imax / 1000
        print(f'!!!!!!!!!!!!!!!!!!! highest conductor temp at {utc_timestamp}: {con_temp.max()}')
        np.save(ppc_path + f'/{utc_timestamp}_{tdpf_analysis}_results.npy', result, allow_pickle=True)


    base_result_file = ppc_path + f'/{utc_timestamp}_base_results.npy'
    if os.path.exists(base_result_file):
        base_result = np.load(base_result_file, allow_pickle=True).item()
        base_runtime = base_result['run_time'] 
        base_cost = base_result['obj']
        con_temp = base_result['con_temp']

    if warm_start == False:
        init_result = None
    else:
        init_result = base_result

    # if con_temp.max() > 30:
    ### modify temperature-dependent resistance ###
    resistance_ratio = conductor['resistance_ratio']
    if  'seg' in tdpf_analysis:
        td_seg_resistance = td_resistance(base_seg_resistance, air_temp, ref_temp, resistance_ratio) 
        ppc_expr['branch'][:, idx_brch.BR_R] = np.sum(td_seg_resistance, -1)
    else:
        td_branch_resistance = td_resistance(base_resistance, air_temp, ref_temp, resistance_ratio)
        ppc_expr['branch'][:, idx_brch.BR_R] = np.squeeze(td_branch_resistance, 1)

    if tdpf_analysis in ['base']:
        result = base_result
        runtime = base_runtime
    elif tdpf_analysis in ['quadratic', 'quadratic_seg']:
        st = time.time()
        result = pyomo_solve_ac(ppc_expr,  initial_value=init_result, qua_con =True, 
                                tem_cons=False, angle_cons=False, qlim=False,
                                conductor=conductor, weather=weather)
        et = time.time()
        runtime = et - st
        np.save(ppc_path + f'/{utc_timestamp}_{tdpf_analysis}_results.npy', result, allow_pickle=True)
    elif tdpf_analysis in ['dlr_seg']:
        ### calculating dynamic line rating
        st = time.time()
        result = pyomo_solve_ac(ppc_expr,  initial_value=init_result, qua_con =False, 
                                tem_cons=True, angle_cons=False, qlim=False)
        et = time.time()
        runtime = et - st
        np.save(ppc_path + f'/{utc_timestamp}_{tdpf_analysis}_results.npy', result, allow_pickle=True)
    else:
        num_iter = int(re.findall(r'\d+', tdpf_analysis)[0])
        st = time.time()
        ppc_expr, result = run_td_acopf_eur(ppc_expr, init_result, tdpf_analysis, base_resistance, base_seg_resistance, conductor, weather, num_iter)
        et = time.time()
        runtime = et - st
        np.save(ppc_path + f'/{utc_timestamp}_{tdpf_analysis}_results.npy', result, allow_pickle=True)
    
    result['speedup'] = runtime/base_runtime
    weather_s = copy.deepcopy(weather)
    conductor_s = copy.deepcopy(conductor)
    weather_s['air_temperature'] = segment_wea[:, :, 0]
    weather_s['wind_speed'] = segment_wea[:, :, 1]
    weather_s['solar_heat_intensity'] = segment_wea[:, :, 2]
    conductor_s['elevation'] = segment_wea[:, :, 4]
    Branch_status = np.sign(ppc_expr['branch'][:, idx_brch.BR_STATUS] * ppc_expr['branch'][:, idx_brch.RATE_A])
    current = result['I_pu'] * BaseI * 1000 / num_parallel * Branch_status
    con_temp = heat_banlance_equation(np.expand_dims(current, 1), conductor_s, weather_s) * seg_mask
    result['con_temp'] = con_temp
    # print(bus_temp, (segment_wea[:, :, 0] * seg_prop).sum(1), (con_temp * seg_prop).sum(1) )
    # print(1/0)
    Imax = maximum_allowable_current(conductor_s, weather_s) * np.expand_dims(num_parallel, 1)
    Imax = np.min(Imax, 1) # the minimum one of all segments
    result['Imax'] = Imax / 1000

    ### OPF analysis ###    
    # expr_id = f'{tdpf_analysis}_renew_{renewable_mode}_heat_{heatwave_mode}_{artificial_delta_temp}_{heatwave_year}_load_growth_{load_growth}_{selected_country}_{nbus}_{utc_timestamp}'
    result_row = {'TDPF_solver': tdpf_analysis,
        'renewable_mode': renewable_mode,
        'heatwave_mode': heatwave_mode,
        'artificial_delta_temp': args['artificial_delta_temp'],
        'load_growth': load_growth,
        'selected_country': selected_country,
        'nbus': nbus,
        'utc_timestamp': utc_timestamp,
        'exper_id': expr_id,  
        'year': utc_timestamp.year,
        'air_temp': np.mean(bus_wea[:, 0] ),
        'wind_speed': np.mean(bus_wea[:, 1] ),
        'solar_radia': np.mean(bus_wea[:, 2] )}
    print(tdpf_analysis, result_row['exper_id'], bus_temp.mean())  
    result_row = ACOPF_analysis_eur(network, args, ppc_expr, conductor, result, result_row, base_cost, utc_timestamp, tdpf_analysis)
    df = df._append(result_row, ignore_index=True)
    return df



def __main__(selected_country = 'FR'):
    args = {'date': None,
            'BaseMVA': 100.0, # numberial processing
            'phase_factor': 3.0, # balanced 3-phase ACOPF
            'renewable_mode': True, 
            'heatwave_mode': True, # consider heat wave
            'sensitivity_analysis_mode': False,
            'heatwave_year': 2019, #
            'load_growth': True, # 1.1% base load growth from 2020 - 2050
            'extendable_gen_capacity': False,
            'safe_threshold': True, 
            'reactive_demand_ratio': 0.15, # fixed reactive power demand
            'reactive_gen_upper':  0.8, # reactive power generation upper bound
            'reactive_gen_lower': -0.8, # reactive power generation lower bound
            'voltage_upper': 1.05, # voltage upper bound
            'voltage_lower': 0.95, # voltage lower bound
            'phase_angle_upper': 30, 
            'phase_angle_lower': -30,
            } 

    conductor = {'diameter': 0.0312, # m
                'ref_temperature': 20, # °C
                'max_temperature': 90, # °C 
                'resistance_ratio': 0.00429, # om/°C
                'unit_resistance': 0.03 * 1e-3, # om/m 
                'conductor_angle': 0,
                'elevation': 100, # conductor elevation 
                }
    
    ### weather data (IEEE standard ):
    weather = {'wind_speed':  0.1, # m/s
            'wind_angle': math.pi/2, # rad
            'air_density': 1.029, # kg/m^3
            'air_viscosity': 2.043*1e-5, # kg/m-s
            'air_conductivity': 0.02945, # W/m-°C
            'air_temperature': 30, # °C
            'radiation_emissivity': 0.8, 
            'solar_absorptivity': 0.8, 
            'solar_heat_intensity': 600 # W/m^2
            }



    # find_critical_scenarios(conductor, weather, args, [2019, 2022]+[i for i in range(2025,2051)])
    hot_day_list = load_critical_scenarios(selected_country)
    rcp = 45

    if selected_country in ['GB', 'FR',]:
        ratio = 75
    elif selected_country in ['DE','IT']:
        ratio = 100
    elif selected_country in ['ES']:
        ratio = 50
    elif selected_country in ['BE']:
        ratio = 100

    if ratio < 100:
        network_filename = RESOURCES + f"{selected_country}/networks/elec_s_{ratio}.nc"  
    else:
        network_filename = RESOURCES + f"{selected_country}/networks/elec_s.nc"  
    network = pypsa.Network(network_filename)
    network.buses['pop_ratio'] = 0 
    pop = network.loads_t['p_set'].iloc[1]
    network.buses.loc[network.loads_t['p_set'].columns, 'pop_ratio'] = pop/pop.sum()
    network = divide_segments(network)
    nbus = network.buses.shape[0]
    calibrate_future_load(network, selected_country, year_list=[2017,2019])


    # analysis_list = ['base',
    #                  'quadratic_seg',
    #                  'post_2_tem_seg',
    #                 'post_20_tem_seg'
    #                 ]
    # csv_path = f'code/models/{selected_country}'
    # os.makedirs(csv_path, exist_ok=True)

    # max_tem = 90
    # load_growth = True
    # heatwave_mode = True
    # artificial_delta_temp = 0
    # safe_threshold = False 
    # heatwave_year = 2019
    # renewable_mode = True
    # warm_start = True
    
    # conductor['max_temperature'] = max_tem
    # args['renewable_mode'] = renewable_mode
    # args['load_growth'] = load_growth
    # args['heatwave_year'] = heatwave_year
    # args['safe_threshold'] = safe_threshold
    # args['artificial_delta_temp'] = artificial_delta_temp
    # args['heatwave_mode'] = heatwave_mode
    # args['warm_start'] = warm_start

    # # df_path = csv_path + f'/renewable_{renewable_mode}_load_growth_{load_growth}_heat_{heatwave_mode}_{heatwave_year}_threshold_{safe_threshold}_{selected_country}_{nbus}_bus_{max_tem}_{rcp}_TDOPF_heatflow_analysis.csv'
    # # df = pd.DataFrame(columns=['exper_id'])
    # # # if os.path.exists(df_path):
    # # #     df = pd.read_csv(df_path)
    # # # else:
    # # #     df = pd.DataFrame(columns=['exper_id'])
    # # for tdpf_analysis in analysis_list:
    # #     for utc_dt in hot_day_list:
    # #         expr_id = f'{tdpf_analysis}_renew_{renewable_mode}_heat_{heatwave_mode}_{artificial_delta_temp}_{heatwave_year}_load_growth_{load_growth}_{selected_country}_{nbus}_{utc_dt}'
    # #         # if expr_id not in df['exper_id'].values:
    # #         network = network_load_weather(network, utc_dt, selected_country, args)
    # #         network = network_load_demand(network, utc_dt, selected_country, args)
    # #         network = network_load_renewable(network, utc_dt, selected_country, args, ratio)
    # #         df = TDACOPF_eur(df, network, selected_country, utc_dt, args, conductor, weather, expr_id, tdpf_analysis, warm_start)
    # #         # df.to_csv(df_path, index=False)



    # ### critical dates at 2025
    # hot_date_list = np.load(f'code/models/critical_date/{selected_country}_critical_scenarios.npy', allow_pickle=True)
    # hot_date_list = hot_date_list.item()
    # [hot_date, hot_temp] = hot_date_list[f'{2025}_{7}']
    # weekday_index = np.array([1 if (datetime(2025, 7, i+1).weekday() < 5) else 0 for i in range(31)])
    # hot_date = np.array(hot_date)[weekday_index]
    # hot_temp = np.array(hot_temp)[weekday_index]
    # hot_index = np.argmax(hot_temp)
    # utc_dt = hot_date[hot_index]

    # # df_path = csv_path + f'/renewable_{renewable_mode}_load_growth_{load_growth}_delta_timep_threshold_{safe_threshold}_{selected_country}_{nbus}_bus_{max_tem}_{rcp}_TDOPF_heatflow_analysis.csv'
    # # # df = pd.DataFrame(columns=['exper_id'])
    # # if os.path.exists(df_path):
    # #     df = pd.read_csv(df_path)
    # # else:
    # #     df = pd.DataFrame(columns=['exper_id'])
    # # heatwave_mode = False
    # # args['heatwave_mode'] = heatwave_mode
    # # for tdpf_analysis in ['base', 'post_2_tem_seg']:
    # #     for artificial_delta_temp in range(0, 25, 2):
    # #         args['artificial_delta_temp'] = artificial_delta_temp
    # #         expr_id = f'{tdpf_analysis}_renew_{renewable_mode}_heat_{heatwave_mode}_{artificial_delta_temp}_{heatwave_year}_load_growth_{load_growth}_{selected_country}_{nbus}_{utc_dt}'
    # #         if expr_id not in df['exper_id'].values:
    # #             network = network_load_weather(network, utc_dt, selected_country, args)
    # #             network = network_load_demand(network, utc_dt, selected_country, args)
    # #             network = network_load_renewable(network, utc_dt, selected_country, args, ratio)
    # #             df = TDACOPF_eur(df, network, selected_country, utc_dt, args, conductor, weather, expr_id, tdpf_analysis, warm_start)
    # #             df.to_csv(df_path, index=False)




    # df_path = csv_path + f'/renewable_{renewable_mode}_load_growth_{load_growth}_sensitivity_analysis_{safe_threshold}_{selected_country}_{nbus}_bus_{max_tem}_{rcp}_TDOPF_heatflow_analysis.csv'
    # if os.path.exists(df_path):
    #     df = pd.read_csv(df_path)
    # else:
    #     df = pd.DataFrame(columns=['exper_id'])
    # heatwave_mode = False
    # args['heatwave_mode'] = heatwave_mode
    # """
    # case 1: increase temp, load increase by temperauture-dependent model (joint effects by load + and capacity - )
    # case 2: increase temp, keep load with default temps (effect only by capacity - )
    # case 3: keep default temp, increase load by temperauture-dependent model (effect inly by demand +)
    # """
    # tdpf_analysis = 'post_2_tem_seg'
    # for sensitivity_analysis_mode in [1, 3]:
    #     args['sensitivity_analysis_mode'] = sensitivity_analysis_mode
    #     for artificial_delta_temp in range(0, 25, 2):
    #         args['artificial_delta_temp'] = artificial_delta_temp
    #         expr_id = f'{tdpf_analysis}_renew_{renewable_mode}_heat_{heatwave_mode}_sensitivity_{sensitivity_analysis_mode}_{artificial_delta_temp}_{heatwave_year}_load_growth_{load_growth}_{selected_country}_{nbus}_{utc_dt}'
    #         if expr_id not in df['exper_id'].values:
    #             network = network_load_weather(network, utc_dt, selected_country, args)
    #             network = network_load_demand(network, utc_dt, selected_country, args)
    #             network = network_load_renewable(network, utc_dt, selected_country, args, ratio)
    #             df = TDACOPF_eur(df, network, selected_country, utc_dt, args, conductor, weather, expr_id, tdpf_analysis, warm_start)
    #             df.to_csv(df_path, index=False)




import pandas as pd
if __name__ == '__main__':
    for selected_country in  ['BE', 'IT', 'ES', 'FR', 'GB', 'DE']:  # 'BE', 'IT', 'ES', 'FR', 'GB', 'DE'
        __main__(selected_country)



