from TDOPF import *
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from datetime import datetime
from utils.data_config import *
import xarray as xr
from pypop7.optimizers.de.jade import JADE
from pypop7.optimizers.de.shade import SHADE

from pypop7.optimizers.cem.cem import CEM  # abstract class of all cross-entropy method (CEM) classes
class SCEM(CEM):
    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)
        self.alpha = options.get('alpha', 0.8)  # smoothing factor
        assert 0.0 <= self.alpha <= 1.0

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)
        x = np.empty((self.n_individuals, self.ndim_problem))  # samples (population)
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return mean, x, y

    def iterate(self, mean=None, x=None, y=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            x[i] = mean + self._sigmas*self.rng_optimization.standard_normal((self.ndim_problem,))
            x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def _update_parameters(self, mean=None, x=None, y=None):
        xx = x[np.argsort(y)[:self.n_parents]]
        mean = self.alpha*np.mean(xx, axis=0) + (1.0-self.alpha)*mean
        mean = np.clip(mean, self.lower_boundary, self.upper_boundary)
        self._sigmas = self.alpha*np.std(xx, axis=0) + (1.0-self.alpha)*self._sigmas
        return mean

    def optimize(self, fitness_function=None, args=None):
        fitness = CEM.optimize(self, fitness_function)
        mean, x, y = self.initialize()
        while True:
            x, y = self.iterate(mean, x, y, args)
            self._print_verbose_info(fitness, y)
            if self._check_terminations():
                break
            self._n_generations += 1
            mean = self._update_parameters(mean, x, y)
        return self._collect(fitness, y, mean)



### Code for BAIT is from public code of
### "Staffell, I., Pfenninger, S., & Johnson, N. (2023). A global model of hourly space heating and cooling demand at multiple spatial scales. 
### Nature Energy, 8(12), 1328-1344."
def smooth_temperature_df(temperature_df: pd.DataFrame, weights: list) -> pd.DataFrame:
    assert isinstance(temperature_df, pd.DataFrame), "Input must be a pandas DataFrame"
    # Initialize the smoothed DataFrame
    smoothed_df = temperature_df.copy()
    # Run through each weight in turn going one time-step backwards each time
    for i, w in enumerate(weights):
        if w != 0:
            # Create a time series of temperatures the day before for each location
            lagged = temperature_df.shift(i + 1)
            # Forward fill the missing entries at the beginning of the DataFrame to avoid NaNs
            lagged.fillna(method='bfill', inplace=True)
            # Add on these lagged temperatures multiplied by this smoothing factor
            smoothed_df += lagged * w

    # Normalize the smoothed DataFrame
    total_weight = 1 + sum(weights)
    smoothed_df /= total_weight

    return smoothed_df

def _bait(tem, wind, solar, para):
    smoothing = para['smoothing']#0.5
    solar_gains = para['solar_gains']#0.019
    wind_chill = para['wind_chill']#-0.13
    # humidity_discomfort = para['humidity_discomfort']#0.05
    wind2m = wind_speed_hight(wind, hight0=10, hight1=2)
    bait = copy.deepcopy(tem)
    """
    Building-adjust-internal-temperature
    """
    setpoint_S = 100 + 7 * tem  # W/m2
    setpoint_W = 4.5 - 0.025 * tem  # m/s
    setpoint_H = np.e ** (1.1 + 0.06 * tem)  # g water per kg air
    setpoint_T = 16  # degrees - around which 'discomfort' is measured
    # BAIT = air_temp + 
    # If it's sunny, it feels warmer
    bait = bait + (solar - setpoint_S) * solar_gains
    # If it's windy, it feels colder
    bait = bait + (wind2m - setpoint_W) * wind_chill
    # If it's humid, both hot and cold feel more extreme
    # assume normal cases without data
    # # If it's humid, both hot and cold feel more extreme
    # discomfort = N - setpoint_T
    # N = (
    #     setpoint_T
    #     + discomfort
    #     + (
    #         discomfort
    #         # Convert humidity from g/kg to kg/kg
    #         * ((weather["humidity"] / 1000) - setpoint_H)
    #         * humidity_discomfort
    #     )
    # )

    # Apply temporal smoothing to our temperatures over the last two days
    # we assume 2nd day smoothing is the square of the first day (i.e. compounded decay)
    bait = smooth_temperature_df(pd.DataFrame(bait), weights=[smoothing, smoothing**2]).values

    # These are fixed parameters we don't expose the user to
    lower_blend = 15  # *C at which we start blending T into N
    upper_blend = 23  # *C at which we have fully blended T into N
    max_raw_var = 0.5  # maximum amount of T that gets blended into N
    # Transform this window to a sigmoid function, mapping lower & upper onto -5 and +5
    avg_blend = (lower_blend + upper_blend) / 2
    dif_blend = upper_blend - lower_blend
    blend = (tem - avg_blend) * 10 / dif_blend
    blend = max_raw_var / (1 + np.e ** (-blend))
    # Apply the blend
    bait = (tem * blend) + (bait * (1 - blend))

    return bait

def smooth_temperature(temperature: pd.Series, weights: list) -> pd.Series:
    """
    Smooth a temperature series over time with the given weighting for previous days.

    Params
    ------

    temperature : pd.Series
    weights : list
        The weights for smoothing. The first element is how much
        yesterday's temperature will be, the 2nd element is 2 days ago, etc.

    """
    assert isinstance(temperature, pd.Series)
    lag = temperature.copy()
    smooth = temperature.copy()

    # Run through each weight in turn going one time-step backwards each time
    for w in weights:
        # Create a time series of temperatures the day before
        lag = lag.shift(1, fill_value=lag[0])

        # Add on these lagged temperatures multiplied by this smoothing factor
        if w != 0:
            smooth = (smooth + (lag * w)).reindex()

    smooth = smooth.reindex().dropna()

    # Renormalise and return
    return smooth / (1 + sum(weights))



def compute_r2(y_data, y_pred):
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    return r2_score

def compute_rmse(y_data, y_pred):
    return (np.mean((y_data - y_pred)**2))**0.5

def compute_mape(y_data, y_pred):
    return (np.mean(np.abs(y_data - y_pred)/y_data))



def calibrate_future_load(network, selected_country, year_list):
    EXTERNAL = "/Volumes/T9"
    bus_loc_list = network.buses[['x','y']].values
    pop_ratio = network.buses['pop_ratio'].values.reshape(1,-1)

    weather_filename =   EXTERNAL + '/era5/demand_cali/Historical_era5_data.nc'
    ncfile = xr.open_dataset(weather_filename)
    date_s = datetime(year_list[0], 1, 1, 0, 0, 0)
    date_e = datetime(year_list[-1]+1, 1, 1, 0, 0, 0)
    x_min, x_max = network.buses['x'].min() - 1,  network.buses['x'].max() + 1
    y_min, y_max = network.buses['y'].min() - 1,  network.buses['y'].max() + 1
    ncfile_slice = ncfile.sel(time=slice(date_s, date_e),x=slice(x_min, x_max ), y=slice(y_min, y_max ))    
    date_list = ncfile_slice.time.data
    x_da = xr.DataArray(bus_loc_list[:,0], dims=["bus"])
    y_da = xr.DataArray(bus_loc_list[:,1], dims=["bus"])
    bus_ncfile_slice = ncfile_slice.sel(x=x_da, y=y_da, method="nearest")
    bus_ncfile_slice_ave = bus_ncfile_slice[['temperature','wnd10m','influx']].resample(time='1D').mean()
    temp_data = bus_ncfile_slice_ave['temperature'].data - 273.15
    wind_data = bus_ncfile_slice_ave['wnd10m'].data
    solar_data = bus_ncfile_slice_ave['influx'].data
    date_list = date_list[::24].astype('datetime64[D]').astype('O')
    
    """
    demand data from OPSD 2015-2020
    """
    # demand_filename = './climate_data/entsoe/load_2015-2020.csv'
    # load_demand = pd.read_csv(demand_filename, index_col=False)
    # if code == 'GB':
    #     country_load_all = load_demand[f'{code}_GBN_load_actual_entsoe_transparency'].values + \
    #                 load_demand[f'{code}_NIR_load_actual_entsoe_transparency'].values
    # else:   
    #     country_load_all = load_demand[f'{code}_load_actual_entsoe_transparency'].values
    # date_list_load = pd.to_datetime(load_demand['utc_timestamp'],  format='%Y-%m-%d %H:%M')

    """
    demand data from Entsoe 2015-2019
    """
    country_load_all = []
    demand_filename = f'./climate_data/entsoe/MHLV_{2015}-{2017}.xlsx'
    load_demand = pd.read_excel(demand_filename, index_col=False) 
    country_load = load_demand[load_demand['CountryCode'] == selected_country]
    country_load_all.append(country_load)

    demand_filename = f'./climate_data/entsoe/MHLV_{2018}-{2019}.xlsx'
    load_demand = pd.read_excel(demand_filename, index_col=False) 
    country_load = load_demand[load_demand['CountryCode'] == selected_country]
    country_load_all.append(country_load)

    """
    demand data from Entsoe 2021-2023
    """
    # demand_filename = f'./climate_data/entsoe/MHLV_{2021}.xlsx'
    # load_demand = pd.read_excel(demand_filename, index_col=False) 
    # country_load = load_demand[load_demand['CountryCode'] == code]
    # country_load_all.append(country_load)

    # demand_filename = f'./climate_data/entsoe/MHLV_{2022}.xlsx'
    # load_demand = pd.read_excel(demand_filename, index_col=False) 
    # country_load = load_demand[load_demand['CountryCode'] == code]
    # country_load_all.append(country_load)

    # demand_filename = f'./climate_data/entsoe/MHLV_{2023}.xlsx'
    # load_demand = pd.read_excel(demand_filename, index_col=False) 
    # country_load = load_demand[load_demand['CountryCode'] == code]
    # country_load_all.append(country_load)


    country_load_all = pd.concat(country_load_all, axis=0, ignore_index=True)
    date_list_load = pd.to_datetime(country_load_all['DateShort'], format='%Y-%m-%d %H:%M')
    country_load_all = country_load_all['Value'].values

    """
    filter out weather data with valid load data
    """
    training_data_df = pd.DataFrame(columns=['date', 'load'])
    hourly_load = []
    valid_date = []
    for t in range(temp_data.shape[0]):
        year = date_list[t].year
        month = date_list[t].month
        day =  date_list[t].day
        date = datetime(year, month, day)
        day_of_week = date.weekday()
        load_index = (date_list_load.dt.year == year) & \
                        (date_list_load.dt.month == month) & \
                        (date_list_load.dt.day == day) 
        daily_load = country_load_all[load_index]
        if daily_load.shape[0] == 24 and daily_load.min() > 0:
            if [year, month, day] in holiday_list[selected_country][year]:
                holiday_index = 1
            else:
                holiday_index = 0
            valid_date.append(t)
            hourly_load.append(np.reshape(daily_load, (1,-1)))
            training_data_df = training_data_df._append({'date': f'{year}-{month}-{day}', 
                                                        'day_of_week':day_of_week,
                                                        'holiday': holiday_index,
                                                        'load': daily_load.sum() / 1e3 / 24}, ignore_index=True)
    valid_date = np.array(valid_date)
    hourly_load = np.array(hourly_load)
    weekday_index = ((training_data_df['day_of_week']<5) & (training_data_df['holiday']==0)).values.astype(int)
    weekend_index = ((training_data_df['day_of_week']>=5) | (training_data_df['holiday']==1)).values.astype(int)
    load = training_data_df['load'].values
    temp = temp_data[valid_date]
    wind = wind_data[valid_date]
    solar = solar_data[valid_date]

    xl = np.array([5, 15,   0.019-0.1,      -0.13-0.25,        0.62-0.2,  load.min(),   0])
    xu = np.array([15, 25, 0.019+0.1,      -0.13+0.25,        0.62+0.2, load.max(), load.min()])
    x_range = np.abs(xu - xl)
    def fitness_function(x):
        Th, Tc, solar_gains,wind_chill,smoothing,Pb,alpha = x * x_range
        # Th = demand_model_para[selected_country]['heating_threshold']
        # Tc = demand_model_para[selected_country]['cooling_threshold']
        Ph = demand_model_para[selected_country]['p_heating']
        Pc = demand_model_para[selected_country]['p_cooling']

        para = {}
        para['solar_gains'] = solar_gains
        para['wind_chill'] = wind_chill
        para['humidity_discomfort'] = 0.05
        para['smoothing'] = smoothing
        bait = _bait(temp, wind, solar, para)
        bait = (bait* pop_ratio).sum(-1)
        demand = Pb + Ph * np.maximum(Th - bait, 0)  + Pc * np.maximum(bait - Tc, 0) + alpha * weekday_index
        return compute_mape(load, demand)#(np.abs(demand - load)**2).mean()
    
    problem = {'fitness_function': fitness_function,  # define problem arguments
                'ndim_problem': 5,
                'lower_boundary': xl,
                'upper_boundary': xu}
    options = {'max_function_evaluations': 10000,  # set optimizer options
            'seed_rng': 2024,
            'mean': (xl+xu)/2,  # initial mean of Gaussian search distribution
            'sigma': 3}  # initial std (aka global step-size) of Gaussian search distribution
    demand_curve_path = f'code/models/demand_curve/{selected_country}_demand_curve.npy'
    if os.path.exists(demand_curve_path):
        demand_curve = np.load(demand_curve_path, allow_pickle=True).item()
        Th = demand_curve['heating_threshold']
        Tc = demand_curve['cooling_threshold']
        Ph = demand_curve['p_heating']
        Pc = demand_curve['p_cooling']
        Pb = demand_curve['base_demand']
        alpha = demand_curve['workday_demand']
        bait = _bait(temp, wind, solar, demand_curve)
        bait = (bait* pop_ratio).sum(-1)
    else:
        problem = {'fitness_function': fitness_function,  # define problem arguments
                'ndim_problem': 7,
                'lower_boundary': xl/x_range,
                'upper_boundary': xu/x_range}
        options = {'max_function_evaluations': 50000,  # set optimizer options
                'seed_rng': 8888,
                'mean': (xl+xu)/2/x_range,  # initial mean of Gaussian search distribution
                'alpha': 0.9,
                'sigma': 0.3}  # initial std (aka global step-size) of Gaussian search distribution
        opt = SCEM(problem, options)  # initialize the optimizer class
        results = opt.optimize()  # run the optimization process
        print(f"DE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
        Th, Tc, solar_gains,wind_chill,smoothing,Pb,alpha = results['best_so_far_x'] * x_range
        # Th = demand_model_para[selected_country]['heating_threshold']
        # Tc = demand_model_para[selected_country]['cooling_threshold']
        Ph = demand_model_para[selected_country]['p_heating']
        Pc = demand_model_para[selected_country]['p_cooling']

        demand_curve = demand_model_para[selected_country]
        demand_curve['solar_gains'] = solar_gains
        demand_curve['wind_chill'] = wind_chill
        demand_curve['humidity_discomfort'] = 0.05
        demand_curve['smoothing'] = smoothing
        demand_curve['base_demand'] = Pb
        demand_curve['workday_demand'] = alpha

        bait = _bait(temp, wind, solar, demand_curve)
        bait = (bait* pop_ratio).sum(-1)

        weekday_cooling_hour_ratio = np.mean(hourly_load[(weekday_index==1) & (bait >= Tc)] - Pb, axis=0)
        weekend_cooling_hour_ratio = np.mean(hourly_load[(weekday_index==0) & (bait >= Tc)] - Pb, axis=0)
        weekday_heating_hour_ratio = np.mean(hourly_load[(weekday_index==1) & (bait <= Th)] - Pb, axis=0)
        weekend_heating_hour_ratio = np.mean(hourly_load[(weekday_index==0) & (bait <= Th)] - Pb, axis=0)
        demand_curve['weekday_cooling_hour_ratio'] = weekday_cooling_hour_ratio / weekday_cooling_hour_ratio.sum()
        demand_curve['weekend_cooling_hour_ratio'] = weekend_cooling_hour_ratio / weekend_cooling_hour_ratio.sum()
        demand_curve['weekday_heating_hour_ratio'] = weekday_heating_hour_ratio / weekday_heating_hour_ratio.sum()
        demand_curve['weekend_heating_hour_ratio'] = weekend_heating_hour_ratio / weekend_heating_hour_ratio.sum()
        np.save(f'code/models/demand_curve/{selected_country}_demand_curve', demand_curve, allow_pickle=True)

  
    plt.figure(figsize=[6,5])
    blue = (114/255,169/255,208/255)
    lblue = (186/255,200/255,229/255)
    orange = (245/255,179/255,120/255)
    lorange = (253/255,220/255,186/255)
    green = (142/255,198/255,194/255)
    yellow = (234/255, 190/255, 43/255)
    red = (232/255,57/255,71/255)
    lred = (249/255,192/255,175/255)

    weekday_bait = bait[weekday_index==1]
    weekend_bait = bait[weekday_index==0]
    demand = Pb + Ph * np.maximum(Th - bait, 0)  + Pc * np.maximum(bait - Tc, 0) + alpha * weekday_index
    plt.scatter(weekday_bait, load[weekday_index==1], marker='o', c=lred, linewidth=0, alpha=0.8,s=10)
    plt.scatter(weekend_bait, load[weekday_index==0], marker='s', c=lblue, linewidth=0, alpha=0.8,s=10)

    bait_simu = np.linspace(bait.min(), bait.max(),1000)
    bait_simu = np.linspace(bait.min(),bait.max(),1000)
    work_demand = Pb + Ph * np.maximum(Th - bait_simu, 0)  + Pc * np.maximum(bait_simu - Tc, 0) + alpha * 1
    week_demand = Pb + Ph * np.maximum(Th - bait_simu, 0)  + Pc * np.maximum(bait_simu - Tc, 0) + alpha * 0
    plt.scatter(bait_simu, work_demand, marker='.', alpha=0.5, c=red, s=10)
    plt.scatter(bait_simu, week_demand, marker='.', alpha=0.5, c=blue, s=10)


    plt.scatter([],[], marker='o', c=lred, linewidth=0, alpha=0.99, label='weekday demand')
    plt.plot([],[], alpha=0.8, c=red, linewidth=2, label='weekday prediction')

    plt.scatter([],[], marker='s', c=lblue, linewidth=0, alpha=0.99, label='weekend demand')
    plt.plot([],[], alpha=0.8, c=blue, linewidth=2, label='weekend prediction')

    weekday_demand_pred = Pb + Ph * np.maximum(Th - weekday_bait, 0)  + Pc * np.maximum(weekday_bait - Tc, 0) + alpha * 1
    weekend_demand_pred = Pb + Ph * np.maximum(Th - weekend_bait, 0)  + Pc * np.maximum(weekend_bait - Tc, 0) + alpha * 0
    pred_load = np.zeros(load.shape)
    pred_load[weekday_index==1] = weekday_demand_pred
    pred_load[weekday_index==0] = weekend_demand_pred
    print(compute_mape(load[weekday_index==1], weekday_demand_pred), compute_rmse(load[weekday_index==1], weekday_demand_pred))
    print(compute_mape(load[weekday_index==0], weekend_demand_pred), compute_rmse(load[weekday_index==0], weekend_demand_pred))
    print(compute_mape(load, pred_load), compute_rmse(load, pred_load))
    plt.text(bait.min(), load.min(), 
            f'RMSE = {compute_rmse(load, pred_load):.1f} GW\nMAPE = {compute_mape(load, pred_load)*100:.1f}%', 
            style='italic',
            bbox={'facecolor': lblue, 'alpha': 0.1, 'pad': 10})


    country_name = {'BE': 'Belgium', 'IT': 'Italy', 'ES': 'Spain', 'GB': 'UK', 'FR': 'France', 'DE': 'Germany'}
    plt.legend(ncol=2, loc=9, fontsize=12)
    plt.title(f'Demand Profile in {country_name[selected_country]}', fontsize=16)
    plt.ylabel('Daily Demand (GW)', fontsize=16)
    plt.xlabel('BAIT (°C)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'code/models/demand_curve/{selected_country}_calibrated_model_{year_list}.png', dpi=300)