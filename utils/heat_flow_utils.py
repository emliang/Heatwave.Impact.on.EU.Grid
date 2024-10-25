import numpy as np
import copy
import math 


Stefan_Boltzmann_constant = 5.67 * 1e-8 # W/m^2-K^4

def wind_speed_hight(wind0, hight0=10, hight1=50, Hellmann_Exponent=0.16):
    return wind0 * (hight1/hight0) ** Hellmann_Exponent

def wind_direction_factor(wind_angle=math.pi/2):
    return 1.194 - np.cos(wind_angle) + 0.194 * np.cos(2*wind_angle) + 0.368 * np.sin(2*wind_angle)

def Reynolds_number(diameter, wind_speed, air_density, air_viscosity):
    return diameter * wind_speed * air_density / air_viscosity

def dynamic_air_viscosity(Tfilm):
    # Tfilm = (tem_con + tem_air)/2
    mu = (1.458e-6 * (Tfilm + 273)**1.5) / (Tfilm + 383.4)  # Dynamic viscosity of air (13a)
    return mu

def dynamic_air_density(Tfilm, elevation, conductor_hight=50):
    H = elevation + conductor_hight
    # Tfilm = (tem_con + tem_air)/2
    rho = (1.293 - 1.525e-4 * H + 6.379e-9 * H**2) / (1 + 0.00367 * (Tfilm)) #  Dynamic density of air (14a)
    return rho

def dynamic_air_conductivity(Tfilm):
    # Tfilm = (tem_con + tem_air)/2
    k = ( 2.424e-2 + 7.477e-5 * (Tfilm) - 4.407e-9 * (Tfilm) ** 2 ) # Dynamic conductivity of air (15a)
    return k


def convective_cooling(tem_con, conductor, weather):
    tem_air = weather['air_temperature']
    wind_angle = weather['wind_angle']
    wind_speed = weather['wind_speed']
    diameter = conductor['diameter']
    conductor_angle = conductor['conductor_angle']
    wind_angle = weather['wind_angle']
    elevation = conductor['elevation']
    Tfilm = (tem_con + tem_air)/2
    air_conductivity = dynamic_air_conductivity(Tfilm)#weather['air_conductivity']
    air_density = dynamic_air_density(Tfilm, elevation)#weather['air_density']
    air_viscosity = dynamic_air_viscosity(Tfilm)#weather['air_viscosity']
    # from north (0), east (90), south (180), weast (270)
    anglediff = np.abs(wind_angle%180 - conductor_angle%180) 
    anglediff[anglediff>90] = 180 - anglediff[anglediff>90]
    Phi = np.deg2rad(anglediff)
    Kphi = wind_direction_factor(Phi)
    wind_speed = wind_speed_hight(wind_speed)

    Nre =  Reynolds_number(diameter, wind_speed, air_density, air_viscosity)
    hc_l = Kphi * air_conductivity * (1.01+1.35*Nre**0.52) * (tem_con - tem_air)
    hc_h = 0.754 * Kphi * air_conductivity * Nre**0.6 * (tem_con - tem_air)
    # return np.maximum(hc_l, hc_h)
    hc_0 = 3.645 * air_density ** 0.5 * diameter ** 0.75 * (tem_con - tem_air) ** 1.25
    # print(hc_0,tem_air)
    return np.maximum(np.maximum(hc_l, hc_h), hc_0)

def radiative_cooling(tem_con, conductor, weather):
    tem_air = weather['air_temperature']
    radiation_emissivity = weather['radiation_emissivity']
    diameter = conductor['diameter']
    return np.pi * Stefan_Boltzmann_constant * diameter * radiation_emissivity * ((tem_con+273)**4-(tem_air+273)**4)

def solar_heating(conductor, weather):
    solar_absorptivity = weather['solar_absorptivity']
    solar_heat_intensity = weather['solar_heat_intensity']
    diameter = conductor['diameter']
    return solar_absorptivity * diameter * solar_heat_intensity

def joule_heating(tem_con, current, conductor):
    tem_ref = conductor['ref_temperature']
    resistance_ratio = conductor['resistance_ratio']
    unit_resistance = conductor['unit_resistance']
    resistance_tem = td_resistance(unit_resistance, tem_con, tem_ref, resistance_ratio)
    return current**2 * resistance_tem 

def td_resistance(resistance, tem_con, tem_ref, resistance_ratio):
    return resistance * (1+(tem_con- tem_ref) * resistance_ratio)

def heat_surplus(tem_con, current, conductor, weather):
    Hj = joule_heating(tem_con, current, conductor)
    Hs = solar_heating(conductor, weather)
    Hc = convective_cooling(tem_con, conductor, weather)
    Hr = radiative_cooling(tem_con, conductor, weather)
    return Hj + Hs - (Hc + Hr)

def df_heat_surplus(tem_con, current, conductor, weather):
    delta = 1e-6
    return (heat_surplus(tem_con + delta, current, conductor, weather) - 
            heat_surplus(tem_con, current, conductor, weather)) / delta

def heat_banlance_equation(current, conductor, weather, alg='newton', tol=1e-3):
    tem_air = weather['air_temperature']
    tem_con = copy.deepcopy(tem_air)
    tem_max = 150
    if alg == 'bisection':
        tem_con = (tem_con+tem_max)/2
        delta_H = heat_surplus(tem_con, current, conductor, weather)
        tem_con_l = tem_air * np.ones(delta_H.shape)
        tem_con_u = 150 * np.ones(delta_H.shape)
        tem_con_l[delta_H >=0] = tem_con[delta_H >=0]
        tem_con_u[delta_H <0] = tem_con[delta_H <0]
        tem_con_prev = 0
        for _ in range(1000):
            tem_con = (tem_con_l+tem_con_u)/2
            delta_H = heat_surplus(tem_con, current, conductor, weather)
            tem_con_l[delta_H >=0] = tem_con[delta_H >=0]
            tem_con_u[delta_H <0] = tem_con[delta_H <0]
            if np.abs(tem_con_prev - tem_con).max()<=tol:
                break
            else:
                tem_con_prev = tem_con
    elif alg == 'newton':
        for _ in range(1000):
            delta = heat_surplus(tem_con, current, conductor, weather)/df_heat_surplus(tem_con, current, conductor, weather)
            tem_con = tem_con - delta
            if np.abs(delta).max()<=tol:
                break
    return tem_con



def coefficient_quadratic_approximation(conductor, weather):
    tem_air = weather['air_temperature']
    tem_max = conductor['max_temperature']
    solar_absorptivity = weather['solar_absorptivity']
    solar_heat_intensity = weather['solar_heat_intensity']
    diameter = conductor['diameter']
    tem_air = weather['air_temperature']
    radiation_emissivity = weather['radiation_emissivity']
    diameter = conductor['diameter']
    tem_ref = conductor['ref_temperature']
    resistance_ratio = conductor['resistance_ratio']
    unit_resistance = conductor['unit_resistance']
    wind_angle = weather['wind_angle']
    wind_speed = weather['wind_speed']
    air_density = weather['air_density']
    air_conductivity = weather['air_conductivity']
    resistance_ta = td_resistance(unit_resistance, tem_air, tem_ref, resistance_ratio)
    resistance_tmax = td_resistance(unit_resistance, tem_max, tem_ref, resistance_ratio)

    conductor_angle = conductor['conductor_angle']
    wind_angle = weather['wind_angle']
    anglediff = np.abs(wind_angle%180 - conductor_angle%180) 
    anglediff[anglediff>90] = 180 - anglediff[anglediff>90]
    Phi = np.deg2rad(anglediff)
    Kphi = wind_direction_factor(Phi)
    wind_speed = wind_speed_hight(wind_speed)


    Hc = np.maximum(3.07 /diameter * Kphi * (air_density*diameter*wind_speed)**0.471, 
             8.35/diameter * Kphi * (air_density*diameter*wind_speed)**0.8, )
    Hr0 = 4 * Stefan_Boltzmann_constant * radiation_emissivity * (tem_air + 273)**3
    k1 = 6 * Stefan_Boltzmann_constant * radiation_emissivity * (tem_air + 273)**2
    k2 = resistance_tmax / (np.pi*diameter*(Hc + Hr0 + k1 * (tem_max - tem_air)))
    b0 = (np.pi * diameter * (Hc +  Hr0))
    b1 = unit_resistance * resistance_ratio - np.pi * diameter * k1 * k2
    beta0 = tem_air + solar_absorptivity * diameter * solar_heat_intensity / b0
    beta1 = resistance_ta / b0
    beta2 = k2 * b1 / b0
    return beta0, beta1, beta2

def quadratic_heat_balance_approximation(current,  conductor, weather):
    """
    Tc = beta0 + beta1 I^2 + beta2 I^4
    """
    beta0, beta1, beta2 = coefficient_quadratic_approximation(conductor, weather)
    return beta0 + beta1 * current ** 2 + beta2 * current ** 4



def maximum_allowable_current(conductor, weather):
    tem_max = conductor['max_temperature']
    tem_ref = conductor['ref_temperature']
    resistance_ratio = conductor['resistance_ratio']
    unit_resistance = conductor['unit_resistance']
    resistance_tem = td_resistance(unit_resistance, tem_max, tem_ref, resistance_ratio)
    Hs = solar_heating(conductor, weather)
    Hc = convective_cooling(tem_max, conductor, weather)
    Hr = radiative_cooling(tem_max, conductor, weather)
    Hj = Hr + Hc - Hs
    Imax = (Hj / resistance_tem)**0.5
    return Imax


