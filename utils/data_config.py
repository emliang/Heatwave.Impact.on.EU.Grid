import numpy as np

### Demand model parameters is from Supplementary of 
### "Staffell, I., Pfenninger, S., & Johnson, N. (2023). A global model of hourly space heating and cooling demand at multiple spatial scales. 
### Nature Energy, 8(12), 1328-1344."
demand_model_para = {
'AT': {'heating_threshold': 10.67, 'cooling_threshold': 24.83, 
    'p_heating': 0.12 , 'p_cooling': 0.30,  # GW/(oc)/day-ave 
    },
'BE': {'heating_threshold': 11.28, 'cooling_threshold': 24.63, 
       'p_heating': 0.14 , 'p_cooling': 0.43,  # GW/(oc)/day-ave 
       },
'BG': {'heating_threshold': 13.52, 'cooling_threshold': 21.62, 
       'p_heating': 0.12 , 'p_cooling': 0.10,  # GW/(oc)/day-ave 
       },
'CZ': {'heating_threshold': 14.18, 'cooling_threshold': 100, 
       'p_heating': 0.10 , 'p_cooling': 0,  # GW/(oc)/day-ave 
       },
'DK': {'heating_threshold': 11.01, 'cooling_threshold': 22.70, 
       'p_heating': 0.06 , 'p_cooling': 0.21,  # GW/(oc)/day-ave 
       },
'FR': {'heating_threshold': 12.98, 'cooling_threshold': 22.69, 
       'p_heating': 2.37 , 'p_cooling': 1.59,  # GW/(oc)/day-ave 
       },
'GB': {'heating_threshold': 12.19, 'cooling_threshold': 20.91, 
       'p_heating': 0.83 , 'p_cooling': 1.36,  # GW/(oc)/day-ave 
       },
'DE': {'heating_threshold': 11.39, 'cooling_threshold': 23.06,
       'p_heating': 0.51 , 'p_cooling': 0.55,  # GW/(oc)/day-ave 
       },
'GR': {'heating_threshold': 13.74, 'cooling_threshold': 22.59,
       'p_heating': 0.18 , 'p_cooling': 0.35,  # GW/(oc)/day-ave 
       },
'HU': {'heating_threshold': 13.54, 'cooling_threshold': 24.58,
       'p_heating': 0.05 , 'p_cooling': 0.35,  # GW/(oc)/day-ave 
       },
'IT': {'heating_threshold': 12.01, 'cooling_threshold': 20.56, 
       'p_heating': 0.62 , 'p_cooling': 1.46,  # GW/(oc)/day-ave 
       },
'PL': {'heating_threshold': 13.01, 'cooling_threshold': 23.86, 
       'p_heating': 0.17 , 'p_cooling': 0.82,  # GW/(oc)/day-ave 
       },
'PT': {'heating_threshold': 13.39, 'cooling_threshold': 21.74, 
       'p_heating': 0.18 , 'p_cooling': 0.34,  # GW/(oc)/day-ave 
       },
'ES': {'heating_threshold': 14.67, 'cooling_threshold': 19.82, 
       'p_heating': 0.65 , 'p_cooling': 0.88,  # GW/(oc)/day-ave 
       },
'CH': {'heating_threshold': 12.54, 'cooling_threshold': 25.84, 
       'p_heating': 0.10 , 'p_cooling': 0.17,  # GW/(oc)/day-ave 
       },
'RO': {'heating_threshold': 15.68, 'cooling_threshold': 22.16, 
       'p_heating': 0.09 , 'p_cooling': 0.20,  # GW/(oc)/day-ave 
       },
'SK': {'heating_threshold': 11.98, 'cooling_threshold': 24.95, 
       'p_heating': 0.04 , 'p_cooling': 0.08,  # GW/(oc)/day-ave 
       },
}


holiday_list = {
    'BE': {
        2015: [
            [2015, 1, 1],  # New Year's Day
            [2015, 4, 3],  # Good Friday
            [2015, 4, 5],  # Easter Sunday
            [2015, 4, 6],  # Easter Monday
            [2015, 5, 1],  # Labor Day
            [2015, 5, 14], # Ascension Day
            [2015, 5, 24], # Pentecost Sunday
            [2015, 5, 25], # Pentecost Monday
            [2015, 7, 21], # Belgian National Day
            [2015, 8, 15], # Assumption of Mary
            [2015, 11, 1], # All Saints' Day
            [2015, 11, 11],# Armistice Day
            [2015, 12, 25] # Christmas Day
        ],
        2016: [
            [2016, 1, 1],  # New Year's Day
            [2016, 3, 25], # Good Friday
            [2016, 3, 27], # Easter Sunday
            [2016, 3, 28], # Easter Monday
            [2016, 5, 1],  # Labor Day
            [2016, 5, 5],  # Ascension Day
            [2016, 5, 15], # Pentecost Sunday
            [2016, 5, 16], # Pentecost Monday
            [2016, 7, 21], # Belgian National Day
            [2016, 8, 15], # Assumption of Mary
            [2016, 11, 1], # All Saints' Day
            [2016, 11, 11],# Armistice Day
            [2016, 12, 25] # Christmas Day
        ],
        2017: [
            [2017, 1, 1],  # New Year's Day
            [2017, 4, 14], # Good Friday
            [2017, 4, 16], # Easter Sunday
            [2017, 4, 17], # Easter Monday
            [2017, 5, 1],  # Labor Day
            [2017, 5, 25], # Ascension Day
            [2017, 6, 4],  # Pentecost Sunday
            [2017, 6, 5],  # Pentecost Monday
            [2017, 7, 21], # Belgian National Day
            [2017, 8, 15], # Assumption of Mary
            [2017, 11, 1], # All Saints' Day
            [2017, 11, 11],# Armistice Day
            [2017, 12, 25] # Christmas Day
        ],
        2018: [
            [2018, 1, 1],  # New Year's Day
            [2018, 3, 30], # Good Friday
            [2018, 4, 1],  # Easter Sunday
            [2018, 4, 2],  # Easter Monday
            [2018, 5, 1],  # Labor Day
            [2018, 5, 10], # Ascension Day
            [2018, 5, 20], # Pentecost Sunday
            [2018, 5, 21], # Pentecost Monday
            [2018, 7, 21], # Belgian National Day
            [2018, 8, 15], # Assumption of Mary
            [2018, 11, 1], # All Saints' Day
            [2018, 11, 11],# Armistice Day
            [2018, 12, 25] # Christmas Day
        ],
        2019: [
            [2019, 1, 1],  # New Year's Day
            [2019, 4, 19], # Good Friday
            [2019, 4, 21], # Easter Sunday
            [2019, 4, 22], # Easter Monday
            [2019, 5, 1],  # Labor Day
            [2019, 5, 30], # Ascension Day
            [2019, 6, 9],  # Pentecost Sunday
            [2019, 6, 10], # Pentecost Monday
            [2019, 7, 21], # Belgian National Day
            [2019, 8, 15], # Assumption of Mary
            [2019, 11, 1], # All Saints' Day
            [2019, 11, 11],# Armistice Day
            [2019, 12, 25] # Christmas Day
        ]
    },
    'FR': {
        2015: [
            [2015, 1, 1],  # New Year's Day
            [2015, 4, 6],  # Easter Monday
            [2015, 5, 1],  # Labor Day
            [2015, 5, 8],  # Victory in Europe Day
            [2015, 5, 14], # Ascension Day
            [2015, 5, 25], # Whit Monday
            [2015, 7, 14], # Bastille Day
            [2015, 8, 15], # Assumption of Mary
            [2015, 11, 1], # All Saints' Day
            [2015, 11, 11],# Armistice Day
            [2015, 12, 25] # Christmas Day
        ],
        2016: [
            [2016, 1, 1],  # New Year's Day
            [2016, 3, 28], # Easter Monday
            [2016, 5, 1],  # Labor Day
            [2016, 5, 5],  # Ascension Day
            [2016, 5, 8],  # Victory in Europe Day
            [2016, 5, 16], # Whit Monday
            [2016, 7, 14], # Bastille Day
            [2016, 8, 15], # Assumption of Mary
            [2016, 11, 1], # All Saints' Day
            [2016, 11, 11],# Armistice Day
            [2016, 12, 25] # Christmas Day
        ],
        2017: [
            [2017, 1, 1],  # New Year's Day
            [2017, 4, 17], # Easter Monday
            [2017, 5, 1],  # Labor Day
            [2017, 5, 8],  # Victory in Europe Day
            [2017, 5, 25], # Ascension Day
            [2017, 6, 5],  # Whit Monday
            [2017, 7, 14], # Bastille Day
            [2017, 8, 15], # Assumption of Mary
            [2017, 11, 1], # All Saints' Day
            [2017, 11, 11],# Armistice Day
            [2017, 12, 25] # Christmas Day
        ],
        2018: [
            [2018, 1, 1],  # New Year's Day
            [2018, 4, 2],  # Easter Monday
            [2018, 5, 1],  # Labor Day
            [2018, 5, 8],  # Victory in Europe Day
            [2018, 5, 10], # Ascension Day
            [2018, 5, 21], # Whit Monday
            [2018, 7, 14], # Bastille Day
            [2018, 8, 15], # Assumption of Mary
            [2018, 11, 1], # All Saints' Day
            [2018, 11, 11],# Armistice Day
            [2018, 12, 25] # Christmas Day
        ],
        2019: [
            [2019, 1, 1],  # New Year's Day
            [2019, 4, 22], # Easter Monday
            [2019, 5, 1],  # Labor Day
            [2019, 5, 8],  # Victory in Europe Day
            [2019, 5, 30], # Ascension Day
            [2019, 6, 10], # Whit Monday
            [2019, 7, 14], # Bastille Day
            [2019, 8, 15], # Assumption of Mary
            [2019, 11, 1], # All Saints' Day
            [2019, 11, 11],# Armistice Day
            [2019, 12, 25] # Christmas Day
        ]
    },
    'IT': {
       2015: [
       [2015, 1, 1],   # New Year's Day
       [2015, 1, 6],   # Epiphany
       [2015, 4, 6],   # Easter Monday
       [2015, 4, 25],  # Liberation Day
       [2015, 5, 1],   # Labor Day
       [2015, 6, 2],   # Republic Day
       [2015, 8, 15],  # Assumption of Mary
       [2015, 11, 1],  # All Saints' Day
       [2015, 12, 8],  # Immaculate Conception
       [2015, 12, 25], # Christmas Day
       [2015, 12, 26]  # St. Stephen's Day
       ],
       2016: [
       [2016, 1, 1],   # New Year's Day
       [2016, 1, 6],   # Epiphany
       [2016, 3, 28],  # Easter Monday
       [2016, 4, 25],  # Liberation Day
       [2016, 5, 1],   # Labor Day
       [2016, 6, 2],   # Republic Day
       [2016, 8, 15],  # Assumption of Mary
       [2016, 11, 1],  # All Saints' Day
       [2016, 12, 8],  # Immaculate Conception
       [2016, 12, 25], # Christmas Day
       [2016, 12, 26]  # St. Stephen's Day
       ],
       2017: [
       [2017, 1, 1],   # New Year's Day
       [2017, 1, 6],   # Epiphany
       [2017, 4, 17],  # Easter Monday
       [2017, 4, 25],  # Liberation Day
       [2017, 5, 1],   # Labor Day
       [2017, 6, 2],   # Republic Day
       [2017, 8, 15],  # Assumption of Mary
       [2017, 11, 1],  # All Saints' Day
       [2017, 12, 8],  # Immaculate Conception
       [2017, 12, 25], # Christmas Day
       [2017, 12, 26]  # St. Stephen's Day
       ],
       2018: [
       [2018, 1, 1],   # New Year's Day
       [2018, 1, 6],   # Epiphany
       [2018, 4, 2],   # Easter Monday
       [2018, 4, 25],  # Liberation Day
       [2018, 5, 1],   # Labor Day
       [2018, 6, 2],   # Republic Day
       [2018, 8, 15],  # Assumption of Mary
       [2018, 11, 1],  # All Saints' Day
       [2018, 12, 8],  # Immaculate Conception
       [2018, 12, 25], # Christmas Day
       [2018, 12, 26]  # St. Stephen's Day
       ],
       2019: [
       [2019, 1, 1],   # New Year's Day
       [2019, 1, 6],   # Epiphany
       [2019, 4, 22],  # Easter Monday
       [2019, 4, 25],  # Liberation Day
       [2019, 5, 1],   # Labor Day
       [2019, 6, 2],   # Republic Day
       [2019, 8, 15],  # Assumption of Mary
       [2019, 11, 1],  # All Saints' Day
       [2019, 12, 8],  # Immaculate Conception
       [2019, 12, 25], # Christmas Day
       [2019, 12, 26]  # St. Stephen's Day
       ]
    },
    'ES': {
        2015: [
            [2015, 1, 1],   # New Year's Day
            [2015, 1, 6],   # Epiphany
            [2015, 4, 3],   # Good Friday
            [2015, 5, 1],   # Labor Day
            [2015, 8, 15],  # Assumption of Mary
            [2015, 10, 12], # National Day of Spain
            [2015, 11, 1],  # All Saints' Day
            [2015, 12, 6],  # Constitution Day
            [2015, 12, 8],  # Immaculate Conception
            [2015, 12, 25]  # Christmas Day
        ],
        2016: [
            [2016, 1, 1],   # New Year's Day
            [2016, 1, 6],   # Epiphany
            [2016, 3, 25],  # Good Friday
            [2016, 5, 1],   # Labor Day
            [2016, 8, 15],  # Assumption of Mary
            [2016, 10, 12], # National Day of Spain
            [2016, 11, 1],  # All Saints' Day
            [2016, 12, 6],  # Constitution Day
            [2016, 12, 8],  # Immaculate Conception
            [2016, 12, 25]  # Christmas Day
        ],
        2017: [
            [2017, 1, 1],   # New Year's Day
            [2017, 1, 6],   # Epiphany
            [2017, 4, 14],  # Good Friday
            [2017, 5, 1],   # Labor Day
            [2017, 8, 15],  # Assumption of Mary
            [2017, 10, 12], # National Day of Spain
            [2017, 11, 1],  # All Saints' Day
            [2017, 12, 6],  # Constitution Day
            [2017, 12, 8],  # Immaculate Conception
            [2017, 12, 25]  # Christmas Day
        ],
        2018: [
            [2018, 1, 1],   # New Year's Day
            [2018, 1, 6],   # Epiphany
            [2018, 3, 30],  # Good Friday
            [2018, 5, 1],   # Labor Day
            [2018, 8, 15],  # Assumption of Mary
            [2018, 10, 12], # National Day of Spain
            [2018, 11, 1],  # All Saints' Day
            [2018, 12, 6],  # Constitution Day
            [2018, 12, 8],  # Immaculate Conception
            [2018, 12, 25]  # Christmas Day
        ],
        2019: [
            [2019, 1, 1],   # New Year's Day
            [2019, 1, 6],   # Epiphany
            [2019, 4, 19],  # Good Friday
            [2019, 5, 1],   # Labor Day
            [2019, 8, 15],  # Assumption of Mary
            [2019, 10, 12], # National Day of Spain
            [2019, 11, 1],  # All Saints' Day
            [2019, 12, 6],  # Constitution Day
            [2019, 12, 8],  # Immaculate Conception
            [2019, 12, 25]  # Christmas Day
        ]
    },
    'GB': {
        2015: [
            [2015, 1, 1],   # New Year's Day
            [2015, 4, 3],   # Good Friday
            [2015, 4, 6],   # Easter Monday (England, Wales, Northern Ireland)
            [2015, 5, 4],   # Early May Bank Holiday
            [2015, 5, 25],  # Spring Bank Holiday
            [2015, 8, 31],  # Summer Bank Holiday (England, Wales, Northern Ireland)
            [2015, 12, 25], # Christmas Day
            [2015, 12, 28]  # Boxing Day (substitute day)
        ],
        2016: [
            [2016, 1, 1],   # New Year's Day
            [2016, 3, 25],  # Good Friday
            [2016, 3, 28],  # Easter Monday (England, Wales, Northern Ireland)
            [2016, 5, 2],   # Early May Bank Holiday
            [2016, 5, 30],  # Spring Bank Holiday
            [2016, 8, 29],  # Summer Bank Holiday (England, Wales, Northern Ireland)
            [2016, 12, 26], # Boxing Day
            [2016, 12, 27]  # Christmas Day (substitute day)
        ],
        2017: [
            [2017, 1, 2],   # New Year's Day (substitute day)
            [2017, 4, 14],  # Good Friday
            [2017, 4, 17],  # Easter Monday (England, Wales, Northern Ireland)
            [2017, 5, 1],   # Early May Bank Holiday
            [2017, 5, 29],  # Spring Bank Holiday
            [2017, 8, 28],  # Summer Bank Holiday (England, Wales, Northern Ireland)
            [2017, 12, 25], # Christmas Day
            [2017, 12, 26]  # Boxing Day
        ],
        2018: [
            [2018, 1, 1],   # New Year's Day
            [2018, 3, 30],  # Good Friday
            [2018, 4, 2],   # Easter Monday (England, Wales, Northern Ireland)
            [2018, 5, 7],   # Early May Bank Holiday
            [2018, 5, 28],  # Spring Bank Holiday
            [2018, 8, 27],  # Summer Bank Holiday (England, Wales, Northern Ireland)
            [2018, 12, 25], # Christmas Day
            [2018, 12, 26]  # Boxing Day
        ],
        2019: [
            [2019, 1, 1],   # New Year's Day
            [2019, 4, 19],  # Good Friday
            [2019, 4, 22],  # Easter Monday (England, Wales, Northern Ireland)
            [2019, 5, 6],   # Early May Bank Holiday
            [2019, 5, 27],  # Spring Bank Holiday
            [2019, 8, 26],  # Summer Bank Holiday (England, Wales, Northern Ireland)
            [2019, 12, 25], # Christmas Day
            [2019, 12, 26]  # Boxing Day
        ]
    },
    'DE': {
        2015: [
            [2015, 1, 1],   # New Year's Day
            [2015, 4, 3],   # Good Friday
            [2015, 4, 6],   # Easter Monday
            [2015, 5, 1],   # Labor Day
            [2015, 5, 14],  # Ascension Day
            [2015, 5, 25],  # Whit Monday
            [2015, 10, 3],  # German Unity Day
            [2015, 12, 25], # Christmas Day
            [2015, 12, 26]  # Second Day of Christmas
        ],
        2016: [
            [2016, 1, 1],   # New Year's Day
            [2016, 3, 25],  # Good Friday
            [2016, 3, 28],  # Easter Monday
            [2016, 5, 1],   # Labor Day
            [2016, 5, 5],   # Ascension Day
            [2016, 5, 16],  # Whit Monday
            [2016, 10, 3],  # German Unity Day
            [2016, 12, 25], # Christmas Day
            [2016, 12, 26]  # Second Day of Christmas
        ],
        2017: [
            [2017, 1, 1],   # New Year's Day
            [2017, 4, 14],  # Good Friday
            [2017, 4, 17],  # Easter Monday
            [2017, 5, 1],   # Labor Day
            [2017, 5, 25],  # Ascension Day
            [2017, 6, 5],   # Whit Monday
            [2017, 10, 3],  # German Unity Day
            [2017, 12, 25], # Christmas Day
            [2017, 12, 26]  # Second Day of Christmas
        ],
        2018: [
            [2018, 1, 1],   # New Year's Day
            [2018, 3, 30],  # Good Friday
            [2018, 4, 2],   # Easter Monday
            [2018, 5, 1],   # Labor Day
            [2018, 5, 10],  # Ascension Day
            [2018, 5, 21],  # Whit Monday
            [2018, 10, 3],  # German Unity Day
            [2018, 12, 25], # Christmas Day
            [2018, 12, 26]  # Second Day of Christmas
        ],
        2019: [
            [2019, 1, 1],   # New Year's Day
            [2019, 4, 19],  # Good Friday
            [2019, 4, 22],  # Easter Monday
            [2019, 5, 1],   # Labor Day
            [2019, 5, 30],  # Ascension Day
            [2019, 6, 10],  # Whit Monday
            [2019, 10, 3],  # German Unity Day
            [2019, 12, 25], # Christmas Day
            [2019, 12, 26]  # Second Day of Christmas
        ]
    }
}



ieee_conductor_config = {'diameter': 28.1 / 1e3,
                         'ref_temperature': 25,
                         'max_temperature': 90,
                         'resistance_ratio': 0.00429,
                         'unit_resistance': 7.283 * 1e-5,
                         }


blue = (114/255,169/255,208/255)
lblue = (186/255,200/255,229/255)
orange = (245/255,179/255,120/255)
lorange = (253/255,220/255,186/255)
green = (142/255,198/255,194/255)
yellow = (234/255, 190/255, 43/255)
red = (232/255,57/255,71/255)
lred = (249/255,192/255,175/255)