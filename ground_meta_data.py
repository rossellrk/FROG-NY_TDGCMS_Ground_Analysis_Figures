# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:01:36 2025
Averaging weather and trace gas data to VOC sampling blocks
@author: rossellr
"""
# module imports
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import matplotlib.dates as md
from matplotlib.dates import DateFormatter
import datetime as datetime
from datetime import timedelta
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import pandas as pd

import statistics as stat
import math as mat

from collections import defaultdict
# import act
# from act.io.icartt import read_icartt

today = datetime.date.today().strftime('%Y%m%d')



# files to import: weather data, o3 data, nox data, org aerosol data, and ground start/stop times (+ sample names)
weather_file = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\full_weather_data_20230705_20230814.csv'
ground_o3_file = pd.read_csv(r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\Field Campaign Data\Final Data\O3_FinalTs_20230321.csv')
ground_nox_file = pd.read_csv(r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\Field Campaign Data\Final Data\icartt_frogNox_final20230322.csv')
ground_start_stop = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\meta_data\ground_sampling_times.csv'
hotspot_start_stop = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\meta_data\HotSpot_sampling_times.csv'
oa_file = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\meta_data\FROG_ACSM_NR-PM1.csv'

# path for exported file:
save_path = r'C:\\Users\\rossellr\\OneDrive - Colostate\\Desktop\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\ground_data_workup\\meta_data\\'
# name for exported file:   
hs_file_name = 'HotSpot_meta_data.csv'
ground_file_name = 'ground_meta_data.csv'

# Reading in the weather metadata file
col_names_weather = ['dt', 'temp_c', 'wind_speed_m/s', 'wind_dir_deg', 'hourly_rain_in/s', # naming columns
             'relative_press_inHg', 'solar_rad_W/m2']
weather = pd.read_csv(weather_file, header = None, usecols = [1, 2, 5, 8, 9, 15, 18], names = col_names_weather)
weather.drop(weather.index[0], inplace = True) # drop first row with additional header info
weather['temp_c'].dropna(axis = 0) # remove NA values
weather['dt'] = pd.to_datetime(weather['dt']) # transform into python datetime objects
weather['temp_c'] = [int(float(x)) for x in weather['temp_c']]
weather['relative_press_inHg'] = pd.to_numeric(weather['relative_press_inHg'], errors = 'coerce') # remove string values
weather['solar_rad_W/m2'] = pd.to_numeric(weather['solar_rad_W/m2'], errors = 'coerce')
weather['wind_speed_m/s'] = pd.to_numeric(weather['wind_speed_m/s'], errors = 'coerce')
weather['hourly_rain_in/s'] = pd.to_numeric(weather['hourly_rain_in/s'], errors = 'coerce')
weather['wind_dir_deg'] = pd.to_numeric(weather['wind_dir_deg'], errors = 'coerce')

df = weather.copy()



# ground trace gas data
ground_o3 = pd.DataFrame(ground_o3_file)
ground_o3.drop(columns = ['MidPoint_Times', 'O3_Raw', 'Start_Times_UTC', 'Stop_Times'])
ground_o3['Start_Times'] = pd.to_datetime(ground_o3['Start_Times']) # transform into python datetime objects
ground_o3.rename(columns = {'Start_Times' : 'dt'}, inplace = True) # rename to match code chunks
o3_mask = ground_o3['O3_Corrected'] == 999 # remove flagged values
ground_o3 = ground_o3[~o3_mask]

ground_nox = pd.DataFrame(ground_nox_file)
ground_nox.drop(columns = ['NO_raw', 'NO_Corrected', 'NO2_raw', 'NO2_Corrected', 'NOx_raw',
                          'NOx_Corrected', 'Start_Times_UTC', 'Stop_Times_EDT'], inplace = True)
ground_nox['Start_Times_EDT'] = pd.to_datetime(ground_nox['Start_Times_EDT']) # transform into python datetime objects
ground_nox.rename(columns = {'Start_Times_EDT' : 'dt'}, inplace = True) # rename to match code chunks
nox_mask = ground_nox['NO_Corrected_Flagged'] == 9999.999999 # remove flagged values
ground_nox = ground_nox[~nox_mask]


# ACSM OA data
# reading in ACSM OA Data
col_names = ['datetime_local', 'total_ug_m3', 'OA_ug_m3', 'no3_ug_m3', 'nh4_ug_m3', 'so4_ug_m3', 'chl_ug_m3']
oa_data = pd.read_csv(oa_file, skiprows = [0], 
                      names = col_names, parse_dates = ['datetime_local'])


# definining sampling timeblocks
block_avgs = weather.copy()

def time_blocks_func(dt): # applying function to the dt column within the dataframe
    time_blocks = [
        ('06:30', '10:30'), # creating the four groupings
        ('10:20', '14:30'),
        ('14:20', '18:30'),
        ('18:00', '22:00')]
    time_str = dt.strftime('%H:%M') # keeping everything in time
    for start_time, end_time in time_blocks: 
        if start_time <= time_str < end_time:
            return f'{start_time} - {end_time}'


block_avgs['time_block'] = block_avgs['dt'].apply(time_blocks_func)
time_block_avgs = block_avgs.groupby('time_block')['temp_c'].mean().reset_index()
time_block_avgs.rename(columns = {'temp_c' : 'campaign_avg_temp_c'}, inplace = True)

time_block_avgs.insert(2, 'block_start_time', ['06:30', '10:20', '14:20', '18:00'])
time_block_avgs['block_start_time'] = pd.to_datetime(time_block_avgs['block_start_time'])
time_block_avgs['block_start_time'] = time_block_avgs['block_start_time'].dt.time
time_block_avgs.insert(3, 'block_end_time', ['10:30', '14:30', '18:30', '22:00'])
time_block_avgs['block_end_time'] = pd.to_datetime(time_block_avgs['block_end_time'])
time_block_avgs['block_end_time'] = time_block_avgs['block_end_time'].dt.time


 # reading in start/stop times
ground_start_stop = pd.read_csv(ground_start_stop)
cols = ground_start_stop.columns[1:2]
ground_start_stop[cols] = ground_start_stop[cols].apply(pd.to_datetime)

hs_start_stop = pd.read_csv(hotspot_start_stop)
cols = hs_start_stop.columns[3:4]
hs_start_stop[cols] = hs_start_stop[cols].apply(pd.to_datetime)
hs_start_stop.rename(columns = {'Collection Start Time' : 'start_time', 'Collection End Time' : 'end_time', 
                                'Sample Code' : 'sample_name'}, inplace = True)
hs_start_stop.drop(columns = 'Tube Number', inplace = True)


# # CO data from TM
# file = r'C:\\Users\\rossellr\\OneDrive - Colostate\\Desktop\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\meta_data\\FROGNY-LGR_Ground_20230711_R0_summercut.ict'
# ds = read_icartt(file)
# ds.data_vars # looking at variable names 
# data = {
#         'datetime' : ds['Start_EDT'], # creating a dictionary of variables from icartt 
#         'co_ppb' : ds['CO_ppb'],
#         'co_inlet_flag' : ds['flag_inlet']
#         }
# co_data = pd.DataFrame(data) # dataframe that bitch
# co_start_day = pd.to_datetime('2023-07-11') # icartt data is in DOY past start date
# co_data['datetime'] = co_data['datetime'].apply(lambda x: co_start_day + timedelta(seconds = x)) # turning into dt 



# defining functions
# taking values based on sample start/stop times from full weather csv file

def calc_meta_data(start_time, stop_time):
    # filtering weather data to match the time blocks
    filtered_weather = df[(df['dt'] >= start_time) & (df['dt'] <= stop_time)].reset_index(drop = True) 
    # getting medians or averages of variables of interest
    median_temp = filtered_weather['temp_c'].median()
    avg_temp = filtered_weather['temp_c'].mean()
    max_temp = filtered_weather['temp_c'].max()
    avg_press = filtered_weather['relative_press_inHg'].mean()
    median_SolarRad = filtered_weather['solar_rad_W/m2'].median()
    avg_WindSpeed = filtered_weather['wind_speed_m/s'].mean()
    median_WindDir = filtered_weather['wind_dir_deg'].median()
    avg_rain = filtered_weather['hourly_rain_in/s'].mean()
    # gotta do the same thing to the trace gas data
    filtered_o3 = ground_o3[(ground_o3['dt'] >= start_time) & (ground_o3['dt'] <= stop_time)].reset_index(drop = True)
    avg_o3 = filtered_o3['O3_Corrected'].mean()
    filtered_nox = ground_nox[(ground_nox['dt'] >= start_time) & 
                              (ground_nox['dt'] <= stop_time)].reset_index(drop = True)
    avg_no = filtered_nox['NO_Corrected_Flagged'].mean()
    avg_no2 = filtered_nox['NO2_Corrected_Flagged'].mean()
    # and the OA data now
    filtered_oa = oa_data[(oa_data['datetime_local'] >= start_time) & (oa_data['datetime_local'] <= stop_time)].reset_index(drop = True)
    avg_oa = filtered_oa['OA_ug_m3'].mean()
    avg_total_aerosol = filtered_oa['total_ug_m3'].mean()
    avg_no3_aerosol = filtered_oa['no3_ug_m3'].mean()
    avg_so4_aerosol = filtered_oa['so4_ug_m3'].mean()
    avg_nh4_aerosol = filtered_oa['nh4_ug_m3'].mean()
    
    return { # creating a dictionary of returned values
        'median_temp_c' : median_temp,
        'avg_temp_c' : avg_temp,
        'max_temp_c' : max_temp,
        'avg_press_inHg' : avg_press,
        'median_SolarRad_W/m2' : median_SolarRad,
        'avg_WindSpeed_m/s' : avg_WindSpeed,
        'median_WindDir' : median_WindDir,
        'avg_rain_in/s' : avg_rain,
        'avg_o3_ppb' : avg_o3,
        'avg_no_ppb' : avg_no,
        'avg_no2_ppb' : avg_no2,
        'avg_oa_ug_m3' : avg_oa,
        'avg_total_aerosol_ug_m3' : avg_total_aerosol,
        'avg_no3_ug_m3' : avg_no3_aerosol,
        'avg_so4_ug_m3' : avg_so4_aerosol,
        'avg_nh4_ug_m3' : avg_nh4_aerosol
        }

# now, we need to apply the calculated functions based on our df, and add in the sampling and times to the dictionay
def meta_data_functions (row):
    start_time = row['start_time']
    stop_time = row['end_time']
    meta_data = calc_meta_data(start_time, stop_time)
    return {**meta_data, 
            'sample_name' : row['sample_name'],
            'start_time' : row['start_time'],
            'end_time' : row['end_time']}



# applying functions to ground data
meta_data_list = []
meta_data_list = ground_start_stop.apply(meta_data_functions, axis = 1).to_list() # apply functions to df
ground_meta_data = pd.DataFrame(meta_data_list)

cols_to_move = ground_meta_data.columns[-3:] # put the sample names and start/end times up front
cols_to_move = cols_to_move.to_list()
ground_meta_data = ground_meta_data[cols_to_move + [col for col in ground_meta_data.columns if col not in cols_to_move]]
ground_meta_data = ground_meta_data.sort_values('start_time')
ground_meta_data.drop(index = 5, inplace = True)

full_path = save_path + today + '_' + ground_file_name
#ground_meta_data.to_csv(full_path, index = False) # export as csv file



# applying functions to hotspot data
meta_data_list = []
meta_data_list = hs_start_stop.apply(meta_data_functions, axis = 1).to_list() # apply functions to df
hs_meta_data = pd.DataFrame(meta_data_list)

cols_to_move = hs_meta_data.columns[-3:] # put the sample names and start/end times up front
cols_to_move = cols_to_move.to_list()
hs_meta_data = hs_meta_data[cols_to_move + [col for col in hs_meta_data.columns if col not in cols_to_move]]
# need to sort by sample name
sorter = []
for i in range (1, 17):
    if i == 2 or i == 9 or i == 14:
        continue     
    sorter.append(f'FNY_HS{i}') # generating a new list of sample names, exclusing FB samples

hs_meta_data.set_index('sample_name', drop = False, inplace = True) # setting sample name as index
hs_meta_data = hs_meta_data.loc[sorter] # matching index with sorter list
hs_meta_data.drop(hs_meta_data.columns[0], axis = 1, inplace = True) # dropping sorter list
hs_meta_data.reset_index(inplace = True) # re-setting index so only have one column of sample names


full_path = save_path + hs_file_name
#hs_meta_data.to_csv(full_path, index = False) # export as csv file

# %% Weather Stats 

all_weather_stats = weather.describe()
block_stats = ground_meta_data.describe()


# %% Temp  SI Figure

fig, ax = plt.subplots(
    sharex = True,
    figsize = (8, 5),
    dpi = 300
    )
plt1 = ax.plot(
    weather['dt'], weather['temp_c'],
    label = 'Temperature (°C)',
    color = '#FFB400',
    linestyle = '--',
    linewidth = 1.5,
    zorder = 2
    )
ax.set(
       ylabel = 'Temperature (°C)',
       ylim = [0, 45],      
       xlim = [datetime.date(2023, 7, 12), datetime.date(2023, 8, 12)]
       )

ax.legend()

fig.autofmt_xdate()
ax.xaxis.set_major_formatter(md.DateFormatter('%D %H:%M'))


plt.show()




# %% Meta weather figure

fig, (ax1, ax3) = plt.subplots(
    ncols = 1,
    nrows = 2,
    sharex = True,
    figsize = (8, 5),
    dpi = 300
    )
plt1 = ax1.plot(
    weather['dt'], weather['temp_c'],
    label = 'Temperature (°C)',
    color = '#FFB400',
    linestyle = '--',
    linewidth = 1.5,
    zorder = 2
    )
ax1.set(
       ylim = [0, 45],      
       xlim = [datetime.date(2023, 7, 12), datetime.date(2023, 8, 12)]
       )
ax1.set_ylabel(
    'Temperature (°C)',
    color = '#A87703'
    )

ax2 = ax1.twinx()
plt2 = ax2.plot(
    ground_o3['dt'], ground_o3['O3_Corrected'],
    color = '#2F4B26',
    linewidth = 1.5,
    linestyle = '--',
    alpha = 0.6,
    label = 'Ozone (ppbv)',
    zorder = 1
    )
ax2.set(
        ylim = [0, 120],
        )
ax2.set_ylabel(
    'Ozone ($ppb_v$)',
    color = '#2C4423'
    )

lns = plt1 + plt2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, 
           loc = 'upper right',
           framealpha = 1,
           ncol = 2,
           frameon = False)



plt3 = ax3.plot(
    ground_nox['dt'], ground_nox['NO_Corrected_Flagged'],
    label = 'NO ($ppb_v$)',
    color = '#083D77',
    zorder = 1
    )
ax3.set(
        ylim = [0, 100]
        )
ax3.set_ylabel(
    'NO ($ppb_v$)',
    color = '#083D77'
    )

ax4 = ax3.twinx()
plt4 = ax4.plot(
    ground_nox['dt'], ground_nox['NO2_Corrected_Flagged'],
    label = '$NO_2$ ($ppb_v$)',
    color = '#878787',
    zorder = 2,
    alpha = 0.7
    )
ax4.set(
        ylim = [0, 75]
        )
ax4.set_ylabel(
    '$NO_2$ ($ppb_v$)',
    color = '#4B4848'
    )



lns = plt3 + plt4
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, 
           loc = 'upper right',
           framealpha = 1,
           ncol = 2,
           frameon = False)


fig.autofmt_xdate()
ax1.xaxis.set_major_formatter(md.DateFormatter('%D %H:%M'))

plt.show()

