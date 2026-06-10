# -*- coding: utf-8 -*-
"""
Created on 20260216
Checking the final runs + the 5-factor 100 run + rotations
@author: rossellr
"""

'''
Pulling and organizing factor solutions from the PMF output files, into organized lists that contain dataframes for each factor solution.
'''

# %% module imports

# module imports
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import numpy as np
import math as math
import scipy.stats as scipy
import statsmodels.api as sm
import matplotlib.dates as md
from matplotlib.dates import DateFormatter
import datetime as datetime
from datetime import timedelta
from windrose import WindroseAxes

from pathlib import Path
from matplotlib.lines import Line2D

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import time as time
from collections import defaultdict
import molmass
from molmass import Formula
import warnings
import random

import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import textwrap

# turning off "A value is trying to be set on a copy of a slice " warning b/c its annoying
pd.options.mode.chained_assignment = None  # default='warn' 

dt_format = '%Y-%m-%d %H:%M:%S'
today = datetime.date.today().strftime('%Y%m%d')

# %% functions

# function to read in most recent csv file given a folder path and a file end string
def find_updated_csv(folder_path, suffix):
    folder = Path(folder_path)

    # Find all CSV files that end with the given suffix
    matching_files = [
        f for f in folder.glob(f"*{suffix}")
        if f.suffix == ".csv"
    ]

    if not matching_files:
        raise FileNotFoundError(f"No CSV files found with suffix '{suffix}'")

    # Sort by filename (works because files start with date)
    most_recent_file = sorted(matching_files)[-1]

    print(f"Reading file: {most_recent_file.name}")
    return most_recent_file

# function to add totals to the top of each bar in the percentage bar plot
def add_percent_bar_totals(axis, x, y, total, units, fontsize, fontweight): 
    for i in range(len(x)):
        axis.text(x = i, y = y + (y * 0.02) ,  # setting x and y positions of labels - centering with bar and then slightly above top of bar
                 s = f'{total[i]:0.2f} {units}', # setting and formatting what appears in the label
                 ha = 'center',
                 fontsize = fontsize,
                 fontweight = fontweight
                 )

# function to add rectangular patches to TS graphs
def rect_box(start_dt, end_dt, axis, hatch):
    start = md.date2num(start_dt) # converting dt object to numerical object for axis coordinates
    end = md.date2num(end_dt)
    width = end - start # width of rectangle
    box = Rectangle(
        xy = (start, -0.5), # xy coordinates of bottom-left corner
        width = width,
        height = height,
        alpha = set_alpha,
        color = box_color,
        hatch = hatch
        )
    ax = axis
    ax.add_patch(box)
    
# function to add rectangular patches to non-TS graphs
def points_rect_box(start, end, axis, color):
    width = end - start # width of rectangle
    box = Rectangle(
        xy = (start, start_y_point), # xy coordinates of bottom-left corner
        width = width,
        height = height,
        alpha = set_alpha,
        color = color
        )
    ax = axis
    axis[ax].add_patch(box)
    
# function for a general TS plot
def general_ts_plot(chosen_df, axis, color, linestyle, mc):
    axis.plot(chosen_df.index, chosen_df[col], 
            label = col,
            linewidth = 2,
            marker = 'o',
            markeredgecolor = 'black',
            markerfacecolor = None,
            markersize = mc,
            color = color,
            linestyle = linestyle
            )
    axis.legend(loc = 'upper right',
                fontsize = 'large')
    axis.set_yticks(np.linspace(chosen_df[col].min() - (chosen_df[col].min() * 0.2), 
                                (chosen_df[col].max() + (chosen_df[col].max() * 0.2)), 3),
                    labels = np.linspace(chosen_df[col].min(), chosen_df[col].max(), 3),
                    fontsize = 13,
                    ) 
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# %% dictionaries of colors and linestyles

# setting colors for 4 factor solution to always be consistent:
color_dict = {
    'Factor 1' : '#840032',
    'Factor 4' : '#789180', 
    'Factor 2' : '#E49273',
    'Factor 3' : '#48d1cc',
    'Factor 4' : '#004F2D',
    'Factor 5' : '#91AEC1',
    'Factor 6' : '#772F1A',
    'Factor 7' : '#4B3B47',
    'total_oh' : 'darkmagenta',
    'Total OH Reactivity' : 'darkmagenta'
    }

linestyle_dict = {
    'Factor 1' : 'dashdot',
    'Factor 4' : 'solid', 
    'Factor 2' : 'dashdot',
    'Factor 3' : 'solid',
    'total_oh' : 'dashed',
    'Total OH Reactivity' : 'dashed'
    }

cmpd_colors_dict = { # setting color of cmpd based on source
    'toluene' : '#80798b', 
    'ethylbenzene' : '#727a91', 
    'decamethylcyclopentasiloxane' : '#ffcc75', 
    'benzyl acetate' : '#AA336A', 
    'allyl heptanoate' : '#e493a1', 
    'naphthalene' : '#5e7b93', 
    '2-methyl-undecanal' : '#e4cbd1', 
    'diphenyl ether' : '#ff859c', 
    'o-xylene' : '#708090', 
    'a-pinene' : '#559b39', 
    'benzaldehyde' : '#dc8e43', 
    '3-ethyl toluene' : '#899499', 
    'limonene' : '#41772c', 
    'benzyl alcohol' : '#F5477E',
    'camphor' : '#c44f69', 
    'methyl salicylate' : '#d577af', 
    'benzothiazole' : '#777793', 
    '4-tert-butylcyclohexyl acetate peak 1' : '#ff859c', 
    '4-tert-butylcyclohexyl acetate 1' : '#ff859c',
    '4-tert-butylcyclohexyl acetate peak 2' : '#ffe3ea', 
    '4-tert-butylcyclohexyl acetate 2' : '#ffe3ea', 
    'hexadecane' : '#715412',
    'dodecane' : '#795C34',
    'benzene' : '#766856',
    '1_3_5-trioxane peak 1' : '#6c541e',
    '1_3_5-trioxane 1' : '#6c541e',
    'hexamethylcyclotrisiloxane' : '#DAC17C',
    'tetrachloroethylene' : '#b6a999', 
    '4-chlorobenzotrifluoride' : '#708090', 
    'm-xylene' : '#9ba6b1',
    '1_3_5_7-tetroxane peak 1' : '#ba7dd2', 
    '1_3_5_7-tetroxane 1' : '#ba7dd2',
    'propylbenzene' : '#702b18', 
    'phenol' : '#7b4a43', 
    '1_3_5-trioxepane peak 2' : '#d17f66', 
    '1_3_5-trioxepane 2' : '#d17f66', 
    'acetophenone' : '#ffbdd3', 
    'nonanal' : '#6892ec',  
    '1_3_5_7-tetroxane peak 2' : '#d17f66', 
    '1_3_5_7-tetroxane 2' : '#d17f66',
    'nopinone' : '#665d1e' ,
    'sabina ketone' : '#cc8899',
    'decanal' : '#219ed6',
    'phenoxyacetone' : '#ba7dd2', 
    'phenoxyisopropanol' : '#ec5fbe',
    'bornyl acetate' : '#ba7dd2', 
    '2-tert-butylcyclohexyl acetate peak 1' : '#a63375', 
    '2-tert-butylcyclohexyl acetate 1' : '#a63375', 
    'methyl naphthalene peak 1' :  '#54424a', 
    'methylnaphthalene 1' :  '#54424a', 
    '6-tert-butyl-o-cresol' : '#ffb8f8', 
    'methyl naphthalene peak 2' : '#86727b', 
    'methylnaphthalene 2' : '#86727b', 
    'methylnapthalene 2' : '#86727b',
    'dodecamethylcyclohexasiloxane' : '#F9E076',
    'tetradecane' : '#532915', 
    'tricyclodecenyl acetate' : '#F33A6A', 
    '2-ethyl-1-hexanol' : '#D3D3D3',
    'ethylhexyl benzoate' : '#C9BB8E',
    'heptadecane' : '#412e33',
    'txib' : '#4a3e20',
    'other' : '#87ceeb',
    'total_oh' : 'darkmagenta',
    'Total OH Reactivity' : 'darkmagenta',
    'homosalate' : '#BE6E46'
    }
# %% reading in and organizing metadata


# reading in meta data of weather and trace gas data (pre-averaged by GC time blocks)
file_path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\meta_data\\'
file = '_ground_meta_data.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(file_path, file)
meta_data = pd.read_csv(updated_file, 
                        parse_dates = ['start_time', 'end_time'])

meta_data.drop(columns = ['sample_name', 'end_time'], inplace = True )
meta_data['hour'] = meta_data['start_time'].round('H')
meta_data['hour'] = meta_data['hour'].dt.hour
full_meta_data = meta_data.copy()

median_temp = meta_data['median_temp_c'].median()
avg_temp = meta_data['avg_temp_c'].mean()

# grouping meta data by hour of the day for future diel plots
grouped_metadata = meta_data.groupby(['hour']).median()
grouped_metadata.drop(columns = ['start_time'], inplace = True)

# reading in full weather data (not averaged by GC blocks)
weather_file = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\full_weather_data_20230705_20230814.csv'
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

# reading in ground cmpd data
path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\pmf\input\\'
file = '_cmpd_reference_sheet.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(file_path, file)
cmpd_refs = pd.read_csv(updated_file)

# ground compounds
path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\python\outputs\\'
file = '_groundVOCs_all_compounds_requant.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(file_path, file)
ground_cmpds_ppb = pd.read_csv(updated_file, parse_dates = ['start_time_edt'])

# grouping cmpd data by hour of the day for future diel plots
ground_cmpds_ppb['hour'] = ground_cmpds_ppb['start_time_edt'].round('H')
ground_cmpds_ppb['hour'] = ground_cmpds_ppb['hour'].dt.hour

ground_cmpds_ppb.drop(columns = [
    'end_time_edt',
    'sample_id',
    'unit'
    ], inplace = True)
ground_cmpds_ppb.replace('ND', np.nan, inplace = True)

# grouped_cmpds = ground_cmpds_ppb.groupby(['hour']).median()
# grouped_cmpds.drop(columns = ['start_time_edt'], inplace = True)

# reading in ACSM OA Data
file = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\meta_data\FROG_ACSM_NR-PM1.csv'
col_names = ['datetime_local', 'OA_ug_m3']
oa_data = pd.read_csv(file, usecols = [0, 2], skiprows = [0], 
                      names = col_names, parse_dates = ['datetime_local'])


path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\meta_data\\'
file = '20250613_back_traj_full.xlsx'

back_traj =  pd.read_excel(path + file, 
                           sheet_name = 'Full_Trajectory_withPop_240129', 
                           header = 0,
                           usecols = [
                               'date_arrival',
                               'Traj_Age_hr_c',
                               'people_per_mi2_c',
                               'Dist_Traveled_m',
                               'land area mi2',
                               'on_land',
                               'Ambient_Temp',
                               'Rel_Humidity',
                               'Height_m',
                               'Mixed_Layer_Depth_m',
                               'windspeed_m_s',
                               'Pop_ed'
                               ],
                           skiprows = lambda x: x not in [0] and x < 25
                           ) 
# Renaming columns nicely for the plot
back_traj.rename(columns = {
                       'Traj_Age_hr_c' : 'traj_age_hr',
                       'Dist_Traveled_m' : 'Distance\nTraveled (m)',
                       'land area mi2' :'Land Area ($mi^2$)',
                       'Ambient_Temp' : 'Temperature (K)',
                       'Rel_Humidity' : 'Relative\nHumidity (%)',
                       'Height_m' : 'Airmass \nHeight (m)',
                       'Mixed_Layer_Depth_m' : 'Mixed Layer\nDepth (m)',
                       'Pop_ed' : 'Population (#)',
                       'windspeed_m_s' : 'Windspeed (m/s',
                       'people_per_mi2_c' : 'Population Density\n(people $mi^{-2}$)'
    }, inplace = True)
back_traj.set_index('date_arrival', inplace = True)

# %% reading factor solution data # edit here!

# file containing factor solution folders
names = ['compounds']
base_path = r'C:\\Users\\rossellr\\OneDrive - Colostate\\Desktop\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\pmf\\output'
specific_date_file = '\\20260324\\' # edit! # needs to end in '\\'
path = base_path + specific_date_file
base_naming = '_factor' # edit !

num_cmpds = 40 # edit!

# need to know what factors are containined in the output file.
min_factor_soln = 3 # edit !
max_factor_soln = 7 # edit !
factor_soln_range = range(min_factor_soln, (max_factor_soln + 1)) 
factors = []
for i in factor_soln_range:
    factors.append('factor' + f'{i}') # list of factors

file_paths = [] # creating a path to each folder containing solution information
for i in factor_soln_range:
    file_paths.append(path + '\\' + f'{i}' + base_naming + '\\') # getting file path for each factor folder
    
profile_conc = [] # going to store each set of factor dfs in its own dict. 
percent_SppSum = []
percent_factor_total = [] 
factor_ts = []
residual_ts = []
scaled_residual_ts = []

for i, folder in enumerate(file_paths):
    factor = factors[i] # getting each factor name from the factor list
    
    for x in os.listdir(folder): 
        if x.endswith('base.xlsx'): # only pulling from the base file
            full_file = os.path.join(folder, x)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore') # ignoring all the parse formatting warnings
                
                df = pd.read_excel(full_file, sheet_name = 'Profiles', 
                                         skiprows = [0, 1, 2], nrows = num_cmpds) 
                profile_conc.append({'factors' : factor, 'df' : df}) # creating a dict entry of each factor and its df 

                df = pd.read_excel(full_file, sheet_name = 'Profiles', 
                                        skiprows = range(0, num_cmpds + 7), 
                                        nrows = num_cmpds) # change to numcmpds + 7
                percent_SppSum.append({'factors' : factor, 'df' : df})
    
                df = pd.read_excel(full_file, sheet_name = 'Profiles', 
                                        skiprows = range(0, (num_cmpds * 2) + 11), nrows = num_cmpds)
                percent_factor_total.append({'factors' : factor, 'df' : df})
                
                df = pd.read_excel(full_file, sheet_name = 'Contributions', 
                                           skiprows = [0, 1, 2], 
                                           nrows = len(ground_cmpds_ppb) + 5,
                                           parse_dates = [1], date_format = dt_format)
                factor_ts.append({'factors' : factor, 'df' : df})
                
                df = pd.read_excel(full_file, sheet_name = 'Residuals', 
                                        skiprows = [0, 1], 
                                        nrows = len(ground_cmpds_ppb),
                                        parse_dates = [1], 
                                        date_format = dt_format
                                        ) 
                residual_ts.append({'factors' : factor, 'df' : df})
                
                df = pd.read_excel(full_file, sheet_name = 'Residuals', 
                                           skiprows = range(0, len(ground_cmpds_ppb) + 5), 
                                           nrows = len(ground_cmpds_ppb),
                                           parse_dates = [1], 
                                           date_format = dt_format
                                           )
                scaled_residual_ts.append({'factors' : factor, 'df' : df})
                
def dict_to_df(dict):
    df = None
    name = None
    for x in dict.values():
        if isinstance(x, pd.DataFrame):
            df = x
        elif isinstance(x, str):
            name = x
    if df is not None and name is not None:
        df.name = name  # Assign name to the DataFrame within the list; can call using df.name
        return df
    return None

profile_conc_dfs = [] # turning them into dataframes
for dict in profile_conc:
    result = dict_to_df(dict)
    profile_conc_dfs.append(result)
    
percent_SppSum_dfs = []
for dict in percent_SppSum:
    result = dict_to_df(dict)
    percent_SppSum_dfs.append(result)
    
percent_factor_total_dfs = []
for dict in percent_factor_total:
    result = dict_to_df(dict)
    percent_factor_total_dfs.append(result)
    
factor_ts_dfs = []
for dict in factor_ts:
    result = dict_to_df(dict)
    factor_ts_dfs.append(result)
    
residual_ts_dfs = []
for dict in residual_ts:
    result = dict_to_df(dict)
    residual_ts_dfs.append(result)
    
scaled_residual_ts_dfs = []
for dict in scaled_residual_ts:
    result = dict_to_df(dict)
    scaled_residual_ts_dfs.append(result)
    

EVERYTHING = [profile_conc_dfs, percent_SppSum_dfs, percent_factor_total_dfs, factor_ts_dfs]
for each_list in EVERYTHING:
    for df in each_list:
        df.drop(columns = ['Unnamed: 0'], inplace = True)
        df.rename(columns = {'Unnamed: 1' : 'compounds'}, inplace = True)
for df in factor_ts_dfs:
    df.replace(-999, np.nan, inplace = True)
    df.rename(columns = {'compounds' : 'datetime'}, inplace = True)
    df['hour'] = df['datetime'].round('H')
    df['hour'] = df['hour'].dt.hour


residuals = [residual_ts_dfs, scaled_residual_ts_dfs]
for each_list in residuals:
    for df in each_list:
        df.drop(columns = ['Run'], inplace = True)
        df.rename(columns = {'Date_Time' : 'datetime'}, inplace = True)
        df.replace(-999, np.nan, inplace = True)
        df['hour'] = df['datetime'].round('H')
        df['hour'] = df['hour'].dt.hour

# OKAY
# so now we have 3 lists, each containin 7 dataframes. Each df contains data for a different factor solution.

# code used in blocks below
ignore = ['datetime', 'hour']


# %% checking on traffic/industrial tracers

traffic_tracers = ground_cmpds_ppb[[
    'benzene',
    'toluene', 
    'ethylbenzene',
    'm-xylene',
    'o-xylene',
    'propylbenzene',
    '3-ethyl toluene',
    'naphthalene',
    'methylnaphthalene 1',
    'methylnapthalene 2',
    'dodecane',
    'tetradecane',
    'hexadecane',
    'octadecane',
    'eicosane',
    'heptadecane'
    ]]

df = traffic_tracers.copy()
df = df.corr()  # getting pearson coeficcients
df = np.square(df) # getting R2 values

fig, ax = plt.subplots(figsize = (24, 20), dpi = 300)
cax = ax.imshow(df, cmap = 'coolwarm',  # plot matrix w/ chosen colormap
          interpolation = 'nearest') # round to nearest decimal place for heatmap value
  
cbar = plt.colorbar(cax, shrink = 0.95, label = 'R2 Coefficient Correlations')
cbar.ax.tick_params(labelsize = 18)
ax.set_xticks(range(len(df.columns))) # set xticks based on the columns in the matrix
ax.set_xticklabels(df.columns, 
         rotation = 90, fontsize = 14) # set x values based on columns and rotate for visibility
ax.set_yticks(range(len(df.columns))) # repeat for yaxis
ax.set_yticklabels(df.columns, fontsize = 14)
for (i, j), value in np.ndenumerate(df): # (coordinates) & r-value in loop in values within the mask:
  #     if i > j: # ensure annotated values are in the bottom half of the plot
  if float(f'{value:.2}') >= 0.2: # only displaying R2 higher than 0.5
      ax.text(j, i, f'{value:.2}', ha = 'center', va = 'center', color = 'black', fontsize = 8) # adjusting the value:# will change the sigfigs

plt.tight_layout()

plt.show()
plt.close()



# %% percent distribution plots

copy_list = [df.copy() for df in percent_factor_total_dfs] # no overwriting! 

factor_dfs_totals = [df.copy() for df in profile_conc_dfs] # no overwriting! 
factor_dfs_totals = [df.sum(axis = 0, numeric_only = True) for df in factor_dfs_totals] # summing each factor total

# plots for each solution
for i, df in enumerate(copy_list):


    # getting the dfs organized correctly
    units = 'ppbv'
    #df_total = profile_conc.sum(axis = 0, numeric_only = True)
    df = df.transpose()  # need cmpds as column headers and factors as index
    df.columns = df.iloc[0] # setting cmpd row as column headers
    df = df.iloc[1:]  # dropping extra cmpd name row
    
    # need list of compounds
    variable_count = len(df.index) # of factors
    
    df.rename(columns = {'4-tert-butylcyclohexyl acetate peak 1' : '4-tert-butylcyclohexyl acetate 1',
                           '4-tert-butylcyclohexyl acetate peak 2' : '4-tert-butylcyclohexyl acetate 2',
                           '2-tert-butylcyclohexyl acetate peak 1' : '2-tert-butylcyclohexyl acetate 1'}, 
                     inplace = True)
    
    compounds = df.columns
    
    fontsize = 18
    fontweight = 'normal'
    
    fig, ax = plt.subplots(figsize = (16, 20), dpi = 300)
    bars = df.plot(ax = ax, 
                   stacked = True,
                     kind = 'bar', 
                     subplots = False, 
                     sharex = True, 
                     use_index = True, 
              ylim = [0, 100],
              fontsize = fontsize,
              color = [cmpd_colors_dict[col] for col in df.columns],
              edgecolor = 'white', 
              linewidth = 0.75,
              rot = 360,
              width = 0.97
              )
    ax.set_ylabel(
        'Compound Distribution (%)',
        fontsize = fontsize,
        fontweight = fontweight
        )
    
    ax.set_yticklabels(ax.get_yticks(), weight = fontweight) # setting y-ticks to bold
    for label in ax.get_xticklabels(): # setting axis labels to bold
        label.set_fontweight(fontweight)
    
    
    
    # labeling the top 5 compounds making up each factor:
    for bar_idx, (index, row) in enumerate(df.iterrows()):
        top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(5).index.tolist()
    
        cum_height = 0
        for col in df.columns:
            height = row[col]
            mid_point = cum_height + height / 2
            if col in top_cols and height > 0:
                    # have to wrap the really long compound labels so they fit in the bar
                    wrapped_label = '\n'.join(textwrap.wrap(col, width = 25))  # Adjust width as needed
    
                    ax.text(
                        x = bar_idx,
                        y = mid_point,
                        s = f'{wrapped_label}' + f'\n{height:.1f}%', 
                        ha = 'center',
                        va = 'center',
                        fontsize = 10 if height < 3 else 15,
                        color = 'black'  # or a color that contrasts the bar
                        )
            cum_height += height
    
    plt.legend(ncols = 1, fontsize = 16, loc = 'best',
               bbox_to_anchor = (1, 1), reverse = True)
    
    fontsize = 20
    totals_this_solution = factor_dfs_totals[i].reindex(df.index).values
    add_percent_bar_totals(
        ax,
        x=range(len(df)),
        y=99,
        total=totals_this_solution,
        units=units,
        fontsize=fontsize,
        fontweight=fontweight
        )
    
    plt.show()
    plt.close()

# %% factors by total compound mass

def add_bar_totals(x, y): # function to add totals to the top of each bar in the bar plot
    for i in range(len(x)):
        plt.text(x = i, y = y.iloc[i] + 0.005 ,  # setting x and y positions of labels - centering with bar and then slightly above top of bar
                 s = f'{y.iloc[i]:0.3f}', # setting and formatting what appears in the label
                 ha = 'center',
                 rotation = 15)

factor_dfs_totals = [df.copy() for df in profile_conc_dfs] # no overwriting! 
factor_dfs_totals = [df.sum(axis = 0, numeric_only = True) for df in factor_dfs_totals] # summing each factor total

# plots for each solution
for i, df in enumerate(factor_dfs_totals):
    fig, ax = plt.subplots(figsize = (5, 8), dpi = 300)
    bar_plot = df.plot(ax = ax, kind = 'bar', sharex = True,
                       title = f'{len(df)}' + ' Factor Solution\nTargeted + Nontargeted Compounds',
                       ylabel = 'Total Factor Mixing Ratio (ppb)',
                       rot = 45,
                       edgecolor = 'gray', 
                       color = 'cadetblue',
                       linewidth = 0.75,
                       )
    add_bar_totals(range(len(df)), df.iloc[range(len(df))])
    plt.show() 
    
# %% correlation matrices

# getting the big ass meta data df ready

meta_data_copy = meta_data.copy() # no overwriting !
cmpds_ppb_copy = ground_cmpds_ppb.copy()

starttime = meta_data_copy['start_time']

meta_data_copy.drop(columns = ['start_time', 'avg_rain_in/s'], inplace = True)
cmpds_ppb_copy.drop(columns = ['start_time_edt'], inplace =  True)
meta_data_copy.set_index('hour', inplace = True)
cmpds_ppb_copy.set_index('hour', inplace = True)

meta_data_concat = pd.concat([meta_data_copy, cmpds_ppb_copy], axis = 1)

factor_ts_corr_dfs = [df.copy() for df in factor_ts_dfs] # no overwriting! 
for df in factor_ts_corr_dfs:
    df.drop(columns = ['datetime'], inplace = True)
    df.set_index('hour', inplace = True)

# adding metadata to each factor df to run the corr matrix
factor_ts_corr_dfs = [pd.concat([df, meta_data_concat], axis = 1) for df in factor_ts_corr_dfs] 

factor_ts_corr_dfs = [df.corr() for df in factor_ts_corr_dfs] # getting pearson coeficcients
factor_ts_corr_dfs = [np.square(df) for df in factor_ts_corr_dfs] # getting R2 values


for i, df in enumerate(factor_ts_corr_dfs):
    fig, ax = plt.subplots(figsize = (24, 20), dpi = 300)
    cax = ax.imshow(df, cmap = 'coolwarm',  # plot matrix w/ chosen colormap
                        interpolation = 'nearest') # round to nearest decimal place for heatmap value
    
    cbar = plt.colorbar(cax, shrink = 0.95, label = 'R2 Coefficient Correlations')
    cbar.ax.tick_params(labelsize = 18)
    ax.set_xticks(range(len(df.columns))) # set xticks based on the columns in the matrix
    ax.set_xticklabels(df.columns, 
                       rotation = 90, fontsize = 14) # set x values based on columns and rotate for visibility
    ax.set_yticks(range(len(df.columns))) # repeat for yaxis
    ax.set_yticklabels(df.columns, fontsize = 14)
    for (i, j), value in np.ndenumerate(df): # (coordinates) & r-value in loop in values within the mask:
    #     if i > j: # ensure annotated values are in the bottom half of the plot
        if float(f'{value:.2}') >= 0.2: # only displaying R2 higher than 0.5
           ax.text(j, i, f'{value:.2}', ha = 'center', va = 'center', color = 'black', fontsize = 8) # adjusting the value:# will change the sigfigs
    plt.tight_layout()
  #  plt.title('Ground Targeted Compounds')
    plt.show()
    plt.close()

# %% factor ratios t:b

def add_bar_totals(x, y): # function to add totals to the top of each bar in the bar plot
    for i in range(len(x)):
        plt.text(x = i, y = y.iloc[i] + (y.iloc[i] * 0.02) ,  # setting x and y positions of labels - centering with bar and then slightly above top of bar
                 s = f'{y.iloc[i]:0.3f}', # setting and formatting what appears in the label
                 ha = 'center'
                 ) 


for df in profile_conc_dfs:
    df.set_index('compounds', inplace = True)
    df.loc['t_b'] = df.loc['toluene'] / df.loc['benzene'] 

# plots for each solution
for i, df in enumerate(profile_conc_dfs):
    row = df.loc['t_b']
    fig, ax = plt.subplots(figsize = (5, 8), dpi = 300)
    bar_plot = row.plot(ax = ax, kind = 'bar', 
                       title = f'{len(df.columns)}' + ' Factor Solution',
                       ylabel = 'Toluene : Benzene',
                       rot = 45,
                       color = 'cadetblue',
                       edgecolor = 'gray', 
                       linewidth = 0.75,
                       )
    add_bar_totals(range(len(row)), row.iloc[range(len(row))])
    plt.show() 


for df in profile_conc_dfs:
    df.drop(labels = ['t_b'], inplace = True)
    df.reset_index(inplace = True)
    
# %% averaged back trajectories and factors

copy_list = [df.copy() for df in factor_ts_dfs]

for df, name in zip(copy_list, factors):
    df['datetime_hour'] = df['datetime'].dt.round('h')
    df.set_index('datetime_hour', inplace = True)

    # filtering backtrajectories for voc sampling times
    backtraj_filtered = back_traj[back_traj.index.isin(df.index)]
    # dropping all airmasses >12 hours old
    backtraj_filtered = backtraj_filtered[backtraj_filtered['traj_age_hr'] >= -12]
    # averaging for same arrival times
    backtraj_filtered.reset_index(inplace = True, drop = False)
    backtraj_filtered['date_arrival'] = pd.to_datetime(backtraj_filtered['date_arrival'])
    backtraj_filtered = backtraj_filtered.groupby('date_arrival').mean().reset_index()

    # now dropping sample times w/ no back-traj.
    df = df[df.index.isin(backtraj_filtered['date_arrival'])]
    df.reset_index(inplace = True, drop = False)

    concat = pd.concat([df, backtraj_filtered], axis = 1)
    concat.rename(columns = {
        'datetime_hour' : 'factor_dt_rounded',
        'datetime' : 'factor_dt',
        'date_arrival' : 'airmass_dt_arrival'
        }, inplace = True)
    concat.set_index('airmass_dt_arrival', inplace = True)
    concat.drop(columns = ['factor_dt', 
                           'factor_dt_rounded',
                           'traj_age_hr',
                           'hour',
                           'on_land',
                           'Mixed Layer\nDepth (m)',
                           'Population (#)'
                           ], inplace = True)
    
    df = concat.corr() # getting pearson coeficcients
    # half matrix plot
    lower_mask = np.tri(df.shape[0], df.shape[1], k = -1)
    upper_mask = lower_mask.T
    matrix_mask = np.ma.array(df, mask = upper_mask) # only plotting bottom half of matrix plot

    plt.rcParams["mathtext.default"] = 'regular'
    fig, ax = plt.subplots(figsize = (24, 20), 
                           dpi = 300)
    cax = ax.imshow(matrix_mask, cmap = 'coolwarm',  # plot matrix w/ chosen colormap
                      interpolation = 'nearest') # round to nearest decimal place for heatmap value
      
    cbar = plt.colorbar(cax, shrink = 0.95)
    cbar.set_label(label = 'Pearson Coefficient Correlations',
                   weight = 'bold', 
                   fontsize = 24)

    cbar.ax.tick_params(labelsize = 24)
    ax.set_xticks(range(len(df.columns))) # set xticks based on the columns in the matrix
    ax.set_xticklabels(df.columns, 
                     rotation = 90, 
                     fontsize = 24) # set x values based on columns and rotate for visibility
    ax.set_yticks(range(len(df.columns))) # repeat for yaxis
    ax.set_yticklabels(df.columns, fontsize = 24)

    for (i, j), value in np.ndenumerate(df): # (coordinates) & r-value in loop in values within the mask:
      if i > j: # ensure annotated values are in the bottom half of the plot
          if float(f'{value:.2}') >= 0.3 : # only displaying r higher than 0.3
             ax.text(j, i, f'{value:.2f}', ha = 'center', va = 'center', 
                     color = 'black', fontsize = 24) 
          if float(f'{value:.2}') <= - 0.3 : # adjusting the value:# will change the sigfigs
             ax.text(j, i, f'{value:.2f}', ha = 'center', va = 'center', 
                     color = 'black', fontsize = 24)
          if float(f'{value:.2}') == 1.0 : 
              ax.text(j, i, f'{value:.1f}', ha = 'center', va = 'center', 
                      color = 'black', fontsize = 24)
        

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    ax.set_title(
         f'{name}',
         fontsize = 20,
         fontweight = 'normal'
         )
    
    plt.tight_layout()

    plt.show()
    plt.close()

# %% averaged backtraj analyses (EBF): 6 hours 

# 6  hour
path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\meta_data\\'
file = 'Back_Traj_Summaries_6hr_20260117.csv'

backtraj =  pd.read_csv(path + file, 
                           header = 0,
                           parse_dates = ['date_arrival']
                           )
backtraj['date_arrival'] = pd.to_datetime(backtraj['date_arrival'])
hour = 6
backtraj.set_index('date_arrival', inplace = True)
backtraj.drop(columns = ['date_arrival.r',], inplace = True)

copy_list = [df.copy() for df in factor_ts_dfs]

for df, name in zip(copy_list, factors):
    df['datetime_hour'] = df['datetime'].dt.round('h')
    df.set_index('datetime_hour', inplace = True)
    df.drop(columns = ['datetime', 'hour'], inplace = True)
    

    # check if df indices match:
    [x for x in list(df.index.unique()) if x not in list(backtraj.index.unique())]
    
    # filtering backtrajectories for voc sampling times
    backtraj = backtraj[backtraj.index.isin(df.index)]
    
    # concat factor TS and backtrajs
    concat = pd.concat([df, backtraj], axis = 1)
    
    
    # making the correlation plot
    df = concat.corr() # getting pearson coeficcients
    # half matrix plot
    lower_mask = np.tri(df.shape[0], df.shape[1], k = -1)
    upper_mask = lower_mask.T
    matrix_mask = np.ma.array(df, mask = upper_mask) # only plotting bottom half of matrix plot
    
    
    fig, ax = plt.subplots(figsize = (24, 20), dpi = 300)
    cax = ax.imshow(matrix_mask, cmap = 'coolwarm',  # plot matrix w/ chosen colormap
                      interpolation = 'nearest') # round to nearest decimal place for heatmap value
      
    cbar = plt.colorbar(cax, shrink = 0.95)
    cbar.set_label(label = 'Pearson Coefficient Correlations', weight = 'bold', fontsize = 24)
    
    cbar.ax.tick_params(labelsize = 24)
    ax.set_xticks(range(len(df.columns))) # set xticks based on the columns in the matrix
    ax.set_xticklabels(df.columns, 
                     rotation = 90, fontsize = 24) # set x values based on columns and rotate for visibility
    ax.set_yticks(range(len(df.columns))) # repeat for yaxis
    ax.set_yticklabels(df.columns, fontsize = 24)
    for (i, j), value in np.ndenumerate(df): # (coordinates) & r-value in loop in values within the mask:
      if i > j: # ensure annotated values are in the bottom half of the plot
          if float(f'{value:.2}') >= 0.3 : # only displaying R higher than 0.5
             ax.text(j, i, f'{value:.1}', ha = 'center', va = 'center', 
                     color = 'black', fontsize = 24) # adjusting the value:# will change the sigfigs
          if float(f'{value:.2}') <= - 0.3 : 
             ax.text(j, i, f'{value:.1}', ha = 'center', va = 'center', 
                     color = 'black', fontsize = 24)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    
    ax.set_title(
         f'{name} - averaged 6 hours',
         fontsize = 20,
         fontweight = 'normal'
         )
    
    plt.show()
    plt.close()

# %% averaged backtraj analyses (EBF): 12 hours 

# 12 hour
path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\meta_data\\'
file = 'Back_Traj_Summaries_12hr_20260117.csv'

backtraj =  pd.read_csv(path + file, 
                           header = 0,
                           parse_dates = ['date_arrival']
                           )
backtraj['date_arrival'] = pd.to_datetime(backtraj['date_arrival'])
hour = 12
backtraj.set_index('date_arrival', inplace = True)
backtraj.drop(columns = ['date_arrival.r',], inplace = True)

copy_list = [df.copy() for df in factor_ts_dfs]

for df, name in zip(copy_list, factors):
    df['datetime_hour'] = df['datetime'].dt.round('h')
    df.set_index('datetime_hour', inplace = True)
    df.drop(columns = ['datetime', 'hour'], inplace = True)

    # check if df indices match:
    [x for x in list(df.index.unique()) if x not in list(backtraj.index.unique())]
    
    # filtering backtrajectories for voc sampling times
    backtraj = backtraj[backtraj.index.isin(df.index)]
    
    
    # concat factor TS and integ. BT
    concat = pd.concat([df, backtraj], axis = 1)
    
    
    # making the correlation plot
    df = concat.corr() # getting pearson coeficcients
    # half matrix plot
    lower_mask = np.tri(df.shape[0], df.shape[1], k = -1)
    upper_mask = lower_mask.T
    matrix_mask = np.ma.array(df, mask = upper_mask) # only plotting bottom half of matrix plot
    
    
    fig, ax = plt.subplots(figsize = (24, 20), dpi = 300)
    cax = ax.imshow(matrix_mask, cmap = 'coolwarm',  # plot matrix w/ chosen colormap
                      interpolation = 'nearest') # round to nearest decimal place for heatmap value
      
    cbar = plt.colorbar(cax, shrink = 0.95)
    cbar.set_label(label = 'Pearson Coefficient Correlations', weight = 'bold', fontsize = 24)
    
    cbar.ax.tick_params(labelsize = 24)
    ax.set_xticks(range(len(df.columns))) # set xticks based on the columns in the matrix
    ax.set_xticklabels(df.columns, 
                     rotation = 90, fontsize = 24) # set x values based on columns and rotate for visibility
    ax.set_yticks(range(len(df.columns))) # repeat for yaxis
    ax.set_yticklabels(df.columns, fontsize = 24)
    for (i, j), value in np.ndenumerate(df): # (coordinates) & r-value in loop in values within the mask:
      if i > j: # ensure annotated values are in the bottom half of the plot
          if float(f'{value:.2}') >= 0.3 : # only displaying R higher than 0.5
             ax.text(j, i, f'{value:.2}', ha = 'center', va = 'center', color = 'black', fontsize = 20) # adjusting the value:# will change the sigfigs
          if float(f'{value:.2}') <= - 0.3 : 
             ax.text(j, i, f'{value:.1}', ha = 'center', va = 'center', color = 'black', fontsize = 20)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
   
    ax.set_title(
         f'{name} - averaged 12 hours',
         fontsize = 20,
         fontweight = 'normal'
         )
    
    plt.show()
    plt.close()
    
    
# %% pollution roses + factors

copy_list = [df.copy() for df in factor_ts_dfs]
for df, name in zip(copy_list, factors):
    dt_cols_ignore = ['datetime', 'hour']
    
    cols_to_plot = [c for c in df.columns if c not in dt_cols_ignore]
    if len(cols_to_plot) == 0:
        continue
    
    # Parameters
    num_bins = 36  # Number of directional bins (e.g., 15 degrees per bin)
    bin_edges = np.linspace(0, 360, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    wind_spd = meta_data['avg_WindSpeed_m/s']
    wind_dir = meta_data['median_WindDir']

    n_plots = len(cols_to_plot)
    n_cols = math.ceil(np.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    # Plot
    fig, axs = plt.subplots(n_rows, 
                            n_cols, 
                           subplot_kw = {'projection': 'polar'}, 
                           figsize = (4 * n_cols, 4 * n_rows),
                           dpi = 300
                           )
    axs = np.atleast_1d(axs).flatten()
    plt.subplots_adjust(hspace = 0.35, wspace = 0) # making the subplots closer together

    for idx, col in enumerate(cols_to_plot):
        ax = axs[idx] 
        voc_conc = df[col]
        # Bin the data by wind direction
        bin_indices = np.digitize(wind_dir, bin_edges) - 1

        # Initialize arrays to hold mean wind speed and VOC concentration per bin
        mean_spd = np.zeros(num_bins)
        mean_voc = np.zeros(num_bins)
        
        for i in range(num_bins):
            mask = bin_indices == i
            if np.any(mask):
                mean_spd[i] = np.mean(wind_spd[mask])
                mean_voc[i] = np.mean(voc_conc[mask])
            else:
                mean_spd[i] = 0
                mean_voc[i] = np.nan  # or 0, depending on preference
        
        # Convert bin centers to radians for plotting
        theta = np.deg2rad(bin_centers)
           
        bars = ax.bar(theta, 
                      mean_spd, 
                      width = np.deg2rad(360/num_bins),
                      bottom = 0.0,
                      color = plt.cm.Reds(mean_voc / np.nanmax(mean_voc)), 
                      edgecolor = 'black',
                      alpha = 1,
                      linewidth = 0.02)
        
        # Add colorbar for VOC concentration
        sm = plt.cm.ScalarMappable(cmap ='Reds', 
                                   norm = plt.Normalize(vmin = np.nanmin(mean_voc), 
                                                        vmax = np.nanmax(mean_voc)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax = ax, pad = 0.15, 
                            orientation = 'horizontal',
                            shrink = 0.8,
                            format = lambda x, _: f"{x:.1f}")
        cbar.set_label(f'{col} Mixing Ratio ($ppb_v$)',
                       fontsize = 12)
        
        # Set the zero location and direction
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        # Set radial grid and labels
        ax.set_rlabel_position(135)  # Move radial labels away from crowded angles
        ax.set_title(f'{col}', 
                     fontsize = 14, 
                     pad = 30)
        ax.grid(
            visible = True,
            which = 'major',
            axis = 'both',
            color = 'gray',
            alpha = 0.3
            )

            
    plt.show()
    
# %% factor diel plots

copy_list = [df.copy() for df in factor_ts_dfs]

for df, name in zip(copy_list, factors):
    fig, ax = plt.subplots()
    
    for col in df.columns:
        if col not in ignore:
            grouped = df.groupby('hour')[col].mean()
            ax.plot(grouped.index, grouped.values, label = col,
                    color = color_dict[col],
                     linestyle = '--', marker = 'o')
       
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Median Normalized Cmpd Conc.')
    
    ax.legend()
    
    ax.set_title(
         f'{name}',
         fontsize = 16,
         fontweight = 'normal'
         )
    
    plt.show()
    plt.close()  
    
# %% factor time series against temp and smoke events  

copy_list = [df.copy() for df in factor_ts_dfs]

for df, name in zip(copy_list, factors):
    
    df.set_index('datetime', inplace = True)
    df.drop(columns = ['hour'], inplace = True)
    
    fig, ax = plt.subplots(
        figsize = (8, 6), 
        dpi = 300)
    
    # plotting against temperature
    ax2 = ax.twinx()
    ax2.plot(weather['dt'], weather['temp_c'],
      color = 'maroon',
      linestyle = '--', 
      linewidth = 1,
      alpha = 0.5)
    ax2.set(
            ylim  = [0, 40],
          )
    ax2.set_ylabel('Median Temperature (°C)', color = 'maroon')
    # changing axis color for plot readibility
    ax2.spines['right'].set_color('maroon')
    ax2.tick_params(axis = 'y', colors = 'maroon')
    [t.set_color('maroon') for t in ax2.yaxis.get_ticklines()]
    [t.set_color('maroon') for t in ax2.yaxis.get_ticklabels()]
    
    # plotting against OA conc:
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 50))  # scotting axis to the right for visability
    ax3.plot(oa_data['datetime_local'], oa_data['OA_ug_m3'],
             color = 'black',
             linestyle = '--', 
             linewidth = 1,
             alpha = 0.5)
    ax3.set(
           ylim  = [0, 35]
                     )
    ax3.set_ylabel('Organic Aerosol Mass (μg $m^-3$)', color = 'darkslategray')
    
    # changing axis color for plot readibility
    [t.set_color('darkslategray') for t in ax3.yaxis.get_ticklines()]
    [t.set_color('darkslategray') for t in ax3.yaxis.get_ticklabels()]
    ax3.tick_params(axis = 'y', colors = 'darkslategray')
    ax3.spines['right'].set_color('darkslategray')
     
    # plotting factor solns
    for col in df.columns:
         ax.plot(df.index, df[col],
                 label = col,
                 linewidth = 2,
                 marker = 'o',
                 markeredgecolor = 'black',
                 markerfacecolor = None,
                 markersize = 4,
                 color = color_dict[col]
                 ) 
    ax.set(
           ylabel = 'Normalized Cmpd Conc.',
           xlim = (df.index[0] - timedelta(hours = 6), df.index[-1] + timedelta(hours = 6)), # adjusting x-lim
           xlabel = 'Local Date/Time',
           title = f'{len(df.columns)} Factor Solution'
           )
    legend = ax.legend(
        loc = 'best',
        shadow = True,
        framealpha = 1.0,
        facecolor = 'white',
        labelcolor = 'black',
        edgecolor = 'black',
        ncols = len(df.columns)
        )
    legend.get_frame().set_alpha(1.0)
    
     
    # adding heatwave and smoke event boxes box
    heatwave_start = datetime.datetime(2023, 7, 27) # dt object type # edit !
    heatwave_end = datetime.datetime(2023, 7, 29) # edit ! 
    smoke1_start = datetime.datetime(2023, 7, 19) # edit !
    smoke1_end = datetime.datetime(2023, 7, 21) # edit !
    smoke2_start = datetime.datetime(2023, 8, 5) # edit !
    smoke2_end = datetime.datetime(2023, 8, 7) # edit !
     

    height = 20 
    box_color = 'maroon' # edit ! 
    set_alpha = 0.2 # edit ! 
    rect_box(heatwave_start, heatwave_end, ax, '') # adding heatwave box
    
    box_color = 'gray'
    rect_box(smoke1_start, smoke1_end, ax, '')
    rect_box(smoke2_start, smoke2_end, ax, '') # adding smoke boxes
    
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(md.DateFormatter('%D %H:%M'))
    
        
    ax.set_title(
         f'{name}',
         fontsize = 16,
         fontweight = 'normal'
         )
    
    plt.show()
    plt.close()
    
# %% factor weekday vs weekend trends


weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] # set days of week in a weekday
weekend = ['Saturday', 'Sunday']  # set days in a weekend
grouping = ['weekend', 'weekday'] # we want to sort data based on weekday or weekend status

ignore = ['datetime', 'hour', 'dow', 'weekday_weekend'] 

copy_list = [df.copy() for df in factor_ts_dfs]

for df, name in zip(copy_list, factors):

    df['dow'] = df['datetime'].dt.day_name()
    df['weekday_weekend'] = df['dow'].apply(lambda x: 'weekend' if x in weekend else 'weekday') 
    
    df_weekday = df[df['weekday_weekend'] == 'weekday'] # boolean mask to select weekdays
    df_weekend = df[df['weekday_weekend'] == 'weekend'] # boolean mask to select weekends
    
    # making the plot
    fig, axes = plt.subplots(
         nrows = len(df_weekday.columns) - len(ignore),
         ncols = 1,
         sharex = True,
         dpi = 200,
         figsize = (6, 8)
         )
    i = 0 # starting the counter at 0
          
    for col in df_weekday.columns:
        if col not in ignore:
            ax = axes[i]
            
            grouped_weekday = df_weekday.groupby('hour')[col].describe()
            grouped_weekend = df_weekend.groupby('hour')[col].describe()
            # plotting weekday day
            ax.plot(grouped_weekday.index, grouped_weekday['50%'], 
                    #label = f'{col} - Weekday',
                     linestyle = '--', marker = 'o',
                     color = 'chocolate',
                     label = 'Weekday')
            ax.fill_between(grouped_weekday.index, grouped_weekday['25%'], grouped_weekday['75%'],
                            alpha = 0.3, color = 'chocolate')
            # plotting weekend data
            ax.plot(grouped_weekend.index, grouped_weekend['50%'], 
                    #label = f'{col} - Weekend',
                     linestyle = '--', marker = 'o',
                     label = 'Weekend',
                     color = 'darkolivegreen')
            ax.fill_between(grouped_weekend.index, grouped_weekend['25%'], grouped_weekend['75%'],
                            alpha = 0.3, color = 'darkolivegreen')
            ax.set_title(
                label = f'{col}',
                loc = 'right',
                x = 0.97,
                y = 0.8,
                transform=ax.transAxes
                )
            i += 1  # move to next axis
            
    # setting overall plot title and legend
    
    # create manual symbols for legend
    weekday_point = Line2D([0], [0], 
                           label = 'Weekday', 
                           marker = 'o', 
                           linestyle = '--',
                           markersize = 10, 
                           markeredgecolor = 'chocolate', 
                           markerfacecolor = 'chocolate',
                           color = 'chocolate'
                           )
    weekend_point = Line2D([0], [0], 
                           label = 'Weekend', 
                           marker = 'o', 
                           linestyle = '--',
                           markersize = 10, 
                           markeredgecolor = 'darkolivegreen', 
                           markerfacecolor = 'darkolivegreen',
                           color = 'darkolivegreen'
                           )
    fig.legend(
        handles = [weekday_point, weekend_point],
        labels = ['Weekday', 'Weekend'],
        ncol = 2,
        loc = 'upper center',
        bbox_to_anchor = [0.5, 0.92],
        framealpha = 0.2
        )

    ax.text( # making a y-axis label for all subplots
        x = 3.9,
        y = 10, 
        s = 'Median Cmpd Conc. (Normalized ppb)',
        rotation = 'vertical'
        )
    plt.xlabel('Hour of Day')# setting x-axis label on bottom plot only
    
    ax.set_title(
         f'{name}',
         x = 0.5,
         y = -0.6,
         fontsize = 16,
         fontweight = 'normal',
         transform=ax.transAxes
         )
    
    plt.show()
    plt.close()
    
# %% looking at the summed residuals across the campaign

top_number = 10 # edit!

copy_list = [df.copy() for df in scaled_residual_ts_dfs]

for df, i in zip(copy_list, factors):
    df.drop(columns = [
        'hour',
        'Q(true)/Qexp'
        ], inplace = True)
    df.set_index('datetime', inplace = True)
    # need cmpds as index
    df = df.transpose() 
    df.index.rename('compounds', inplace = True)


    # adding totals  across the samples
    conc_totals = df.sum(axis = 1, numeric_only = True)
    total_conc_ppb = conc_totals.sum()
    conc_totals = conc_totals.to_frame()
    conc_totals.rename(columns = {0 : 'summed_conc_ppb'}, inplace = True)
    conc_totals['%_conc'] = conc_totals['summed_conc_ppb'] / total_conc_ppb * 100
    
    # getting top cmpds + summing the rest into 'other'
    conc_totals.sort_values(by = '%_conc', ascending = False, inplace = True)
    others = pd.DataFrame(data = {
        'other' : conc_totals[top_number:].sum()
        })
    others = others.transpose()
    df = conc_totals[:top_number].copy()
    df = pd.concat([df, others], axis = 0)
    
    # making a pie chart
    fig, ax = plt.subplots(figsize = (16, 16), dpi = 300)
    
    values = df['summed_conc_ppb'].value_counts(dropna = True)
    
    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = pct * total / 100.0
            return '{:.1f}%\n({v:0.1f} $ppb_v$)'.format(pct, v = val)
        return my_format
    
    df.plot(ax = ax,
            kind = 'pie',
            y = '%_conc',
            use_index = 'True',
            colors = [cmpd_colors_dict[value] for value in df.index],
            wedgeprops = {'edgecolor' : 'white', 'linewidth' : 3},
            autopct = autopct_format(df['summed_conc_ppb']),
            #autopct = lambda x: '{:.3f}'.format(x * values.sum() / 100) ,
            #startangle=90),
            # autopct = lambda pct: '{:1.0f}%'.format(pct), #if pct > 4 else '',
            hatch = ['', '', '', '', '', ''],
            textprops = {'fontsize' : 16, 'fontweight' : 'normal'},
            pctdistance = 0.83
                )
    ax.get_legend().remove()
    ax.axis('off')
    

    ax.set_title(
        f'{i}',
        x = 0.48,
        y = 0.99,
        fontsize = 20,
        fontweight = 'normal'
        )
    
    plt.show()
        

# %% scaled residual box plots

# taken from Morh et al (2012): https://doi.org/10.5194/acp-12-1649-2012

copy_list = [df.copy() for df in scaled_residual_ts_dfs]

hydrocarbons = ['dodecane', 'hexadecane', 'tetradecane']
industry = ['toluene', 'benzene','o-xylene', 'm-xylene', 'ethylbenzene', 
            '3-ethyl toluene', 'naphthalene',
           'propylbenzene', 'methylnaphthalene 1', 'methylnapthalene 2']
biogenic = ['a-pinene', 'limonene', 'camphor']
vcp = ['decamethylcyclopentasiloxane', 'dodecamethylcyclohexasiloxane', 
       'hexamethylcyclotrisiloxane', 'benzyl alcohol', 'benzyl acetate', 
       'allyl heptanoate', '4-tert-butylcyclohexyl acetate 1', '2-methyl-undecanal',
       '4-tert-butylcyclohexyl acetate 2', 'diphenyl ether', 
       'sabina ketone', 'phenoxyisopropanol','bornyl acetate', 
       '2-tert-butylcyclohexyl acetate 1', '6-tert-butyl-o-cresol', 'tricyclodecenyl acetate',
       ]
other = ['benzaldehyde', 'methyl salicylate', 'benzothiazole', 'phenol', 'acetophenone',
         'txib', 'ethylhexyl benzoate',
         '2-ethyl-1-hexanol']

for i, df in enumerate(copy_list):
    df.drop(columns = ['datetime', 'hour', 'Q(true)/Qexp'], 
                       inplace = True) # don't want datetime values

for df, name in zip(copy_list, factors):
    
    # calculate spread using IQR
    iqr = df.quantile(0.75) - df.quantile(0.25)
    
    # reorder columns from largest spread to smallest
    df = df.loc[:, iqr.sort_values(ascending = False).index]   
    
    fig, ax = plt.subplots(figsize = (8, 4), 
                           dpi = 300)
    
    plot = ax.boxplot(df,
                orientation = 'vertical',
                widths = 0.4,
                sym = '.',
                medianprops = {'color' : 'black'},
                patch_artist = True,
                tick_labels = df.columns,
                showfliers = True
                )
    ax.set_xticks(
        ticks = range(1, len(df.columns) + 1),
        labels = df.columns,
        fontsize = 8,
        rotation = 90
        )
    ax.set_title(
        f'{name}',
        x = 0.48,
        y = 0.99,
        fontsize = 16,
        fontweight = 'normal'
        )
    ax.set_ylabel(
        'Scaled Residuals',
        fontsize = 12
        )
   
    for box, label in zip(plot['boxes'], df.columns):
        box.set_alpha(0.5)
        if label in industry:
            box.set_facecolor('gray')
        elif label in hydrocarbons:
            box.set_facecolor('brown')
        elif label in biogenic:
            box.set_facecolor('lightgreen')
        elif label in vcp:
            box.set_facecolor('pink')
        elif label in other:
            box.set_facecolor('yellow')
        else:
            box.set_facecolor('white')
    
    plt.show()
    
# %% scaled residual violin plots

copy_list = [df.copy() for df in scaled_residual_ts_dfs]

hydrocarbons = ['dodecane', 'hexadecane', 'tetradecane']
industry = ['toluene', 'benzene','o-xylene', 'm-xylene', 'ethylbenzene', 
            '3-ethyl toluene', 'naphthalene',
           'propylbenzene', 'methylnaphthalene 1', 'methylnapthalene 2']
biogenic = ['a-pinene', 'limonene', 'camphor']
vcp = ['decamethylcyclopentasiloxane', 'dodecamethylcyclohexasiloxane', 
       'hexamethylcyclotrisiloxane', 'benzyl alcohol', 'benzyl acetate', 
       'allyl heptanoate', '4-tert-butylcyclohexyl acetate 1', '2-methyl-undecanal',
       '4-tert-butylcyclohexyl acetate 2', 'diphenyl ether', 
       'sabina ketone', 'phenoxyisopropanol','bornyl acetate', 
       '2-tert-butylcyclohexyl acetate 1', '6-tert-butyl-o-cresol', 'tricyclodecenyl acetate',
       ]
other = ['benzaldehyde', 'methyl salicylate', 'benzothiazole', 'phenol', 'acetophenone',
         'txib', 'ethylhexyl benzoate',
         '2-ethyl-1-hexanol']


for i, df in enumerate(copy_list):
    df.drop(columns = ['datetime', 'hour', 'Q(true)/Qexp'], 
                       inplace = True) # don't want datetime values

    
for df, name in zip(copy_list, factors):
    
    # calculate spread using IQR
    iqr = df.quantile(0.75) - df.quantile(0.25)
    
    # reorder columns from largest spread to smallest
    df = df.loc[:, iqr.sort_values(ascending = False).index]   
    
    fig, ax = plt.subplots(figsize = (10, 4), 
                           dpi = 300)
    
    plot = ax.violinplot(df,
                orientation = 'vertical',
                widths = 0.75,
                showmedians = True,
                )
    ax.set_xlim(0.3, len(df.columns) + 0.5)
    ax.set_xticks(
        ticks = range(1, len(df.columns) + 1),
        labels = df.columns,
        fontsize = 8,
        rotation = 90
        )
    ax.set_title(
        f'{name}',
        x = 0.48,
        y = 0.99,
        fontsize = 16,
        fontweight = 'normal'
        )
    ax.set_ylabel(
        'Scaled Residuals',
        fontsize = 12
        )
   
     # coloring the shaded part of the violins
    for box, label in zip(plot['bodies'], df.columns):
        box.set_alpha(0.5)
        box.set_edgecolor('black')
        if label in industry:
            box.set_facecolor('gray')
        elif label in hydrocarbons:
            box.set_facecolor('brown')
        elif label in biogenic:
            box.set_facecolor('lightgreen')
        elif label in vcp:
            box.set_facecolor('pink')
        elif label in other:
            box.set_facecolor('yellow')
        else:
            box.set_facecolor('white')
    # coloring the data spines
    for partname in ('cbars','cmins','cmaxes','cmedians'):
        vp = plot[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(0.5) 
    
    plt.show()

# %% scaled residual time series against Qexp

# taken from Morh et al (2012): https://doi.org/10.5194/acp-12-1649-2012

copy_list = [df.copy() for df in scaled_residual_ts_dfs]

df = copy_list[0]
df.set_index('datetime', inplace = True)
sep = df[['Q(true)/Qexp']]
df.drop(columns = ['hour', 'Q(true)/Qexp'], inplace = True)


df = pd.DataFrame(data = df.sum(axis = 1))
df.rename(columns = {0 : 'residual_sum'}, inplace = True)
df.insert(loc = 1, column = 'Q/Qexp', value = sep)

fig, (ax1, ax2) = plt.subplots(
    figsize = (10, 5), 
    ncols = 1, nrows = 2,
    dpi = 300)

ax1.plot(df.index, df['residual_sum'],
                 color = 'darkgray', marker = 'o',
                 label = 'Sum\nScaled Residuals')
ax1.set_ylim(0, df['residual_sum'].max() + 5)
ax1.set_yticks(ticks = np.linspace(0, df['residual_sum'].max() + 5, 5),
               labels = np.linspace(0, df['residual_sum'].max() + 5, 5)
               )
ax1.set_ylabel('Sum Scaled Residuals',
               fontsize = 14)

ax2.plot(df.index, df['Q/Qexp'],
                 color = 'darkgray', marker = 'o',
                 label = 'Q/Qexp')
ax2.set_ylim(0, df['Q/Qexp'].max() + 5)
ax2.set_yticks(ticks = np.linspace(0, df['Q/Qexp'].max() + 5, 5),
               labels = np.linspace(0, df['Q/Qexp'].max() + 5, 5)
               )
ax2.set_ylabel('Q/$Q_{exp}$',
               fontsize = 14)

for ax in [ax1, ax2]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.autofmt_xdate()
ax2.xaxis.set_major_formatter(md.DateFormatter('%D %H:%M'))

# ax1.set_title(
#     f'{name}',
#     x = 0.48,
#     y = 0.99,
#     fontsize = 16,
#     fontweight = 'normal'
#     )

plt.show()

# %% scaled residuals vs metadata drivers

#getting the big ass meta data df ready

meta_data_copy = meta_data.copy() # no overwriting !

starttime = meta_data_copy['start_time']

meta_data_copy.drop(columns = ['start_time', 'avg_rain_in/s'], inplace = True)
meta_data_copy.set_index('hour', inplace = True)

copy_list = [df.copy() for df in scaled_residual_ts_dfs]
for df, name in zip(copy_list, factors):
    df.drop(columns = ['datetime'], inplace = True)
    df.set_index('hour', inplace = True)
    
    # adding metadata to each factor df to run the corr matrix
    df = pd.concat([df, meta_data_copy], axis = 1) 
    
    df = df.corr()  # getting pearson coeficcients
    # df = np.square(df)  # getting R2 values
    
    
    fig, ax = plt.subplots(figsize = (24, 20), dpi = 300)
    cax = ax.imshow(df, cmap = 'coolwarm',  # plot matrix w/ chosen colormap
                    interpolation = 'nearest') # round to nearest decimal place for heatmap value
        
    cbar = plt.colorbar(cax, shrink = 0.95, label = 'Pearson Coefficient Correlations')
    cbar.ax.tick_params(labelsize = 18)
    ax.set_xticks(range(len(df.columns))) # set xticks based on the columns in the matrix
    ax.set_xticklabels(df.columns, 
                           rotation = 90, fontsize = 14) # set x values based on columns and rotate for visibility
    ax.set_yticks(range(len(df.columns))) # repeat for yaxis
    ax.set_yticklabels(df.columns, fontsize = 14)
    for (i, j), value in np.ndenumerate(df): # (coordinates) & r-value in loop in values within the mask:
        #     if i > j: # ensure annotated values are in the bottom half of the plot
            if float(f'{value:.2}') >= 0.7 : # only displaying R higher than 0.5
               ax.text(j, i, f'{value:.2}', ha = 'center', va = 'center', color = 'black', fontsize = 7) # adjusting the value:# will change the sigfigs
            if float(f'{value:.1}') <= - 0.3 : 
               ax.text(j, i, f'{value:.1}', ha = 'center', va = 'center', color = 'black', fontsize = 7)
    plt.tight_layout()

    ax.set_title(
        f'{name}',
        x = 0.48,
        y = 0.99,
        fontsize = 24,
        fontweight = 'normal'
        )

    plt.show()
    plt.close()
    
# %% original G-space plot of final 5-factor solution (FPEAK = 0) # uncomment to run
    
# reading in the 100 run 5-factor solution and the rotational analysis
# just modified original code
    
# file containing factor solution folders
names = ['compounds']
base_path = r'C:\\Users\\rossellr\\OneDrive - Colostate\\Desktop\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\pmf\\output'
specific_date_file = '\\20260324\\' # edit! # needs to end in '\\'
path = base_path + specific_date_file
base_naming = '_factor_100_runs' # edit !

num_cmpds = 40 # edit!

# need to know what factors are containined in the output file.
min_factor_soln = 4 # edit !
max_factor_soln = 4 # edit !
factor_soln_range = range(min_factor_soln, (max_factor_soln + 1)) 
factors = []
for i in factor_soln_range:
    factors.append('factor' + f'{i}') # list of factors

file_paths = [] # creating a path to each folder containing solution information
for i in factor_soln_range:
    file_paths.append(path + '\\' + f'{i}' + base_naming + '\\') # getting file path for each factor folder
    
profile_conc = [] # going to store each set of factor dfs in its own dict. 
percent_SppSum = []
percent_factor_total = [] 
factor_ts = []
residual_ts = []
scaled_residual_ts = []

for i, folder in enumerate(file_paths):
    factor = factors[i] # getting each factor name from the factor list
    
    for x in os.listdir(folder): 
        if x.endswith('base.xlsx'): # only pulling from the base file
            full_file = os.path.join(folder, x)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore') # ignoring all the parse formatting warnings
                
                df = pd.read_excel(full_file, sheet_name = 'Profiles', 
                                         skiprows = [0, 1, 2], nrows = num_cmpds) 
                profile_conc.append({'factors' : factor, 'df' : df}) # creating a dict entry of each factor and its df 

                df = pd.read_excel(full_file, sheet_name = 'Profiles', 
                                        skiprows = range(0, num_cmpds + 7), 
                                        nrows = num_cmpds) # change to numcmpds + 7
                percent_SppSum.append({'factors' : factor, 'df' : df})
    
                df = pd.read_excel(full_file, sheet_name = 'Profiles', 
                                        skiprows = range(0, (num_cmpds * 2) + 11), nrows = num_cmpds)
                percent_factor_total.append({'factors' : factor, 'df' : df})
                
                df = pd.read_excel(full_file, sheet_name = 'Contributions', 
                                           skiprows = [0, 1, 2], 
                                           nrows = len(ground_cmpds_ppb) + 5,
                                           parse_dates = [1], date_format = dt_format)
                factor_ts.append({'factors' : factor, 'df' : df})
                
                df = pd.read_excel(full_file, sheet_name = 'Residuals', 
                                        skiprows = [0, 1], 
                                        nrows = len(ground_cmpds_ppb),
                                        parse_dates = [1], 
                                        date_format = dt_format
                                        ) 
                residual_ts.append({'factors' : factor, 'df' : df})
                
                df = pd.read_excel(full_file, sheet_name = 'Residuals', 
                                           skiprows = range(0, len(ground_cmpds_ppb) + 5), 
                                           nrows = len(ground_cmpds_ppb),
                                           parse_dates = [1], 
                                           date_format = dt_format
                                           )
                scaled_residual_ts.append({'factors' : factor, 'df' : df})
                
def dict_to_df(dict):
    df = None
    name = None
    for x in dict.values():
        if isinstance(x, pd.DataFrame):
            df = x
        elif isinstance(x, str):
            name = x
    if df is not None and name is not None:
        df.name = name  # Assign name to the DataFrame within the list; can call using df.name
        return df
    return None

profile_conc_dfs = [] # turning them into dataframes
for dict in profile_conc:
    result = dict_to_df(dict)
    profile_conc_dfs.append(result)
    
percent_SppSum_dfs = []
for dict in percent_SppSum:
    result = dict_to_df(dict)
    percent_SppSum_dfs.append(result)
    
percent_factor_total_dfs = []
for dict in percent_factor_total:
    result = dict_to_df(dict)
    percent_factor_total_dfs.append(result)
    
factor_ts_dfs = []
for dict in factor_ts:
    result = dict_to_df(dict)
    factor_ts_dfs.append(result)
    
residual_ts_dfs = []
for dict in residual_ts:
    result = dict_to_df(dict)
    residual_ts_dfs.append(result)
    
scaled_residual_ts_dfs = []
for dict in scaled_residual_ts:
    result = dict_to_df(dict)
    scaled_residual_ts_dfs.append(result)
    

EVERYTHING = [profile_conc_dfs, percent_SppSum_dfs, percent_factor_total_dfs, factor_ts_dfs]
for each_list in EVERYTHING:
    for df in each_list:
        df.drop(columns = ['Unnamed: 0'], inplace = True)
        df.rename(columns = {'Unnamed: 1' : 'compounds'}, inplace = True)
for df in factor_ts_dfs:
    df.replace(-999, np.nan, inplace = True)
    df.rename(columns = {'compounds' : 'datetime'}, inplace = True)
    df['hour'] = df['datetime'].round('H')
    df['hour'] = df['hour'].dt.hour


residuals = [residual_ts_dfs, scaled_residual_ts_dfs]
for each_list in residuals:
    for df in each_list:
        df.drop(columns = ['Run'], inplace = True)
        df.rename(columns = {'Date_Time' : 'datetime'}, inplace = True)
        df.replace(-999, np.nan, inplace = True)
        df['hour'] = df['datetime'].round('H')
        df['hour'] = df['hour'].dt.hour


# code used in blocks below
ignore = ['datetime', 'hour']

# now, we're making the original G-Space plot (FPEAK = 0)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

copy_list = [df.copy() for df in factor_ts_dfs]
df = copy_list[0]

df.drop(columns = ['datetime', 'hour'], inplace = True)

sns.set(font_scale = 1.5) # setting font scale for all subplots

plot = sns.PairGrid(df,  # create the grid of plots for all numeric data columns in the df 
                    corner = True, # only plots bottom half 
                    layout_pad = 0.05 # shrinking space between subplots
                    )
 
plot.map(sns.scatterplot, # plots a scatter plot on each grid 
         color = 'rebeccapurple',
         #alpha = 0.75,
         s = 60, # markersize
         ) 

for ax in plot.axes.flatten(): # for every axes in the subplot:
    if ax is not None: # ignoring the ones that we are hiding b/c of Corner = True
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, prune = None)) # setting equal gride lines despite scale
        ax.yaxis.set_major_locator(MaxNLocator(nbins = 5, prune = None))
        ax.grid(True)  # Optional: show grid lines
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # formatting decimal points 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_facecolor('white')
        ax.grid(color = 'gray', alpha = 0.6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('gray')   # Box color
            
fig.tight_layout()
plt.show()
 
# %% rotational analysis # uncomment to run

# neg_fpeak_min = 0.1 # edit ! # absolute value of fpeak
# neg_fpeak_max = 0.3 # edit !

# pos_fpeak_min = 0.1 # edit! 
# pos_fpeak_max = 0.2 # edit! 


# # negative FPEAK values
# fpeak_values = []
# for i in np.arange(neg_fpeak_min, (neg_fpeak_min + 1), 0.1):
#     fpeak_values.append(i)
# fpeak_values = [-x for x in fpeak_values]
# fpeak_values = [round(x, 1) for x in fpeak_values]

# fpeak_neg_pt1 = []
# fpeak_neg_pt2 = []
# fpeak_neg_pt3 = []

# names = ['empty_1', 'datetime', 'Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5']

# for i, folder in enumerate(file_paths):
#     factor = factors[i] # getting each factor name from the factor list
    
#     for x in os.listdir(folder): 
#         if x.endswith('_fpeak.xlsx'): # only pulling from the base file
#             full_file = os.path.join(folder, x)
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore') # ignoring all the parse formatting warnings
                
#                 fpeak_neg_pt1 = pd.read_excel(full_file, sheet_name = 'Contributions', 
#                                            skiprows = range(0, 3), 
#                                            nrows = len(ground_cmpds_ppb),
#                                            parse_dates = [1], 
#                                            date_format = dt_format,
#                                            header = None,
#                                            names = names
#                                            )
#                 fpeak_neg_pt2 = pd.read_excel(full_file, sheet_name = 'Contributions', 
#                                            skiprows = range(0, 79), 
#                                            nrows = len(ground_cmpds_ppb),
#                                            parse_dates = [1], 
#                                            date_format = dt_format,
#                                            header = None,
#                                            names = names
#                                            )
#                 fpeak_neg_pt3 = pd.read_excel(full_file, sheet_name = 'Contributions', 
#                                            skiprows = range(0, 155), 
#                                            nrows = len(ground_cmpds_ppb),
#                                            parse_dates = [1], 
#                                            date_format = dt_format,
#                                            header = None,
#                                            names = names
#                                            )
                
# datetime = fpeak_neg_pt1['datetime']
    
# fpeaks = [fpeak_neg_pt1, fpeak_neg_pt2, fpeak_neg_pt3]
# for df in fpeaks:
#     df.drop(columns = ['empty_1'], inplace = True)

 
# fpeaks_copies = [df.copy() for df in fpeaks] # no overwriting! 
# for df, name in zip(fpeaks_copies, fpeak_values):
#     df.drop(columns = ['datetime'], inplace = True)
    
#     sns.set(font_scale = 1.5) # setting font scale for all subplots

#     plot = sns.PairGrid(df,  # create the grid of plots for all numeric data columns in the df 
#                         corner = True, # only plots bottom half 
#                         layout_pad = 0.05 # shrinking space between subplots
#                         )
     
#     plot.map(sns.scatterplot, # plots a scatter plot on each grid 
#              color = 'plum',
#              #alpha = 0.75,
#              s = 60, # markersize
#              ) 

#     for ax in plot.axes.flatten(): # for every axes in the subplot:
#         if ax is not None: # ignoring the ones that we are hiding b/c of Corner = True
#             ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, prune = None)) # setting equal gride lines despite scale
#             ax.yaxis.set_major_locator(MaxNLocator(nbins = 5, prune = None))
#             ax.grid(True)  # Optional: show grid lines
#             ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # formatting decimal points 
#             ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#             ax.set_facecolor('white')
#             ax.grid(color = 'gray', alpha = 0.6)
#             for spine in ax.spines.values():
#                 spine.set_visible(True)
#                 spine.set_linewidth(1)
#                 spine.set_edgecolor('gray')   # Box color
                
#     plt.title(f'FPEAK = {name}',
#               loc = 'left',
#               x = -1.5,
#               y = 4
#               )
           
#     fig.tight_layout()
#     plt.show()
    
# # positive FPEAK values
# fpeak_values = []
# for i in np.arange(pos_fpeak_min, (pos_fpeak_max + 1), 0.1):
#     fpeak_values.append(i)
# fpeak_values = [round(x, 1) for x in fpeak_values]

# fpeak_pos_pt1 = []
# fpeak_pos_pt2 = []


# names = ['empty_1', 'datetime', 'Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5']

# for i, folder in enumerate(file_paths):
#     factor = factors[i] # getting each factor name from the factor list
    
#     for x in os.listdir(folder): 
#         if x.endswith('_fpeak.xlsx'): # only pulling from the base file
#             full_file = os.path.join(folder, x)
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore') # ignoring all the parse formatting warnings
                
#                 fpeak_pos_pt1 = pd.read_excel(full_file, sheet_name = 'Contributions', 
#                                            skiprows = range(0, 41), 
#                                            nrows = len(ground_cmpds_ppb),
#                                            parse_dates = [1], 
#                                            date_format = dt_format,
#                                            header = None,
#                                            names = names
#                                            )
#                 fpeak_pos_pt2 = pd.read_excel(full_file, sheet_name = 'Contributions', 
#                                            skiprows = range(0, 117), 
#                                            nrows = len(ground_cmpds_ppb),
#                                            parse_dates = [1], 
#                                            date_format = dt_format,
#                                            header = None,
#                                            names = names
#                                            )
                
                
# datetime = fpeak_pos_pt1['datetime']
    
# fpeaks = [fpeak_pos_pt1, fpeak_pos_pt2]
# for df in fpeaks:
#     df.drop(columns = ['empty_1'], inplace = True)

 
# fpeaks_copies = [df.copy() for df in fpeaks] # no overwriting! 
# for df, name in zip(fpeaks_copies, fpeak_values):
#     df.drop(columns = ['datetime'], inplace = True)
    
#     sns.set(font_scale = 1.5) # setting font scale for all subplots

#     plot = sns.PairGrid(df,  # create the grid of plots for all numeric data columns in the df 
#                         corner = True, # only plots bottom half 
#                         layout_pad = 0.05 # shrinking space between subplots
#                         )
     
#     plot.map(sns.scatterplot, # plots a scatter plot on each grid 
#              color = 'plum',
#              #alpha = 0.75,
#              s = 60, # markersize
#              ) 

#     for ax in plot.axes.flatten(): # for every axes in the subplot:
#         if ax is not None: # ignoring the ones that we are hiding b/c of Corner = True
#             ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, prune = None)) # setting equal gride lines despite scale
#             ax.yaxis.set_major_locator(MaxNLocator(nbins = 5, prune = None))
#             ax.grid(True)  # Optional: show grid lines
#             ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # formatting decimal points 
#             ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#             ax.set_facecolor('white')
#             ax.grid(color = 'gray', alpha = 0.6)
#             for spine in ax.spines.values():
#                 spine.set_visible(True)
#                 spine.set_linewidth(1)
#                 spine.set_edgecolor('gray')   # Box color
                
#     plt.title(f'FPEAK = {name}',
#               loc = 'left',
#               x = -1.5,
#               y = 4
#               )
           
#     fig.tight_layout()
#     plt.show()

   
    
