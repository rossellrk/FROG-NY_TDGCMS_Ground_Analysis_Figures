# -*- coding: utf-8 -*-
"""
Re-analysis of the data after reviewer comments. 
Based on 20261019_full_reanalysis_V2 and 2026109_data_tidy_pmf_reinput versions
@author: rossellr
"""

# %% Importing libaries

# module imports with: conda install -c conda-forge missing_module_here

import os
# import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
# from matplotlib import cbook
from matplotlib import colors
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
#from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator #, MultipleLocator
import matplotlib.patches as mpatches
#from matplotlib.cbook import boxplot_stats
import numpy as np
# import math as mat
import scipy.stats as scipy
#import statsmodels.api as sm
import matplotlib.dates as md
# from matplotlib.dates import DateFormatter
import datetime as datetime
from datetime import timedelta
from windrose import WindroseAxes
# from matplotlib.ticker import FormatStrFormatter
# from matplotlib.ticker import ScalarFormatter
# import time as time
from collections import defaultdict
#import molmass
from molmass import Formula
import warnings
# import random
import textwrap
#import seaborn as sns

from pathlib import Path

# turning off italicized subperscripts and such
plt.rcParams["mathtext.default"] = 'regular'

# %% Defining all fuctions

plt.rcParams["mathtext.default"] = 'regular' # getting rid of italics with subscripts
plt.rcParams["mathtext.fontset"] = 'dejavusans' # setting font for subscripted numbers

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

# function to simple correlation plots
def simple_corr_plot (axis, x, y, df, num_dec, sci_note, show_slope, show_r, show_both,
                   color, linecolor, lw, linestyle, label_fontsize, tick_fontsize, 
                   fontweight, markersize, legend_fontsize, loc,
                   x_unit, y_unit):
    
    import statsmodels.api as sm
    df_all = df.copy()
    
    # OLS model:
    y_variable_all = df_all[y]
    x_variable_all = df_all[x]
    pearson_all = scipy.pearsonr(x_variable_all, y_variable_all).statistic # getting r-value
    x_variable_all = sm.add_constant(x_variable_all)
    model_all = sm.OLS(y_variable_all, x_variable_all).fit() # running ols model
    y_pred_all = model_all.predict(x_variable_all)
    
    
    if sci_note == True:
        note = 'E'
    else:
        note = 'f'
    
    if show_both == True:
        label = f'slope = {model_all.params.iloc[1]:{num_dec}{note}} (±{model_all.bse.iloc[1]:{num_dec}{note}})\nr: {pearson_all:.2f}'
    elif show_slope == True:
        label = f'slope = {model_all.params.iloc[1]:{num_dec}f} (±{model_all.bse.iloc[1]:{num_dec}f})'
    elif show_r == True:
        label = f'r: {pearson_all:.2f}'
    else:
        label = ''
    
    # creating the plot
    axis.plot(df_all[x], df_all[y],
            linestyle = '',
            marker = 'o',
            markersize = markersize,
            color = color,
            mec = 'black',
            alpha = 0.7
            )
    axis.plot(df_all[x], y_pred_all,
            color = linecolor,
            lw = lw,
            alpha = 0.8,
            linestyle = linestyle,
           # dashes = (5, 10), # manually changing the dash spacing of the line
           # label = f'{model_all.params.iloc[1]:0.2E}'
            label = label
            )
    axis.legend(
        fontsize = legend_fontsize,
        loc = loc
        )
    axis.set_xticks(
        ticks = np.linspace(df_all[x].min(), df_all[x].max(), 5),
        labels = np.linspace(df_all[x].min(), df_all[x].max(), 5),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_yticks(
        ticks = np.linspace(df_all[y].min(), df_all[y].max(), 5),
        labels = np.linspace(df_all[y].min(), df_all[y].max(), 5),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_xlabel(f'{x} ({x_unit})', 
                  size = label_fontsize, 
                  fontweight = fontweight)
    axis.set_ylabel(f'{y} ({y_unit})',
                  size = label_fontsize, 
                  fontweight = fontweight)
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    


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


# function for a timeseries plot w/ tracer compounds
def ts_plot(chosen_df, tracer, axis):
    axis.plot(chosen_df.index, chosen_df[tracer], 
            label = tracer,
            linewidth = 3,
            marker = 'o',
            markeredgecolor = 'black',
            markerfacecolor = None,
            markersize = 12,
            color = cmpd_colors_dict[tracer]
            )
    axis.legend(loc = 'lower left',
                fontsize = 'large')
    axis.set_yticks(np.linspace(0, (chosen_df[tracer].max() + 0.1), 3),
        #np.linspace(chosen_df[tracer].max() + (chosen_df[tracer].max() * 0.1), # adding some space to top of axis
                                #3),
                    labels = np.linspace(chosen_df[tracer].min(), chosen_df[tracer].max(), 3),
                    fontsize = 13,
                    ) 
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))



# removing 'peak' labels and other text from cmpd names
def remove_text(text, to_replace):
    text = str(text)
    if isinstance(to_replace, str):
        to_replace = [to_replace]
    for char in to_replace:
        text = text.replace(char, '') 
    return text

# removing commas and replace with underscores
def remove_commas(text):
    chars_to_replace = [','] # text to remove
    for char in chars_to_replace:
        text = str(text)
        text = text.replace(char, '_') 
    return text

# function to create dataframes out of dictionaries
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

# function to add rectangular patches to TS graphs
def rect_box(start_dt, end_dt, axis, hatch):
    start = md.date2num(start_dt) # converting dt object to numerical object for axis coordinates
    end = md.date2num(end_dt)
    width = end - start # width of rectangle
    box = Rectangle(
        xy = (start, start_y_point), # xy coordinates of bottom-left corner
        width = width,
        height = height,
        alpha = set_alpha,
        color = box_color,
        hatch = hatch
        )
    ax = axis
    ax.add_patch(box)
    
# function to add rectangular patches to non-TS graphs
def points_rect_box(start, end, axis, color, hatch):
    width = end - start # width of rectangle
    box = Rectangle(
        xy = (start, start_y_point), # xy coordinates of bottom-left corner
        width = width,
        height = height,
        alpha = set_alpha,
        color = color,
        hatch = hatch
        )
    ax = axis
    axes[ax].add_patch(box)

# function to add totals to the top of each bar in the bar plot
def add_bar_totals(x, y, total, units, fontsize, fontweight): 
    for i in range(len(x)):
        plt.text(x = i, y = y + (y * 0.02) ,  # setting x and y positions of labels - centering with bar and then slightly above top of bar
                 s = f'{total[i]:0.1f} {units}', # setting and formatting what appears in the label
                 ha = 'center',
                 fontsize = fontsize,
                 fontweight = fontweight
                 )

# function to add totals to the top of each bar in the percentage bar plot
def add_percent_bar_totals(axis, x, y, total, units, fontsize, fontweight): 
    for i in range(len(x)):
        axis.text(x = i, y = y + (y * 0.02) ,  # setting x and y positions of labels - centering with bar and then slightly above top of bar
                 s = f'{total[i]:0.2f} {units}', # setting and formatting what appears in the label
                 ha = 'center',
                 fontsize = fontsize,
                 fontweight = fontweight
                 )

# function to create correlation plots, ignoring outlier points
def ols_corr_plot_outliers (axis, x, y, df, index, 
                   color, label_fontsize, tick_fontsize, 
                   fontweight, markersize, legend_fontsize, loc):
    
    import statsmodels.api as sm
    
    df_all = df.copy()
    df_no_out =  df_all.drop(index = index) # dropping outliers
    df_no_out.reset_index(inplace = True, drop = True) # dropping old index column
    # OLS model with all points:
    y_variable_all = df_all[y]
    x_variable_all = df_all[x]
    pearson_all = scipy.pearsonr(x_variable_all, y_variable_all) # getting r-value
    x_variable_all = sm.add_constant(x_variable_all)
    model_all = sm.OLS(y_variable_all, x_variable_all).fit() # running ols model
    y_pred_all = model_all.predict(x_variable_all)
    # OLS model with outliers removed
    y_variable_no_out = df_no_out[y]
    x_variable_no_out = df_no_out[x] 
    pearson_no_out = scipy.pearsonr(x_variable_no_out, y_variable_no_out) # getting r-value
    x_variable_no_out = sm.add_constant(x_variable_no_out)
    model_no_out = sm.OLS(y_variable_no_out, x_variable_no_out).fit() # running ols model
    y_pred_no_out = model_no_out.predict(x_variable_no_out)
    
    # creating the plot
    axis.plot(df_all[x], df_all[y],
            linestyle = '',
            marker = 'o',
            markersize = markersize,
            color = color,
            mec = 'black',
            alpha = 0.2)
    axis.plot(df_all[x], y_pred_all,
            color = 'darkgray',
            alpha = 0.5,
            linestyle = '--',
            dashes = (5, 10), # manually changing the dash spacing of the line
            label = f'All Values\nY = {model_all.params.iloc[1]:0.2f}x\n r: {pearson_all[0]:0.2f}')
    axis.plot(df_no_out[x], df_no_out[y],
            linestyle = '',
            marker = 'o',
            markersize = markersize,
            color = color,
            mec = 'black'
            )
    axis.plot(df_no_out[x], y_pred_no_out,
            color = 'black',
            linestyle = '--',
            dashes = (5, 10), # manually changing the dash spacing of the line
            label = f'No Outliers\nY = {model_no_out.params.iloc[1]:0.2f}x\n r: {pearson_no_out[0]:0.2f}')
    axis.legend(
        #title = 'OLS Models',
        fontsize = legend_fontsize,
        loc = loc
        )

    axis.set_xticks(
        ticks = np.linspace(df_all[x].min(), df_all[x].max(), 5),
        labels = np.linspace(df_all[x].min(), df_all[x].max(), 5),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_yticks(
        ticks = np.linspace(df_all[y].min(), df_all[y].max(), 5),
        labels = np.linspace(df_all[y].min(), df_all[y].max(), 5),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_xlabel(f'{x} ($ppb_v$)', 
                  size = label_fontsize, 
                  fontweight = fontweight)
    axis.set_ylabel(f'{y} ($ppb_v$)',
                  size = label_fontsize, 
                  fontweight = fontweight)
    
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
# function to calculate number densities from ppb space
def number_density(conc_ppb):
    avo_constant = 6.022E23
    R_constant = 0.08206
    temp_k = meta_data['avg_temp_c'].median() + 273.13 # conver to K
    pressure_atm = meta_data['avg_press_inHg'].median() * 0.03342 # convert from inHg to atm
    vol_conv = 1000 # L to mL to cm3
    
    number_density_value = ((conc_ppb * avo_constant * 1E-9) / ((R_constant * temp_k) / pressure_atm)) / vol_conv 
    return(number_density_value)
    
# function to calculate ug/m3 from ppb
def ug_m3_from_ppb(conc_ppb, ref_data, meta_data):
    R = 8.206E-5
    # convert to arrays (NO pandas alignment issues anymore)
    conc = conc_ppb.to_numpy()
    mw = ref_data['mw_g_mol'].loc[conc_ppb.columns].to_numpy()
    T = meta_data['avg_temp_c'].to_numpy() + 273.15
    P = meta_data['avg_press_inHg'].to_numpy() * 0.03342

    # core computation
    ug = conc * mw * 1E-9 * 1E6
    ug = ug * P[:, None]
    ug = ug / (R * T[:, None])

    return pd.DataFrame(ug, index=conc_ppb.index, columns=conc_ppb.columns)
# def ug_m3_from_ppb (conc_ppb, meta_data, ref_data):
#     R_constant = 8.206E-5 # m3*atm/K*mol
#     temp_k = meta_data['avg_temp_c'] + 273.13 # conver to K
#     pressure_atm = meta_data['avg_press_inHg'] * 0.03342 # convert from inHg to atm
#     mw = ref_data.loc['mw_g_mol']
    
#     mass_value = ((conc_ppb * 1E-9 * mw * 1E-6) / ((R_constant * temp_k) / pressure_atm)) 
#     return(mass_value)

# function to read chemical formulas
def parse_formula(chem_formula): # create a function that counts the number of elements in the compound names
    pattern = r'([A-Z][a-z]?)(\d*)' # pattern: Big Letter with potenital second small letter (covers C and He, for example), followed by a # (default 1)
    counts = defaultdict(int) # creates a dictionary of the counted values
    for (element, count) in re.findall(pattern, chem_formula): # recalls the set pattern and then perfroms the count for each compound
        counts[element] += int(count) if count else 1 # starts with a default count of 1 for each element
    return counts

# function to calculate elemental ratios for Van Krevelan plots
def calc_ratios(compounds_list): 
    ratios = [] # create empty list
    for x in compounds_list: # for every cmpd in the cmpd list:
        counts = parse_formula(x) # perform the formula function, defined in the first code block when reading in the PTR file
        num_o = counts.get('O', 0) # calc the ratios based on the values stored in the created dictionary for each cmpd
        num_h = counts.get('H', 0)
        num_c = counts.get('C', 0)
        if num_c > 0: # calc the ratio if the # of C >1
            ratio_h_c = num_h / num_c
            ratio_o_c = num_o / num_c
        else:
            ratio_h_c = float('inf') # otherwise just store an inf value
            ratio_o_c = float('inf')
        ratios.append((x, ratio_h_c, ratio_o_c)) # add the ratios to the empty list we created earlier with the compound names (x variable)
    df = pd.DataFrame(ratios, columns = ['compound', 'H:C', 'O:C']) # create a df from the stored list 
    df['H:C'] = df['H:C'].round(2) # round the ratios to 2 decimal places
    df['O:C'] = df['O:C'].round(2)
    return df # return the new df

# %% Switching between personal and work computers

# Uncomment to switch between b/c this is redonk:
# work
drive_folder = Path(r'C:\Users\rossellr\OneDrive - Colostate\Desktop')

# home
#drive_folder = Path(r'C:\\Users\\rosse\\OneDrive - Colostate\\Desktop')


# %% Reading in and organizing metadata 

# turning off "A value is trying to be set on a copy of a slice " warning b/c its annoying
pd.options.mode.chained_assignment = None  # default='warn' 


dt_format = '%Y-%m-%d %H:%M:%S'
today = datetime.date.today().strftime('%Y%m%d')


# reading in meta data of weather and trace gas data (pre-averaged by GC time blocks)
path = drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / '2025' / 'ground_data_workup' / 'meta_data'
string = '_ground_meta_data.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(path, string)
meta_data = pd.read_csv(updated_file, 
                        parse_dates = ['start_time', 'end_time'])

meta_data.drop(columns = ['sample_name', 'end_time'], inplace = True )
meta_data['hour'] = meta_data['start_time'].round('H')
meta_data['hour'] = meta_data['hour'].dt.hour
full_meta_data = meta_data.copy()
# grouping meta data by hour of the day for future diel plots
grouped_metadata = meta_data.groupby(['hour']).median()
#grouped_metadata.drop(columns = ['start_time'], inplace = True)


# reading in full weather data (not averaged by GC blocks)
weather_file = drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / '2025' / 'full_weather_data_20230705_20230814.csv'
# Reading in the weather metadata file
col_names_weather = ['dt', 'temp_c', 'wind_speed_m/s', 'wind_dir_deg', 'hourly_rain_in/s', # naming columns
             'relative_press_inHg', 'RH_pct', 'solar_rad_W/m2']
weather = pd.read_csv(weather_file, header = None, usecols = [1, 2, 5, 8, 9, 15, 16, 18], names = col_names_weather)
weather.drop(weather.index[0], inplace = True) # drop first row with additional header info
weather['temp_c'].dropna(axis = 0) # remove NA values
weather['dt'] = pd.to_datetime(weather['dt']) # transform into python datetime objects
weather['temp_c'] = [int(float(x)) for x in weather['temp_c']]
weather.set_index('dt', inplace = True)
for col in weather.columns:
    # removing string values
    weather[col] = pd.to_numeric(weather[col], errors = 'coerce')
weather.reset_index(inplace = True)


# reading in full trace gas data (not averaged by GC blocks)
ground_o3_file = pd.read_csv(drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / 'Field Campaign Data' / 'Final Data' / 'O3_FinalTs_20230321.csv')
ground_nox_file = pd.read_csv( drive_folder /  'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / 'Field Campaign Data' / 'Final Data' / 'icartt_frogNox_final20230322.csv')

# ground trace gas data
ground_o3 = pd.DataFrame(ground_o3_file)
ground_o3.drop(columns = ['MidPoint_Times', 'O3_Raw', 'Start_Times_UTC', 'Stop_Times'])
ground_o3['Start_Times'] = pd.to_datetime(ground_o3['Start_Times']) # transform into python datetime objects
ground_o3.rename(columns = {'Start_Times' : 'dt'}, inplace = True) # rename to match code chunks
# o3_mask = ground_o3['O3_Corrected'] == 999 # remove flagged values
# ground_o3 = ground_o3[~o3_mask]
ground_o3 = ground_o3.replace(999, np.nan)

ground_nox = pd.DataFrame(ground_nox_file)
ground_nox.drop(columns = ['NO_raw', 'NO_Corrected', 'NO2_raw', 'NO2_Corrected', 'NOx_raw',
                           'Start_Times_UTC', 'Stop_Times_EDT'], inplace = True)
ground_nox['Start_Times_EDT'] = pd.to_datetime(ground_nox['Start_Times_EDT']) # transform into python datetime objects
ground_nox.rename(columns = {'Start_Times_EDT' : 'dt'}, inplace = True) # rename to match code chunks
# nox_mask = ground_nox['NO_Corrected_Flagged'] == 9999.999999 # remove flagged values
# ground_nox = ground_nox[~nox_mask]
ground_nox = ground_nox.replace(9999.999999, np.nan)
ground_nox.set_index('dt', inplace = True)
ground_nox = ground_nox.loc[datetime.datetime(2023, 7, 9, 0, 0, 0):]
ground_nox = ground_nox.loc[:datetime.datetime(2023, 8, 14, 12, 12, 0)]
ground_nox.reset_index(inplace = True)


# reading in cmpd refs data
path = drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / '2025' / 'ground_data_workup' / 'meta_data'
string = '_cmpd_reference_sheet.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(path, string)
cmpd_refs = pd.read_csv(updated_file)

# only want specific columns
cmpd_refs = cmpd_refs[['cmpd_name', 'CAS', 'Molecular Weight (g/mol)', 'Molecular Formula', 
                       'MIR_ppb',
                       'sat_vapor_pressure_mmHg_25C', 'k_o3_298', 'k_oh_298']]
# renaming columns
cmpd_refs.rename(columns = {'cmpd_name' : 'compound_name',
                            'Molecular Weight (g/mol)' : 'mw_g_mol',
                            'Molecular Formula' : 'formula'}, inplace = True) # getting rid of caps, etc
cmpd_refs['compound_name'] = [remove_text(x, [' peak']) for x in cmpd_refs['compound_name']]
cmpd_refs['compound_name'] = [remove_commas(x) for x in cmpd_refs['compound_name']] # replacing ',' w/ '_'
cmpd_refs.columns = cmpd_refs.columns.str.rstrip() # removing white sapce at the end of the column names
# updating some spelling
cmpd_refs['compound_name'] = cmpd_refs['compound_name'].replace({
    'methyl naphthalene 1' : 'methylnaphthalene 1',
    'methyl naphthalene 2' : 'methylnaphthalene 2',
    'pentanoic acid_ 2-ethylhexyl ester' : 'pentanoic acid_2-ethylhexyl ester'
    })

cmpd_refs.drop([96, 96], inplace = True) # getting rid of MIR source text
# making sure everything is numerical:
cmpd_refs['MIR_ppb'] = pd.to_numeric(cmpd_refs['MIR_ppb'], errors = 'coerce')

# reading in ACSM OA Data
file = drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / '2025' / 'ground_data_workup' / 'meta_data' / 'FROG_ACSM_NR-PM1.csv'
col_names = ['datetime_local', 'total_ug_m3', 'OA_ug_m3', 'no3_ug_m3', 'nh4_ug_m3', 'so4_ug_m3', 'chl_ug_m3']
oa_data = pd.read_csv(file, skiprows = [0], 
                      names = col_names, 
                      parse_dates = ['datetime_local'])



# %% Reading in and organizing compound concentration datafiles 

# reading in the PMF input  file (>25 samples)
path = drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / '2025' / 'ground_data_workup' / 'pmf' / 'input'
string = '_pmf_conc_reanalysis.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(path, string)
pmf_input = pd.read_csv(updated_file,
                               parse_dates = ['start_time'])

pmf_input.rename(columns = {'start_time': 'start_time_edt'}, inplace = True)
pmf_input['hour'] = pmf_input['start_time_edt'].dt.hour
pmf_input.columns = [remove_text(x, [' peak']) for x in pmf_input.columns]
full_pmf_input_cmpd_ts = pmf_input.copy()
# grouping cmpd data by hour of the day for future diel plots
pmf_input['hour'] = pmf_input['start_time_edt'].round('H')
pmf_input['hour'] = pmf_input['hour'].dt.hour

# reading in the most common ground compounds (>25)
path = drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / '2025' / 'ground_data_workup' / 'python' / 'outputs'
string = '_groundVOCs_common_compounds_requant.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(path, string)
common_cmpds_ppb = pd.read_csv(updated_file,
                               parse_dates = ['start_time'])
common_cmpds_ppb.drop(columns = [
    'end_time',
    'sample_id',
    ], inplace = True)
# Replacing "ND" w/ nan and setting all cols but datetime to numeric
common_cmpds_ppb.replace('ND', np.nan, inplace = True)
for col in  common_cmpds_ppb.columns[1:]:
    common_cmpds_ppb[col] = pd.to_numeric(common_cmpds_ppb[col], errors='coerce') 
# grouping cmpd data by hour of the day for future diel plots
common_cmpds_ppb['hour'] = common_cmpds_ppb['start_time'].round('H')
common_cmpds_ppb['hour'] = common_cmpds_ppb['hour'].dt.hour
# spelling updates
common_cmpds_ppb.rename(columns = {
    '4-chlorobenzotrifluoride 1' : '4-chlorobenzotrifluoride'
    }, inplace = True)


#reading in full dataset
path = drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / '2025' / 'ground_data_workup' / 'python' / 'outputs'
string = '_groundVOCs_all_compounds_requant.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(path, string)
full_cmpd_dataset = pd.read_csv(updated_file,
                               parse_dates = ['start_time'])
full_cmpd_dataset.drop(columns = [
    'end_time',
    'sample_id',
    ], inplace = True)
# spelling updates
full_cmpd_dataset.rename(columns = {
    'hexyl acetate 1' : 'hexyl acetate',
    '4-chlorobenzotrifluoride 1' : '4-chlorobenzotrifluoride'
    }, inplace = True)
# Replacing "ND" w/ nan and setting all cols but datetime to numeric
full_cmpd_dataset.replace('ND', np.nan, inplace = True)
for col in  full_cmpd_dataset.columns[1:]:
    full_cmpd_dataset[col] = pd.to_numeric(full_cmpd_dataset[col], errors='coerce')
# need nearest hour for future diel plots
full_cmpd_dataset['hour'] = full_cmpd_dataset['start_time'].round('H')
full_cmpd_dataset['hour'] = full_cmpd_dataset['hour'].dt.hour








# %% Reading factor solution data

'''
Pulling and organizing factor solutions from the PMF output files, into organized lists that contain dataframes for each factor solution.
'''

# file containing factor solution folders
names = ['compounds']
base_path = drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / '2025' / 'ground_data_workup' / 'pmf' / 'output' 
specific_date_file = '20260324' # edit!
path = base_path / specific_date_file
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
    file_paths.append(path / f'{i}{base_naming}') # getting file path for each factor folder
    
profile_conc = pd.DataFrame()
percent_SppSum = pd.DataFrame()
percent_factor_total = pd.DataFrame()
factor_ts = pd.DataFrame()
residual_ts = pd.DataFrame()
scaled_residual_ts = pd.DataFrame()


for i, folder in enumerate(file_paths):
    factor = factors[i] # getting each factor name from the factor list
    
    for x in os.listdir(folder): 
        if x.endswith('base.xlsx'): # only pulling from the base file
            full_file = os.path.join(folder, x)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore') # ignoring all the parse formatting warnings
                profile_conc = pd.read_excel(full_file, sheet_name = 'Profiles', 
                                         skiprows = [0, 1, 2], nrows = num_cmpds) 

               
                percent_SppSum = pd.read_excel(full_file, sheet_name = 'Profiles', 
                                        skiprows = range(0, num_cmpds + 7), nrows = num_cmpds)

    
                percent_factor_total = pd.read_excel(full_file, sheet_name = 'Profiles', 
                                        skiprows = range(0, (num_cmpds * 2) + 11), nrows = num_cmpds)

                
                factor_ts = pd.read_excel(full_file, sheet_name = 'Contributions', 
                                           skiprows = [0, 1, 2], 
                                           nrows = len(pmf_input) + 5,
                                           parse_dates = [1], 
                                           date_format = dt_format
                                           )

                
                residual_ts = pd.read_excel(full_file, sheet_name = 'Residuals', 
                                        skiprows = [0, 1], 
                                        nrows = len(pmf_input) + 1,
                                        parse_dates = [1], 
                                        date_format = dt_format
                                        ) 

                
                scaled_residual_ts = pd.read_excel(full_file, sheet_name = 'Residuals', 
                                           skiprows = range(0, 40), 
                                           nrows = len(pmf_input) + 1,
                                           parse_dates = [1], 
                                           date_format = dt_format
                                           )

EVERYTHING = [profile_conc, percent_SppSum, percent_factor_total, factor_ts]
for df in EVERYTHING:    
    df.drop(columns = ['Unnamed: 0'], inplace = True)
    df.rename(columns = {'Unnamed: 1' : 'compounds'}, inplace = True)
    
cmpd_dfs = [profile_conc, percent_SppSum, percent_factor_total]
for df in cmpd_dfs:
    df['compounds'] = [remove_text(x, [' peak']) for x in df['compounds']]

# have to handle TS df seperately
factor_ts.rename(columns = {'compounds' : 'datetime'}, inplace = True)
factor_ts['hour'] = factor_ts['datetime'].round('H')
factor_ts['hour'] = factor_ts['hour'].dt.hour
    
residuals = [residual_ts, scaled_residual_ts]
for df in residuals:
    df.drop(columns = ['Run'], inplace = True)
    df.rename(columns = {'Date_Time' : 'datetime'}, inplace = True)
    df['hour'] = df['datetime'].dt.hour
    

   
# OKAY
# so now we have 3 lists, each containin 7 dataframes. Each df contains data for a different factor solution.

# code used in blocks below
ignore = ['datetime', 'hour']

# %% Dictionaries of colors and linestyles

# setting colors for 4 factor solution to always be consistent:
color_dict = {
    'Factor 1' : '#1A1B41', # dark blue
    'Factor 3' : '#92DCE5', # light blue
    'Factor 2' : '#4E3822', # brown
    'Factor 4' : '#F15946', # orange
    'total_oh' : 'darkmagenta',
    'Total OH Reactivity' : 'darkmagenta'
    }

linestyle_dict = {
    'Factor 1' : 'dashdot',
    'Factor 4' : 'solid', 
    'Factor 3' : 'solid',
    'Factor 2' : 'dashdot',
    'total_oh' : 'dashed',
    'Total OH Reactivity' : 'dashed',
    }

hatch_dict = {
    'ethylhexyl benzoate' : '/',
    'benzaldehyde' : '/',
    'limonene' : '\\',
    'a-pinene' : '\\',
    'decamethylcyclopentasiloxane' : '/',
    'hexamethylcyclotrisiloxane' : '/', 
    'dodecamethylcyclohexasiloxane' : '/',
    'carbitol' : '/'
    }
edge_color_dict = {
    'limonene' : 'lightgreen'
    }

cmpd_colors_dict = { # setting color of cmpd based on source
     'toluene' : '#555555', 
     'ethylbenzene' : '#6F7887', 
     'decamethylcyclopentasiloxane' : '#ffcc75', 
     'benzyl acetate' : '#AA336A', 
     'allyl heptanoate' : '#e493a1', 
     'naphthalene' : '#5e7b93', 
     '2-methyl-undecanal' : '#e4cbd1', 
     'diphenyl ether' : '#ff859c', 
     'o-xylene' : '#7D6F6F',
     'a-pinene' : '#559b39', 
     'benzaldehyde' : '#dc8e43', 
     '3-ethyl toluene' : '#7D6783', 
     'limonene' : '#41772c', 
     'benzyl alcohol' : '#F5477E',
     'camphor' : '#c44f69', 
     'methyl salicylate' : '#d577af', 
     'benzothiazole' : '#777793', 
     '4-tert-butylcyclohexyl acetate 1' : '#ffd2da',
     '4-tert-butylcyclohexyl acetate 2' : '#ffe3ea', 
     'hexadecane' : '#715412',
     'dodecane' : '#795C34',
     'benzene' : '#766856',
     'hexamethylcyclotrisiloxane' : '#DAC17C',
     'tetrachloroethylene' : '#b6a999', 
     '4-chlorobenzotrifluoride' : '#b19b9b', 
     'm-xylene' : '#B0B0B0', 
     'propylbenzene' : '#702b18', 
     'phenol' : '#7b4a43', 
     'acetophenone' : '#ffbdd3', 
     'nonanal' : '#6892ec',  
     'nopinone' : '#665d1e' ,
     'sabina ketone' : '#cc8899',
     'decanal' : '#219ed6',
     'phenoxyacetone' : '#d8b6e6', 
     'phenoxyisopropanol' : '#ec5fbe',
     'bornyl acetate' : '#ba7dd2', 
     '2-tert-butylcyclohexyl acetate 1' : '#a63375',  
     '2-tert-butylcyclohexyl acetate 2' : '#A94064',
     'methylnaphthalene 1' :  '#54424a', 
     '6-tert-butyl-o-cresol' : '#ffb8f8', 
     'methylnaphthalene 2' : '#86727b', 
     'dodecamethylcyclohexasiloxane' : '#F9E076',
     'tetradecane' : '#532915', 
     'tricyclodecenyl acetate' : '#F33A6A', 
     '2-ethyl-1-hexanol' : '#D3D3D3',
     'ethylhexyl benzoate' : '#C9BB8E',
     'heptadecane' : '#412e33',
     'txib' : '#4a3e20',
     'other' : '#87ceeb',
     'total_oh' : 'darkmagenta',
     'summed_ofp' : 'lightpink',
     'Total OH Reactivity' : 'darkmagenta',
     'homosalate' : '#BE6E46',
     'butyl acetate' : '#d17f66',
     'octadecane' : '#776f2c',
     '1_4-dichlorobenzene' : '#E6BE8A',
     '3-carene' : '#98FB98',
     'B-pinene' : '#50C878',
     'B-myrcene' : '#2E8B57',
     'camphene' : '#00563B',
     'a-terpineol' : '#588157',
     'B-ocimene' : '#3A5A40',
     'sabinene' : '#29AB87',
    'dimethylsilanediol' : '#FFE135',
    'm-ethylstyrene' : '#7D7D7D',
    '1,2,4-trifluorobenzene' : '#7C6E7F',
    '1_2_4-trifluorobenzene' : '#7C6E7F',
    '1,4-dichlorobenzene' : '#C0C0C0',
    'decane' : '#E2C28E',
    'octamethylcyclotetrasiloxane' : '#FFFACD',
    'carbitol' : '#FAFAD2',
    'propylene glycol' : '#F1C3DD',
    'geranyl acetone' : '#FADDE1',
    'a-terpinyl acetate' : '#F4ACB7',
    'methyl laurate' : '#FF97B7',
    '4-chlorobenzotrifluoride 1' : '#b19b9b',
    'propyltoluene' : '#848482',
    '2-ethylhexyl ester' : '#4A3728',
    'eicosane' : '#EFDECD',
    'component of texanol' : '#E3DAC9',
    'o-tert-butylphenol' : '#ffebfd',
    'methyl isobutyl ketone' : '#EEDC82',
    'pentanoic acid, 2-ethylhexyl ester' : '#FFB7C5',
    'pentanedioic acid, dimethyl ester' : '#A94064',
    'trimethylsilanol' : '#98817B',
    'trans-B-santalol' : '#90ee90',
    'citronellol' : '#e5fbe5',
    '4-propylbenzaldehyde' : '#E4D00A',
    '5-methyltetralin' : '#D2B55B',
    'B-methyltetralin' : '#FFFFE0',
    'butyl carbamate' : '#FFD1DC',
    'dibutoxymethane' : '#FFFFCC',
    'diphenyl sulfide' : '#FFEF00',
    'hexyl acetate' : '#EEE8AA',
    'methyl ethyl ketone' : '#F0E68C',
    'methyl tert-butyl ether' : '#FCE883',
    'pentanoic acid_2-ethylhexyl ester' : '#FADA5E',
    '1-methoxy-2-propyl acetate' : '#F8E473'
    }

# Check for duplicate collors
value_to_keys = defaultdict(list)
for key, value in cmpd_colors_dict.items():
    value_to_keys[value].append(key)

duplicates = {value: keys for value, keys in value_to_keys.items() if len(keys) > 1}

if duplicates:
    print("Duplicate colors found:\n")
    for value, keys in duplicates.items():
        print(f"{value}: {keys}")
else:
    print("No duplicate colors found.")
    
missing_cols = set(full_cmpd_dataset.columns) - set(cmpd_colors_dict.keys())
print(missing_cols)


# %% Figure 3 - Factor Distribution

# getting the dfs organized correctly
df = percent_factor_total.copy()  # no overwriting! 
units = '$ppb_v$'
df_total = profile_conc.sum(axis = 0, numeric_only = True)
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

fontsize = 26
fontweight = 'normal'

fig, ax = plt.subplots(figsize = (20, 20), dpi = 300)
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
          linewidth = 1,
          rot = 360,
          width = 0.97,
          )
ax.set_ylabel(
    'Compound Distribution (%)',
    fontsize = fontsize,
    fontweight = fontweight
    )

ax.set_yticklabels(ax.get_yticks(), weight = fontweight) # setting y-ticks to bold
for label in ax.get_xticklabels(): # setting axis labels to bold
    label.set_fontweight(fontweight)

n_rows = len(df)
n_cols = len(df.columns)

# labeling the top 5 compounds making up each factor:
for bar_idx, (index, row) in enumerate(df.iterrows()):
    top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(6).index.tolist()

    for col_idx, col in enumerate(df.columns):
        # Find the correct bar (patch)
        bar = ax.patches[col_idx * n_rows + bar_idx]
        hatch = hatch_dict.get(col, '')  # Hatch only the 2 compounds of interest
        
        if col in top_cols:
            color = cmpd_colors_dict.get(col, 'gray')
            alpha = 1
            edgecolor = 'gray'
            
        else:
            color = cmpd_colors_dict.get(col, 'gray')
            alpha = 0.2
            edgecolor = 'white'
        
        bar.set_color(color)
        bar.set_alpha(alpha)
        bar.set_edgecolor(edgecolor)
        bar.set_hatch(hatch)
        

    cum_height = 0
    for col in df.columns:
        height = row[col]
        mid_point = cum_height + height / 2
        if col in top_cols and height > 0:
                # have to wrap the really long compound labels so they fit in the bar
                wrapped_label = '\n'.join(textwrap.wrap(col, width = 22))  # Adjust width as needed

                ax.text(
                    x = bar_idx,
                    y = mid_point,
                    s = f'{wrapped_label}: ' + f'{height:.1f}%', 
                    ha = 'center',
                    va = 'center',
                    fontsize = 12 if height < 3 else 16,
                    color = 'black'  # or a color that contrasts the bar
                    )
        cum_height += height

# Find columns that are "important" anywhere
top_n = 6
top_cols_global = set()

for _, row in df.iterrows():
    top_cols_global.update(
        pd.to_numeric(row, errors='coerce').nlargest(top_n).index
    )
    
    
labels = ['Industrial', 'Fragrances', 'Biogenic', 'Other']
legend_colors = ['#B0B0B0', 'hotpink', 'Green', '#DC8E43']
# #F33A6A
hatches = ['', '', '\\', '/']

# adding manual legends
patches = []
for i, color, h in zip(labels, legend_colors, hatches):
    leg = mpatches.Patch(facecolor = color, 
                         edgecolor = 'white',
                         label = i,
                         hatch = h)
    patches.append(leg)
ax.legend(handles = patches, 
          loc = 'lower center', 
          bbox_to_anchor = (0.5, -0.1),
          fontsize = 24,
          ncols = 4, 
         #columnspacing = 15
          )

# adding total labels on top 
fontsize = 24
add_percent_bar_totals(ax, range(len(df)), 99, df_total, units, fontsize, fontweight)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.show()
plt.close()

# %% SI Figure 10: Factor Breakdown

# getting the dfs organized correctly
df = percent_factor_total.copy()  # no overwriting! 
units = '$ppb_v$'
df_total = profile_conc.sum(axis = 0, numeric_only = True)
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

fontsize = 26
fontweight = 'normal'

fig, ax = plt.subplots(figsize = (20, 20), dpi = 300)
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
          linewidth = 1,
          rot = 360,
          width = 0.97,
          )
ax.set_ylabel(
    'Compound Distribution (%)',
    fontsize = fontsize,
    fontweight = fontweight
    )

ax.set_yticklabels(ax.get_yticks(), weight = fontweight) # setting y-ticks to bold
for label in ax.get_xticklabels(): # setting axis labels to bold
    label.set_fontweight(fontweight)

n_rows = len(df)
n_cols = len(df.columns)

# labeling the top 5 compounds making up each factor:
for bar_idx, (index, row) in enumerate(df.iterrows()):
    top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(6).index.tolist()

    for col_idx, col in enumerate(df.columns):
    #     # Find the correct bar (patch)
         bar = ax.patches[col_idx * n_rows + bar_idx]
         hatch = hatch_dict.get(col, '')  # Hatch only the 2 compounds of interest
        
    #     if col in top_cols:
    #         color = cmpd_colors_dict.get(col, 'gray')
    #         alpha = 1
    #         edgecolor = 'gray'
            
    #     else:
    #         color = cmpd_colors_dict.get(col, 'gray')
    #         alpha = 0.2
    #         edgecolor = 'white'
        
    #    bar.set_color(color)
    #    bar.set_alpha(alpha)
    #    bar.set_edgecolor(edgecolor)
         bar.set_hatch(hatch)
        

    cum_height = 0
    for col in df.columns:
        height = row[col]
        mid_point = cum_height + height / 2
        if col in top_cols and height > 0:
                # have to wrap the really long compound labels so they fit in the bar
                wrapped_label = '\n'.join(textwrap.wrap(col, width = 22))  # Adjust width as needed

                ax.text(
                    x = bar_idx,
                    y = mid_point,
                    s = f'{wrapped_label}: ' + f'{height:.1f}%', 
                    ha = 'center',
                    va = 'center',
                    fontsize = 12 if height < 3 else 16,
                    color = 'black'  # or a color that contrasts the bar
                    )
        cum_height += height

# Find columns that are "important" anywhere
# top_n = 6
# top_cols_global = set()

# for _, row in df.iterrows():
#     top_cols_global.update(
#         pd.to_numeric(row, errors='coerce').nlargest(top_n).index
#     )

labels = ['Industrial', 'Fragrances', 'Biogenic', 'Other']
legend_colors = ['#B0B0B0', 'hotpink', 'Green', '#DC8E43']
# #F33A6A
hatches = ['', '', '\\', '/']

# adding manual legends
patches = []
for i, color, h in zip(labels, legend_colors, hatches):
    leg = mpatches.Patch(facecolor = color, 
                         edgecolor = 'white',
                         label = i,
                         hatch = h)
    patches.append(leg)
ax.legend(handles = patches, 
          loc = 'lower center', 
          bbox_to_anchor = (0.5, -0.1),
          fontsize = 24,
          ncols = 4, 
         #columnspacing = 15
          )

# adding total labels on top 
fontsize = 24
add_percent_bar_totals(ax, range(len(df)), 99, df_total, units, fontsize, fontweight)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.show()
plt.close()

# %% SI Figure: Full Factor TS w/ 2 subplots and smoke/heat events

def ts_plot2(chosen_df, axis, color, linestyle):
    axis.plot(df.index, df[col], 
            label = col,
            linewidth = 2,
            marker = 'o',
            markeredgecolor = 'black',
            markerfacecolor = None,
            markersize = 4,
            color = color,
            linestyle = linestyle
            )
    axis.set_yticks(
        ticks = np.linspace(0, df[col].max(), 4),
        labels = np.linspace(0, df[col].max(), 4)
        )


df = factor_ts.copy() # no overwriting! 
df.set_index('datetime', inplace = True)
df.drop(columns = ['hour'], inplace = True)
    
fig, (ax4, ax1, ax2) = plt.subplots(
    nrows = 3,
    ncols = 1,
    sharex = True,
    figsize = (8, 6),
    dpi = 300)

# plotting against temperature
ax4.plot(weather['dt'], weather['temp_c'],
  color = 'maroon',
  linestyle = '--', 
  linewidth = 1,
  alpha = 0.5)
ax4.set(
        ylim  = [0, 40],
      )
ax4.set_ylabel('Median\nTemp.(°C)', color = 'maroon')
# changing axis color for plot readibility
ax4.spines['left'].set_color('maroon')
ax4.tick_params(axis = 'y', colors = 'maroon')
[t.set_color('maroon') for t in ax4.yaxis.get_ticklines()]
[t.set_color('maroon') for t in ax4.yaxis.get_ticklabels()]

# plotting against OA conc:
ax5 = ax4.twinx()
ax5.plot(oa_data['datetime_local'], oa_data['OA_ug_m3'],
         color = 'black',
         linestyle = '--', 
         linewidth = 1,
         alpha = 0.5)
ax5.set_ylim([0, 35])
ax5.set_yticks(
    ticks = np.linspace(0,35, 3),
    labels = np.linspace(0, 35, 3)
    )
ax5.set_ylabel('OA Mass\n(μg $m^-3$)', color = 'darkslategray')


axis1 = ['Factor 1', 'Factor 3']
axis2 = ['Factor 2', 'Factor 4']

  
# plotting factor solns
for col in df.columns:
    if col in axis1:
        plt1 = ts_plot2(df, ax1, color = color_dict[col], linestyle = linestyle_dict[col])
    elif col in axis2:
       plt2 = ts_plot2(df, ax2, color = color_dict[col], linestyle = linestyle_dict[col])
    

ax1.set(
       xlim = (df.index[0] - timedelta(hours = 6), df.index[-1] + timedelta(hours = 6)), # adjusting x-lim
       xlabel = 'Local Date/Time',  
       #title = f'{len(df.columns)} Factor Solution' 
       )


for ax, t in zip([ax4, ax1, ax2], ['A', 'B', 'C']):
    ax.text(
        x = -0.12,
        y = 1., 
        s = t,
        transform = ax.transAxes,
        weight = 'bold',
        fontsize = 12
        )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
for ax in [ax1, ax2]:
    ax.set_ylabel(
        'Normalized\nConc.'
        )
    ax.legend(
        ncols = 1,
        loc = 'upper right',
        fontsize = 'small'
        )

fig.autofmt_xdate()

ax1.xaxis.set_major_formatter(md.DateFormatter('%D'))


# adding heatwave and smoke event boxes box
heatwave_start = datetime.datetime(2023, 7, 27) # dt object type # edit !
heatwave_end = datetime.datetime(2023, 7, 29) # edit ! 
smoke1_start = datetime.datetime(2023, 7, 19) # edit !
smoke1_end = datetime.datetime(2023, 7, 21) # edit !
smoke2_start = datetime.datetime(2023, 8, 5) # edit !
smoke2_end = datetime.datetime(2023, 8, 7) # edit !
 
start_y_point = -0.5 
height = 50 
box_color = 'maroon' # edit ! 
set_alpha = 0.2 # edit ! 
for ax in [ax1, ax2, ax4]:
    rect_box(heatwave_start, heatwave_end, ax, '') # adding heatwave boxes
    rect_box(heatwave_start, heatwave_end, ax, '') # adding heatwave box

box_color = 'gray'
for ax in [ax1, ax2, ax4]:
    rect_box(smoke1_start, smoke1_end, ax, '') # adding smoke boxes
    rect_box(smoke2_start, smoke2_end, ax, '') 
    rect_box(smoke1_start, smoke1_end, ax, '')
    rect_box(smoke2_start, smoke2_end, ax, '') 


plt.show()
plt.close()

# %% diel style plot by time groupings

df = factor_ts.copy() # no overwriting! 
df = df[['datetime', 'Factor 1', 'Factor 3', 'Factor 2', 'Factor 4', 'hour']]


df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x >= 6
                                   else 
                                   'am2' if x <= 12 and x >= 10
                                   else
                                   'pm1' if x <= 16 and x > 13
                                   else 
                                   'pm2'
                                   )

ignore = ['hour', 'timeblock']
frag_factors = ['Factor 1', 'Factor 3']
ind_factors = ['Factor 2', 'Factor 4']

gridspec = {'hspace' : 0.2, 
            'wspace' : 0.2,
            'height_ratios' : [0.25, 0.25, 1],
            }
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows = 3,
    ncols = 1,
    sharex = True,
    figsize = (8, 6),
    dpi = 300,
    gridspec_kw = gridspec
    )

# ax1 = top line plot, F1/F3, 
df = factor_ts.copy() # no overwriting! 
df = df[['datetime', 'Factor 1', 'Factor 3', 'Factor 2', 'Factor 4', 'hour']]
df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x >= 6
                                   else 
                                   'am2' if x <= 12 and x >= 10
                                   else
                                   'pm1' if x <= 16 and x > 13
                                   else 
                                   'pm2'
                                   )
for col in df.columns:
    if col not in ignore:
        if col in frag_factors:
            grouped = df.groupby('timeblock')[col].agg(
                median = 'median',
                q25 = lambda x: x.quantile(0.25),
                q75 = lambda x: x.quantile(0.75))
            ax1.plot(grouped.index, grouped['median'], label = col,
                    color = color_dict[col],
                     linestyle = 'solid', 
                     linewidth = 0.75,
                     marker = 'o',
                     markersize = 7)
            ax1.fill_between(grouped.index, grouped['q25'], grouped['q75'],
                            alpha = 0.1, color = color_dict[col])
        if col in ind_factors:
            grouped = df.groupby('timeblock')[col].agg(
                median = 'median',
                q25 = lambda x: x.quantile(0.25),
                q75 = lambda x: x.quantile(0.75))
            ax2.plot(grouped.index, grouped['median'], label = col,
                    color = color_dict[col],
                     linestyle = 'solid', 
                     linewidth = 0.75,
                     marker = 'o',
                     markersize = 7)
            ax2.fill_between(grouped.index, grouped['q25'], grouped['q75'],
                            alpha = 0.1, color = color_dict[col])
ax1.set_ylim([0, 2])
ax1.set_yticks(
    ticks = np.linspace(0, 2, 3),
    labels = np.linspace(0, 2, 3),
    fontsize = 12
    )
ax2.set_ylim([0, 3])
ax2.set_yticks(
    ticks = np.linspace(0, 3, 3),
    labels = np.linspace(0, 3, 3),
    fontsize = 12
    )
ax1.text(
    x = -0.1,
    y = -1.05,
    s = 'Factor Conc.',
    fontsize = 14,
    rotation = 90,
    transform = ax1.transAxes
    )
ax1.text(
    x = -0.1,
    y = 1.1, 
    s = 'A',
    fontsize = 14,
    fontweight = 'bold',
    transform = ax1.transAxes
    )
#ax2.get_legend().remove()


# ax3 = bottom bar plot
df = factor_ts.copy() # no overwriting! 
df = df[['Factor 1', 'Factor 3', 'Factor 2', 'Factor 4', 'hour']]
df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x >= 6
                                   else 
                                   'am2' if x <= 12 and x >= 10
                                   else
                                   'pm1' if x <= 16 and x > 13
                                   else 
                                   'pm2'
                                   )
temp_list = []
for col in df.columns:
    if col not in ignore:
        grouped = df.groupby('timeblock')[col].median()
        temp_list.append(grouped)
temp_df = pd.DataFrame(data = temp_list)
temp_df = temp_df.transpose()

temp_df.plot(ax = ax3, 
             stacked = True,
             kind = 'bar', 
             subplots = False, 
             sharex = True, 
             use_index = True, 
             width = 0.8,
             fontsize = 14,
             color = [color_dict[col] for col in temp_df.columns],
             edgecolor = 'white', 
             linewidth = 2,
             rot = 360
             )
ax3.set_yticks(
    ticks = np.linspace(0, 5, 5),
    labels = np.linspace(0, 5, 5),
    fontsize = 12
    )
ax3.set_xticks(
    ticks = range(len(temp_df.index)),
    labels = ['Morning', 'Mid-Morning', 'Afternoon', 'Early Evening'],
    fontsize = 14
    )
ax3.set_ylabel('∑Factor Conc.',
               fontsize = 14,
               #labelpad = 9
               )
ax3.set_xlabel('')
ax3.legend(ncols = 4,
          loc = 'upper center',
          bbox_to_anchor = (0.5, 1.9),
          fontsize = 12
          )
ax3.text(
    x = -0.1,
    y = 1, 
    s = 'B',
    fontsize = 14,
    fontweight = 'bold',
    transform = ax3.transAxes
    )
  
for ax in [ax1, ax2, ax3]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    

plt.show()

# %% SI Figure: Bar Charts of OH Reactivities by Factor

df_percent = profile_conc.copy()
df_percent.set_index('compounds', inplace = True)
cmpd_list = df_percent.index

factor_ref_data = cmpd_refs.copy()
factor_ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
factor_ref_data.set_index('compounds', inplace = True)
factor_ref_data = factor_ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

for index, row in df_percent.iterrows(): # applying number density function 
    for col in row.index:
     if pd.api.types.is_numeric_dtype(df_percent[col]):
         df_percent.at[index, col] = number_density(
             row[col],
         )


# converting to reactivities against OH (1/s)
for col in df_percent:
    df_percent[col] = df_percent[col] * factor_ref_data['k_oh_298']

# getting total OH
factor_oh_react = df_percent.copy()
oh_total = factor_oh_react.sum(axis = 0, numeric_only = True)

units = '(sec$^-$$^1$)'
for col in df_percent:
    # getting factor reactivity percentages
    df_percent[col] = (df_percent[col] / df_percent[col].sum()) * 100 

df_percent = df_percent.transpose()  # need cmpds as column headers and factors as index
variable_count = len(df_percent.index) # of factors


# Making the plot

fontsize = 18
fontweight = 'normal'

fig, ax = plt.subplots(figsize = ((20, 20)), 
                       dpi = 300)
bars = df_percent.plot(ax = ax, 
               stacked = True,
                 kind = 'bar', 
                 subplots = False, 
                 sharex = True, 
                 width = 0.95,
                 ylim = [0, 100],
          color = [cmpd_colors_dict[col] for col in df_percent.columns],
          edgecolor = 'white', 
          linewidth = 0.75,
          rot = 360,
          fontsize = fontsize
          )

for bar_idx, container in enumerate(ax.containers):
    compound = df_percent.columns[bar_idx]
    hatch_style = hatch_dict.get(compound, '')

    for bar in container:
        bar.set_hatch(hatch_style)

ax.set_ylabel(
    'OH Reactivity Distribution (%)',
    fontsize = fontsize,
    fontweight = fontweight
    )
# setting stuff to bold
ax.set_yticklabels(ax.get_yticks(), weight = fontweight) # setting y-ticks to bold
for label in ax.get_xticklabels(): # setting axis labels to bold
    label.set_fontweight(fontweight)

# labeling the top 5 compounds making up each factor:
for bar_idx, (index, row) in enumerate(df_percent.iterrows()):
    top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(5).index.tolist()

    cum_height = 0
    for col in df_percent.columns:
        height = row[col]
        mid_point = cum_height + height / 2
        if col in top_cols and height > 0:
                # have to wrap the really long compound labels so they fit in the bar
                wrapped_label = '\n'.join(textwrap.wrap(col, width = 25))  # Adjust width as needed
                
                ax.text(
                    x = bar_idx,
                    y = mid_point,
                    s = col + f'\n{height:.1f}%',
                    ha = 'center',
                    va = 'center',
                    fontsize = 10 if height < 3 else 15,
                    color = 'black'  # or a color that contrasts the bar
                    )
        cum_height += height

plt.legend(ncols = 1, fontsize = 16, loc = 'best',
           bbox_to_anchor = (1, 1))

fontsize = 20
add_percent_bar_totals(ax, range(len(df_percent)), 99, oh_total, units, fontsize, fontweight)

labels = ['Industrial', 'Fragrances', 'Biogenic', 'Other']
legend_colors = ['#B0B0B0', 'hotpink', 'Green', '#DC8E43']
# #F33A6A
hatches = ['', '', '\\', '/']

# adding manual legends
patches = []
for i, color, h in zip(labels, legend_colors, hatches):
    leg = mpatches.Patch(facecolor = color, 
                         edgecolor = 'white',
                         label = i,
                         hatch = h)
    patches.append(leg)
ax.legend(handles = patches, 
          loc = 'lower center', 
          bbox_to_anchor = (0.5, -0.1),
          fontsize = 24,
          ncols = 4, 
         #columnspacing = 15
          )

plt.show()
plt.close()

        
# %% SI Figure: pollution roses by TOD

ignore = ['datetime', 'hour', 'timeblock']

# Parameters
num_bins = 36
bin_edges = np.linspace(0, 360, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# ---- Build combined dataframe ----
df = factor_ts.copy()


# Add wind data
df['avg_WindSpeed_m/s'] = meta_data['avg_WindSpeed_m/s'].values
df['median_WindDir'] = meta_data['median_WindDir'].values

# Timeblock classification
df['timeblock'] = df['hour'].apply(
    lambda x:
        'am1' if 6 <= x <= 9 else
        'am2' if 10 <= x <= 12 else
        'pm1' if 13 <= x <= 16 else
        'pm2'
)

# Columns to plot
cols_to_plot = ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4']
labels = ['A. Morning', 'B. Mid-Morning', 'C. Afternoon', 'D. Early Evening']

factor_norms = {}

for col in cols_to_plot:
    factor_norms[col] = {
        'vmin': df[col].min(),
        'vmax': df[col].max()
    }

# ---- Loop through timeblocks ----
for tb, label in zip(['am1', 'am2', 'pm1', 'pm2'], labels):

    # Filter dataframe
    subdf = df[df['timeblock'] == tb]

    wind_dir = subdf['median_WindDir'].values
    wind_spd = subdf['avg_WindSpeed_m/s'].values

    # Create figure
    fig, axs = plt.subplots(
        4, 1,
        subplot_kw={'projection': 'polar'},
        figsize=(10, 16),
        dpi=300
    )

    axs = axs.flatten()

    plt.subplots_adjust(hspace=0.55)
    
    # Global color scaling for each factor
    

    # ---- Plot each factor ----
    for idx, col in enumerate(cols_to_plot):

        ax = axs[idx]
        voc_conc = subdf[col].values

        # Bin wind directions
        bin_indices = np.digitize(wind_dir, bin_edges) - 1

        mean_spd = np.zeros(num_bins)
        mean_voc = np.zeros(num_bins)

        for i in range(num_bins):

            mask = bin_indices == i

            if np.any(mask):
                mean_spd[i] = np.median(wind_spd[mask])
                mean_voc[i] = np.median(voc_conc[mask])
            else:
                mean_spd[i] = 0
                mean_voc[i] = np.nan

        theta = np.deg2rad(bin_centers)

        vmin = factor_norms[col]['vmin']
        vmax = factor_norms[col]['vmax']
        
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        
        bars = ax.bar(
            theta,
            mean_spd,
            width=np.deg2rad(360 / num_bins),
            bottom=0.0,
            color=plt.cm.Reds(norm(mean_voc)),
            edgecolor='black',
            linewidth=0.02
        )   

        sm = plt.cm.ScalarMappable(
             cmap='Reds',
             norm=norm
         )
         
        sm.set_array([])
         
        cbar = plt.colorbar(
             sm,
             ax=ax,
             pad=0.05,
             orientation='vertical',
             shrink=0.9,
             format=lambda x, _: f"{x:.1f}"
         )

        # Polar formatting
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        ax.set_ylim(0, 1.75)
        ax.set_rticks([0.25, 0.75, 1.25, 1.75])
        ax.set_rlabel_position(135)

        # ax.set_title(
        #     f'{col}',
        #     fontsize=14,
        #     pad=30
        # )
        
        # Subplot label
        ax.text(
            x = -0.4, 
            y = 0.35,
            s = f'{col}',   
            transform = ax.transAxes,
            fontsize = 18,
            fontweight = 'normal',
            rotation = 90,
        )

        ax.grid(
            visible=True,
            which='major',
            axis='both',
            color='gray',
            alpha=0.3
        )
        

    # Figure title
    fig.suptitle(
        f'{label}',
        fontsize=18,
        y=0.93,
        x = 0.6
    )

    plt.show()

# %% Cmpd vs Cmpd OLS Model (w/ and w/out outliers)

import statsmodels.api as sm # re-loading bc dumb new error

# days of high Factor 1 and Factor 4: 
    # Factor 4: July 17, ~6 and 10am (indicies 5, 6) and August 2 (index 20)
    # Factor 1: July 11 - July 16 (indicies 0 - 4)
    
both_days_index = [0, 1, 2, 3, 4, 5, 6, 20]
only_f4_index = [5, 6, 20]

# running function definied previously: requires the following variables to be definied:

# defining general variables 

data_df = pmf_input
loc = 'lower right'

legend_fontsize = 20
label_fontsize = 24
tick_fontsize = 20
fontweight = 'normal'
markersize = 14

gridspec = {'wspace' : 0.3}
fig, (ax2, ax4) = plt.subplots(
    ncols = 2,
    nrows = 1,
    sharey = False,
    sharex = False,
    figsize = (16, 7), 
    dpi = 300,
    gridspec_kw = gridspec
    )


# # defining plot-specific variables
# x_variable_name = 'benzene' # edit !
# y_variable_name = 'toluene' # edit !
# color = 'dodgerblue'
# outlier_index = only_f4_index
# # applying function and creating the plot
# plt1 = ols_corr_plot_outliers(ax1, x_variable_name, y_variable_name, data_df, outlier_index, color, label_fontsize,
#               tick_fontsize,
#               fontweight, markersize, legend_fontsize, loc) 
# ax1.set_title(
#     'High F4 days excluded'
#     )

# # defining plot-specific variables
x_variable_name = 'a-pinene' # edit !
y_variable_name = 'limonene' # edit !
color = 'darkgreen'
outlier_index = only_f4_index
plt2 = ols_corr_plot_outliers(ax2, x_variable_name, y_variable_name, data_df, outlier_index, color, label_fontsize,
              tick_fontsize,
              fontweight, markersize, legend_fontsize, loc) 

ax2.set_ylabel('limonene ($ppb_v$)',
              fontweight = fontweight,
              fontsize = label_fontsize,
              labelpad = 9)
# ax2.set_title(
#     'High F4 days excluded'
#     )


# x_variable_name = 'benzene' # edit !
# y_variable_name = 'toluene' # edit !
# color = 'dodgerblue' # edit ! 
# outlier_index = both_days_index
# plt3 = ols_corr_plot_outliers(ax3, x_variable_name, y_variable_name, data_df, outlier_index, color, label_fontsize,
#               tick_fontsize,
#               fontweight, markersize, legend_fontsize, loc) 
# ax3.set_title(
#     'High F1 and F4 days exclded'
#     )

x_variable_name = 'camphor' # edit !
y_variable_name = 'limonene' # edit !
color = 'darkorange' 
outlier_index = only_f4_index
plt4 = ols_corr_plot_outliers(ax4, x_variable_name, y_variable_name, data_df, outlier_index, color, label_fontsize,
              tick_fontsize,
              fontweight, markersize, legend_fontsize, loc) 

# ax4.set_title(
#     'High F1 and F4 days exclded'
#     )

# ax2.text(
#     x = 0.85, 
#     y = -0.15, 
#     s = 'α-pinene ($ppb_v$)',
#     fontsize = label_fontsize,
#     fontweight = fontweight,
#     transform = ax2.transAxes
#     )


plt.show()
    

# %% SI Figure 2: Bar charts of original ground data by TOD # updated w/ full dataset

#df = common_cmpds_ppb.copy()
df = full_cmpd_dataset.copy()

df.rename(columns = {'start_time_edt' : 'datetime'}, inplace = True)

df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x >= 6
                                   else 
                                   'am2' if x <= 12 and x >= 10
                                   else
                                   'pm1' if x <= 16 and x > 13
                                   else 
                                   'pm2'
                                   )

start_time = df['start_time']
df.drop(columns = ['start_time', 'hour'], inplace = True)
     
am1 = df[df['timeblock'] == 'am1'] # filtering by timeblock
am2 = df[df['timeblock'] == 'am2']
pm1 = df[df['timeblock'] == 'pm1']
pm2 = df[df['timeblock'] == 'pm2']


block_dfs = [am1, am2, pm1, pm2]
block_strings = ['am1', 'am2', 'pm1', 'pm2']

block_dfs = [df.median(numeric_only = True) for df in block_dfs]  # getting median values
block_dfs = [pd.DataFrame(df) for df in block_dfs]
for df, value in zip(block_dfs, block_strings):
    df.rename(columns = {0 : value}, inplace = True)

concat = pd.concat(block_dfs, axis = 1)
df = concat.transpose()

variable_count = len(df.index) 
compounds = df.columns

units = 'ppbv'
df_total = concat.sum(axis = 0, numeric_only = True)

xlabels = [
    'Morning',
    'Mid-Morning',
    'Afternoon',
    'Early Evening'
    ]

fig, ax = plt.subplots(figsize = (13, 5), dpi = 300)
bars = df.plot(ax = ax, 
               stacked = True,
                 kind = 'bar', 
                 subplots = False, 
                 sharex = True, 
                 use_index = True, 
                 width = 0.95,
          ylim = [0, 4],
          fontsize = 14,
          color = [cmpd_colors_dict[col] for col in df.columns],
          edgecolor = 'white', 
          linewidth = 0.75,
          rot = 360
          )

for col_idx, container in enumerate(ax.containers):
    compound = df.columns[col_idx]
    hatch_style = hatch_dict.get(compound, '')

    for bar in container:
        bar.set_hatch(hatch_style)

ax.set_ylabel(
    'Median Mixing Ratios ($ppb_v$)',
    fontsize = 16,
    fontweight = 'normal',
    labelpad = 6
    )
ax.set_xticks(
    ticks = range(len(df)),
    labels = xlabels,
    fontsize = 16,
    fontweight = 'normal'
    )

ax.set_yticks(
    ticks = np.linspace(0, 4.5, 5),
    labels = np.linspace(0, 4.5, 5),
    fontweight = 'normal',
    fontsize = 16)

units = '$ppb_v$'
add_bar_totals(range(len(df)), 4.6, df_total, units, 16, 'normal')

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.get_legend().remove()

labels = ['Industrial', 'Fragrances', 'Biogenic', 'Other']
legend_colors = ['#B0B0B0', 'hotpink', 'Green', '#DC8E43']
# #F33A6A
hatches = ['', '', '\\', '/']

# adding manual legends
patches = []
for i, color, h in zip(labels, legend_colors, hatches):
    leg = mpatches.Patch(facecolor = color, 
                         edgecolor = 'white',
                         label = i,
                         hatch = h)
    patches.append(leg)
ax.legend(handles = patches, 
          loc = 'upper right', 
      #    bbox_to_anchor = (0.5, -0.1),
          fontsize = 16,
          ncols = 2, 
         #columnspacing = 15
          )

plt.show()
plt.close()


# %% TS of specific compounds

df = full_cmpd_dataset.copy()
df.rename(columns = {'start_time' : 'datetime'}, inplace = True)
df.set_index('datetime', inplace = True)

tracer1 = ['toluene']
tracer2 = ['benzyl acetate']
tracer3 = ['phenoxyisopropanol']
tracer4 = ['limonene']
tracer5 = ['decamethylcyclopentasiloxane']
       

mc = 12
df = df

fig, (ax1, 
      ax2, 
      ax3, 
      ax4, 
      ax5
      ) = plt.subplots(
    nrows = 5,
    ncols = 1,
    sharex = True,
    figsize = (8, 8),
    dpi = 300)
          
tracer_axes = { # needs to be created AFTER the figure axes so they aren't overwritten
    ax1: tracer1,
    ax2: tracer2,
    ax3: tracer3,
    ax4: tracer4,
    ax5: tracer5
}  

# plotting 

for ax, tracers in tracer_axes.items():
    for tracer in tracers:
        if tracer in df.columns:
            col = tracer
            general_ts_plot(chosen_df = df, 
                            axis = ax, 
                            color = cmpd_colors_dict[col], 
                            linestyle = 'solid', 
                            mc = mc)
            

ax3.text( # making a y-axis label for all subplots
    x = -0.15,
    y = -0.75, 
    s = 'Mixing Ratio ($ppb_v$)',
    rotation = 'vertical',
    fontsize = 14,
    fontweight = 'bold',
    transform = ax3.transAxes, # relative to axis, not data
    )

for axis in [ax1, ax2, ax3, ax4, ax5]:
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.yaxis.set_minor_locator(AutoMinorLocator()) # turning on minor axis ticks

ax5.set_xticks(pd.date_range(start = df.index[0], end = df.index[-1], periods = 10),
               labels = pd.date_range(start = df.index[0], end = df.index[-1], periods = 10),
               fontsize = 12
                ) 


fig.autofmt_xdate()
ax5.xaxis.set_major_formatter(md.DateFormatter('%D'))



plt.show()
plt.close()

# %% OFP for Factors

'''
Only using compounds with published MIR values 
'''

df = profile_conc.copy() 
cmpd_list = df['compounds']

ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

df.set_index('compounds', inplace = True)
# converting to OFP (ppb)
for col in df.columns:
    df[col] = df[col] * ref_data['MIR_ppb']
    
df_factor_totals = df.sum()
f1 = df['Factor 1']
f2 = df['Factor 2']
f3 = df['Factor 3']
f4 = df['Factor 4']


factor_dfs = [f1, f2, f3, f4] 
factor_dfs = [pd.DataFrame(x) for x in factor_dfs] # making everything a df
for df in factor_dfs:
    for col in df: # col = factor, 1 factor per df
        df.sort_values(by = col, ascending = False, inplace = True) 
top_number = 5
factor_dfs_top4 = [df[:top_number].copy() for df in factor_dfs] # top 4 cmpds per factor
factor_dfs_other = [df[top_number:].sum() for df in factor_dfs] # summing rest into "other" category per factor
factor_dfs_other = [pd.DataFrame(df) for df in factor_dfs_other]
factor_dfs_other = [df.transpose() for df in factor_dfs_other]
for df in factor_dfs_other:
    df.rename(index = {0 : 'other'}, inplace = True)

factor_dfs = []
for x in range(0, 4):
    f = pd.concat([factor_dfs_top4[x], factor_dfs_other[x]])
    factor_dfs.append(f) # putting everything back together
    
ofp_factor_totals = [df.copy() for df in factor_dfs]
ofp_factor_totals = [df.sum() for df in ofp_factor_totals]
ofp_factor_totals = pd.concat(ofp_factor_totals)
ofp_factor_totals = pd.DataFrame(data = ofp_factor_totals)
ofp_factor_totals.reset_index(inplace = True)
ofp_factor_totals.rename(columns = {0 : 'ofp_totals', 'index' : 'Factor'}, inplace = True)


# function to add totals to the top of each bar in the percentage bar plot
def add_percent_bar_totals(axis, x, y, value, total): 
    for i in range(len(x)):
        axis.text(x = i, y = y[i] + 0.05,  # setting x and y positions of labels - centering with bar and then slightly above top of bar
                 s = f'{value[i] / total * 100:0.1f}%', # setting and formatting what appears in the label
                 ha = 'center'
                 )   

def bar_graph_subplot(df, axis):
    values = df.iloc[:, 0].values
    total = values.sum()
    labels = df.index
    color = [cmpd_colors_dict[value] for value in df.index]
    axis.bar(
        x = range(len(labels)),
        height = values,
        #tick_label = labels,
        color = color,
        edgecolor = 'black',
        )
    axis.set_xticks(ticks = range(len(labels)), labels = labels, 
                    fontsize = 9,
                    rotation = 0
                    )
    axis.set_yticks(ticks = np.linspace(0, 2.5, 5),
                    labels = np.linspace(0, 2.5, 5),
                    fontsize = 12)
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.set_title(f'Factor {i + 1}',
                   x = 0.85, 
                   y = 0.88,
                   fontweight = 'bold',
                   transform = axis.transAxes
                   )
    axis.set_ylim(0, 2.5)
    add_percent_bar_totals(axis, df.index, values, values, total)
    

fig, axes = plt.subplots(
     nrows = len(factor_dfs),
     ncols = 1,
     sharey = True,
     dpi = 300,
     figsize = (10, 20)
     )

i = 0 # starting the counter at 0
      
for j, df in enumerate(factor_dfs):
    ax = axes[i]
    bar_graph_subplot(df = df, axis = ax)
    ax.text(
        x = 0.805,
        y = 0.71,
        s = f'Total:\n{ofp_factor_totals.iloc[i, 1]:0.1f} $ppb_v$',
        fontsize = 12,
        transform = ax.transAxes
        )
    ax.text(
        x = -0.1, 
        y = 1,
        s = chr(65 + i),
        fontsize = 14,
        transform = ax.transAxes,
        fontweight = 'bold'
        )
    i += 1  # move to next axis
    
axes[3].text( # making a y-axis label for all subplots
    x = -0.16,
    y = 2.1, 
    s = 'OFP ($ppb_v$)',
    rotation = 'vertical',
    fontsize = 14,
    transform = axes[3].transAxes
    )

plt.show()

   
# %% Van Kreeveln plots vs VP (all compounds)

df = full_cmpd_dataset.copy()
df.drop(columns = ['start_time', 'hour'], inplace = True)
cmpds = df.columns

ref_copy = cmpd_refs[['compound_name', 'formula', 'k_oh_298', 'sat_vapor_pressure_mmHg_25C', 'mw_g_mol']].copy()
ref_copy.set_index('compound_name', inplace = True)
ref_copy = ref_copy.loc[cmpds] # filtering for cmpds  # need exact match for loc / list
ref_copy.set_index('formula', inplace = True)
formulas = ref_copy.index

vankrev_df = calc_ratios(formulas) # apply the van krev function
vankrev_df.set_index('compound', inplace = True)

df = pd.concat([vankrev_df, ref_copy], axis = 1)
df['mw_g_mol'] = pd.to_numeric(df['mw_g_mol'], errors = 'coerce')
df['sat_vapor_pressure_mmHg_25C'] = pd.to_numeric(df['sat_vapor_pressure_mmHg_25C'], errors = 'coerce')

fig, axes = plt.subplots(
    nrows = 2,
    ncols = 1,
    sharex = True,
    figsize = (10, 6), 
    dpi = 300)

axes[0].axvline(x = 0.1, # adding vertical line at 0.1 mmHg
                color = 'gray',
                linestyle = 'dashed',
                alpha = 0.5
                ) 
axes[1].axvline(x = 0.1, # adding vertical line at 0.1 mmHg
                color = 'gray',
                linestyle = 'dashed',
                alpha = 0.5
                ) 

cmap = plt.get_cmap('bone_r') # choose the colormap
color_norm = plt.Normalize(df['k_oh_298'].min(), df['k_oh_298'].max())
size_norm = plt.Normalize(df['mw_g_mol'].min(), df['mw_g_mol'].max())


axes[0].scatter(df['sat_vapor_pressure_mmHg_25C'], df['O:C'], 
         marker = 'o',
         s =  size_norm(df['mw_g_mol']) * 450,
         color = cmap(color_norm(df['k_oh_298'])) ,
         #color = cmap(color_norm(df['mw_g_mol'])),
         edgecolor = 'black'
         )
axes[0].set_xscale('log')
axes[0].set_ylabel('O:C')
axes[0].set_xlim([df['sat_vapor_pressure_mmHg_25C'].min() - 0.000005, df['sat_vapor_pressure_mmHg_25C'].max() + 100])
axes[0].set_ylim([-0.3, 2])


axes[1].scatter(df['sat_vapor_pressure_mmHg_25C'], df['H:C'],
           marker = 'o',
           #color  = cmap(color_norm(df['mw_g_mol'])
           color = cmap(color_norm(df['k_oh_298'])),
           s =  size_norm(df['mw_g_mol']) * 450,
           edgecolor = 'black'
                        )
axes[1].set_ylabel('H:C')
axes[1].set_ylim([-0.3, 5])
axes[1].set_xlabel('Saturated Vapor Pressure (mmHg)')



cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), 
                    ax = axes.ravel().tolist(), pad = 0.1)
cbar.set_label(label = '$k_{OH}$ (cm$^3$ molecules$^{-1}$ sec$^{-1}$)', 
                   fontsize = 11) # set label of the colorbar on the top plot
# cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), ax = ax2, pad = 0.1)
# cbar.set_label(label = 'OH Reactivity ($sec{^-1}$', 
#                    fontsize = 8) # set label of the colorbar on the top plot

axes[0].set_title('All Measured sVOCs (sized by MW)')



plt.show()

# %% Figure 1: Van Krevveln plots vs C* (all compounds)

#cstar equation from SI, Donahue et al (2006); https://doi.org/10.1021/es052297c


df = full_cmpd_dataset.copy()
df.drop(columns = ['start_time', 'hour'], inplace = True)
cmpds = df.columns

ref_copy = cmpd_refs[['compound_name', 'formula', 'k_oh_298', 'sat_vapor_pressure_mmHg_25C', 'mw_g_mol']].copy()
ref_copy.set_index('compound_name', inplace = True)
ref_copy = ref_copy.loc[cmpds] # filtering for cmpds  # need exact match for loc / list
ref_copy.reset_index(inplace = True, drop = False)
ref_copy.set_index('formula', inplace = True, drop = False)
ref_copy.rename(columns = {'index' : 'compound_name'}, inplace = True)
formulas = ref_copy.index

vankrev_df = calc_ratios(formulas) # apply the van krev function
vankrev_df.set_index('compound', inplace = True)

df = pd.concat([vankrev_df, ref_copy], axis = 1)
df['mw_g_mol'] = pd.to_numeric(df['mw_g_mol'], errors = 'coerce')
df['sat_vapor_pressure_mmHg_25C'] = pd.to_numeric(df['sat_vapor_pressure_mmHg_25C'], errors = 'coerce')

# "assuming act_coe is one for all cases" 
act_coe = 1
tmpK = 298.15
R = 8.206E-5 #units = m3atm/molK

# C*(ug/m3) = (MW (g/mol) * 1E6(ug/g) * VP (mmHg)) * act_coe(assume 1) /  (T(K) * R (m3atm/Kmol) * 760 (mmHg/atm))
df['cstar_ug_m3'] = (df['mw_g_mol'] * 1E6 * act_coe * df['sat_vapor_pressure_mmHg_25C']) / (tmpK * R * 760) # converting from atm to mmHg) 

fig, axes = plt.subplots(
    nrows = 2,
    ncols = 1,
    sharex = True,
    figsize = (12, 6), 
    dpi = 300)
plt.subplots_adjust(hspace = 0.07) # making the subplots closer together

cmap = plt.get_cmap('bone_r') # choose the colormap
color_norm = plt.Normalize(df['k_oh_298'].min(), df['k_oh_298'].max())
size_norm = plt.Normalize(df['mw_g_mol'].min(), df['mw_g_mol'].max())

# volatility ranges: doi:10.5194/acp-12-615-2012
# adding volatility  boxes
start_y_point = -5
height = 100
set_alpha = 0.3 # edit ! 

voc_ranges = (
    [df['cstar_ug_m3'].min() - 30, 300], # actual min is 0.3, but need labels to match x-axis limits
    [300, 3E6],
    [3E6, df['cstar_ug_m3'].max() + 2E9] # actual max is 3E15 
    )

box_colors = ['#867f78', # old = #7A8586 , #svoc
              '#bdbcab', # old = #9EBABD ,  #ivoc
              '#faf9e2' # old = #DBE6E7, #voc
              ]
 
label_list = [
    'SVOC', 
    'IVOC',
    'VOC'
    ]

fontweight = 'normal'
hatch = ''

# adding  boxes to both axes.
for axis in [0, 1]:
    for i, color in zip(voc_ranges, box_colors):
        min = i[0]
        max = i[1]
        points_rect_box(min, max, axis, color = color, hatch = hatch)

# adding VOC range labels:
for (i0, i1), j in zip(voc_ranges, label_list):
    axes[0].text(
         x = np.sqrt(i0 * i1), # placing in the middle of the log scale VOC_range box
         y = 0.93,
         s = f'{j}', # linking to the value in the label
         transform = axes[0].get_xaxis_transform(), # turning into data scale
         ha = 'center',
         va = 'top',
         fontsize = 12
        )

# adding manual legends
# patches = []
# for i, color in zip(label_list, box_colors):
#     leg = mpatches.Patch(color = color, label = i)
#     patches.append(leg)
# axes[0].legend(handles = patches, loc = 'upper left', 
#                ncols = 3, 
#                #columnspacing = 15
#                )

# making the plots
sc = axes[0].scatter(df['cstar_ug_m3'], df['O:C'], 
         marker = 'o',
         s =  size_norm(df['mw_g_mol']) * 450,
         color = cmap(color_norm(df['k_oh_298'])) ,
         #color = cmap(color_norm(df['mw_g_mol'])),
         edgecolor = 'black'
         )
axes[0].set_xscale('log')
axes[0].set_ylabel('O:C', fontweight = fontweight)


axes[1].scatter(df['cstar_ug_m3'], df['H:C'],
           marker = 'o',
           #color  = cmap(color_norm(df['mw_g_mol'])
           color = cmap(color_norm(df['k_oh_298'])),
           s =  size_norm(df['mw_g_mol']) * 450,
           edgecolor = 'black'
                        )
axes[1].set_ylabel('H:C', fontweight = fontweight)
axes[1].set_xlabel('C* (μg/$m^3$)', fontweight = fontweight)


cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), ax = axes.ravel().tolist(), pad = 0.03)
cbar.set_label(label = '$k_{OH}$ (cm$^3$ molecules$^{-1}$ sec$^{-1}$)', 
                   fontsize = 11) # set label of the colorbar on the top plot

# setting stuff to bold
for x in [0, 1]:
    axes[x].set_yticklabels(axes[x].get_yticks(), weight = fontweight) # setting y-ticks to bold
    for label in axes[x].get_xticklabels(): # setting axis labels to bold
        label.set_fontweight(fontweight)

axes[0].set_ylim([-0.08, 1.15])
axes[0].set_yticks(ticks = np.linspace(0, 1.15, 6),
                   labels = np.linspace(0, 1.15, 6))
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[1].set_ylim([-0.2, 4.5])
axes[1].set_yticks(ticks = np.linspace(0, 4.5, 6),
                   labels = np.linspace(0, 4.5, 6))
axes[1].set_xlim([df['cstar_ug_m3'].min() - 30, df['cstar_ug_m3'].max() + 2E9])
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#axes[0].set_title('All Quantified Compounds (sized by MW)')

# setting a MW legend
sizes = np.sort(np.unique(sc.get_sizes()))
sizes = sizes[sizes > 0]
mw = size_norm.inverse(sizes[[0, len(sizes)//2, -1]] / 450).tolist()
handles, labels = sc.legend_elements(prop='sizes', 
                                     num=mw,
                                     func=lambda 
                                     s: size_norm.inverse(s / 450), 
                                     fmt='{x:.0f}', 
                                     markerfacecolor='none')
axes[0].legend(handles, labels, title='MW (g/mol)',
               labelspacing = 1.2)

# adding specific compounds to bottom plot
names_to_label = {'toluene', 'octadecane', 
                 'homosalate', 
                  }  
for _, row in df[df['compound_name'].isin(names_to_label)].iterrows():
    axes[1].annotate(
        row['compound_name'],
        (row['cstar_ug_m3'], row['H:C']),
        xytext=(12, 10),
        textcoords='offset points',
        arrowprops = {'arrowstyle' : '->',
                      'color' : 'black',
                      'linewidth' : 1.5,
                      'relpos': (0.0, 0.15), # arrow attachment point
                      'shrinkA': 0
                      },
        rotation = 10
    )
    


plt.show()

# %% SI Figure: Timeseries of summed OHR and OFP # updated w/ full dataset
'''
OHR
'''
#df = common_cmpds_ppb.copy()
df = full_cmpd_dataset.copy()

starttime = df['start_time'] # storing for later
df.drop(columns = ['start_time', 'hour'], inplace = True)

cmpd_list = df.columns # need list of compounds for filtering the ref_data

ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# need compound names as index
df = df.transpose()

for index, row in df.iterrows(): # applying number density function 
    for col in row.index:
     if pd.api.types.is_numeric_dtype(df[col]):
         df.at[index, col] = number_density(
             row[col],
         )

# converting to reactivities against OH (1/s)
for col in df:
    df[col] = df[col] * ref_data['k_oh_298']

# flipping compounds back up to columns and putting back datetime column
df = df.transpose()
df_OH_timeseries = pd.DataFrame()
df_OH_timeseries.insert(loc = 0, column = 'datetime', value = starttime, allow_duplicates = False)
df_OH_timeseries['total_oh'] = df.sum(axis = 1, numeric_only = True) 

df_OH_timeseries.set_index('datetime', inplace = True)
'''
calculating OFP for all pmf compounds
'''
#df_ofp = common_cmpds_ppb.copy()
df_ofp = full_cmpd_dataset.copy()

df_ofp.drop(columns = ['hour'], inplace = True)
df_ofp.set_index('start_time', inplace = True)
# want cmpds as row names and samples as columns:
df_ofp = df_ofp.transpose()
# want to only look at compounds included in the PMF input:
factor_df = profile_conc.copy() 
factor_df.set_index('compounds', inplace = True)
factor_cmpd_list = factor_df.index

df_ofp = df_ofp.loc[factor_cmpd_list] # filtering
cmpd_list = df_ofp.index # list of cmpds in the ground samples

# filtering the reference data for those cmpds
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# currently, df = matrix of ppb values, samples as columns and cmpds as rows
# converting to OFP (ppb)
for value in df_ofp:
    df_ofp[value] = df_ofp[value] * ref_data['MIR_ppb']
    
df_ofp = df_ofp.transpose()
df_ofp.reset_index(inplace = True, drop = True)
ofp_sum = df_ofp.copy()
ofp_sum = pd.DataFrame(data = ofp_sum.sum(axis = 1), columns = ['summed_ofp']) # summing for each date
ofp_sum.set_index(starttime, inplace = True)
df_ofp = pd.DataFrame(data = df_ofp.median(), columns = ['median_ofp'])

# Timeseries plot
fig, (ax1, ax2) = plt.subplots(
    nrows = 2,
    ncols = 1,
    sharex = True,
    figsize = (10, 8),
    dpi = 300)
# OHR
tracer = df_OH_timeseries.columns[0]
ts_plot(chosen_df = df_OH_timeseries, # applying previousy definied timeseries wave 
               tracer = tracer,
               axis = ax1)
ax1.get_legend().remove() # don't need the legend since we're only plotting total OH
# setting y-ticks + y-label
ax1.set_ylabel(
    "∑OHR ($sec^{-1}$)",
    size = 14,
    fontweight = 'normal'
    )
ax1.set_yticks(
    ticks = (np.linspace(0, df_OH_timeseries[tracer].max(), 6)),
    labels = (np.linspace(0, df_OH_timeseries[tracer].max(), 6)),
    size = 12,
    fontweight = 'normal'
    )
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
units = 'sec$^-$$^1$'
ax1.text(
    x = 0.7,
    y = 0.75,
    s = f'Average: {df_OH_timeseries['total_oh'].mean():0.1f} (± {df_OH_timeseries['total_oh'].std():0.1f}) {units}\nMinimum: {df_OH_timeseries['total_oh'].min():0.1f} {units}\nMaximum: {df_OH_timeseries['total_oh'].max():0.1f} {units}',
    transform = ax1.transAxes,
    fontsize = 12
    )
# OFP
tracer = ofp_sum.columns[0]
units = '$ppb_v$'
ts_plot(chosen_df = ofp_sum, # applying previousy definied timeseries wave 
               tracer = tracer,
               axis = ax2)
ax2.get_legend().remove() # don't need the legend since we're only plotting total OH
# setting y-ticks + y-label
ax2.set_ylabel(
    "∑OFP ($sec^{-1}$)",
    size = 14,
    fontweight = 'normal'
    )
ax2.set_yticks(
    ticks = (np.linspace(0, ofp_sum[tracer].max(), 6)),
    labels = (np.linspace(0, ofp_sum[tracer].max(), 6)),
    size = 12,
    fontweight = 'normal'
    )
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.text(
    x = 0.7,
    y = 0.75,
    s = f'Average: {ofp_sum['summed_ofp'].mean():0.0f} (± {ofp_sum['summed_ofp'].std():0.0f}) {units}\nMinimum: {ofp_sum['summed_ofp'].min():0.1f} {units}\nMaximum: {ofp_sum['summed_ofp'].max():0.0f} {units}',
    transform = ax2.transAxes,
    fontsize = 12
    )
# setting x-ticks and labels
plt.xticks(fontsize = 12, fontweight = 'normal')
fig.autofmt_xdate()
ax1.xaxis.set_major_formatter(md.DateFormatter('%D'))

for ax, s in zip([ax1, ax2], ['A', 'B']):
    ax.text(
        x = -0.1,
        y = 0.95,
        s = f'{s}',
        fontweight = 'bold',
        fontsize = 14,
        transform = ax.transAxes
        )



plt.show()
# %% # calculating total OFP, OHR, MR in ug/m3 for Figure 4

totals = {}

'''
Calculating total OH for all PMF compounds:
'''

#df_oh = common_cmpds_ppb.copy()
df_oh = full_cmpd_dataset.copy()


starttime = df_oh['start_time'] # storing for later
hour = df_oh['hour'] # storing for later
df_oh.drop(columns = ['start_time', 'hour'], inplace = True)

cmpd_list = df_oh.columns # need list of compounds for filtering the ref_data
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# need compound names as index
df_oh = df_oh.transpose()
for index, row in df_oh.iterrows(): # applying number density function 
    for col in row.index:
     if pd.api.types.is_numeric_dtype(df_oh[col]):
         df_oh.at[index, col] = number_density(
             row[col],
         )
# converting to reactivities against OH (1/s)
for col in df_oh:
    df_oh[col] = df_oh[col] * ref_data['k_oh_298']

df_oh = df_oh.transpose() # flipping compounds back up to columns and putting back datetime column
units_oh = 'sec$^-$$^1$'
df_oh = pd.DataFrame(data = df_oh.median(), columns = ['median_oh'])

'''
calculating OFP for all pmf compounds
'''
#df_ofp = common_cmpds_ppb.copy()
df_ofp = full_cmpd_dataset.copy()

df_ofp.drop(columns = ['hour'], inplace = True)
df_ofp.set_index('start_time', inplace = True)
# want cmpds as row names and samples as columns:
df_ofp = df_ofp.transpose()
# want to only look at compounds included in the PMF input:
factor_df = profile_conc.copy() 
factor_df.set_index('compounds', inplace = True)
factor_cmpd_list = factor_df.index

df_ofp = df_ofp.loc[factor_cmpd_list] # filtering
cmpd_list = df_ofp.index # list of cmpds in the ground samples

# filtering the reference data for those cmpds
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# currently, df = matrix of ppb values, samples as columns and cmpds as rows
df_ofp = df_ofp.transpose()
df_ofp.reset_index(inplace = True)
meta_data_copy = meta_data.copy()
for df in [meta_data_copy, df_ofp]:
    df.set_index('start_time', inplace = True)
    
# NEW: converting to ug/m3
df_ofp_ug_m3 = ug_m3_from_ppb(df_ofp, ref_data, meta_data_copy)

# converting to OFP (ug/m3)
df_ofp_ug_m3 = df_ofp_ug_m3.transpose() # need compounds as index
for value in df_ofp_ug_m3:
    df_ofp_ug_m3[value] = df_ofp_ug_m3[value] * ref_data['MIR_ppb']
# reset 
df_ofp_ug_m3 = df_ofp_ug_m3.transpose()
df_ofp.reset_index(inplace = True, drop = True)
ofp_sum = df_ofp_ug_m3.copy()
ofp_sum = pd.DataFrame(data = ofp_sum.sum(axis = 1), columns = ['summed_ofp']) # summing for each date
ofp_sum.set_index(starttime, inplace = True)
df_ofp = pd.DataFrame(data = df_ofp.median(), columns = ['median_ofp'])
units_ofp = '$µg/m^3$'

'''
MR df too 
'''
#df_mr = common_cmpds_ppb.copy()
df_mr = full_cmpd_dataset.copy()

df_mr.set_index('start_time', inplace = True)
df_mr.drop(columns = ['hour'], inplace = True)
variable_count = len(df_mr.index) 
compounds = df_mr.columns

ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[compounds] # filtering for cmpds included in PMF solutio


# NEW: converting to ug/m3
df_mr_ug_m3 = ug_m3_from_ppb(df_mr, ref_data, meta_data_copy)
                
units_mr = '$µg/m^3$'
df_mr = pd.DataFrame(data = df_mr_ug_m3.median(), columns = ['median_mr'])

df_copy = df_mr.copy()
df_compare = df_copy.transpose()
df_compare = df_compare[['2-methyl-undecanal', 
                         'ethylhexyl benzoate', 'diphenyl ether',
                         'allyl heptanoate', 'phenoxyisopropanol', 'benzyl acetate',
                         'benzyl alcohol']]
df_compare 
 
'''
final dataframes
'''
df = pd.concat([df_mr, df_oh, df_ofp], axis = 1)
df_pct = df.div(df.sum(axis=0), axis=1) * 100
df_total = pd.DataFrame(data = df.sum(), columns = ['totals'])

ug_m3_median_list = df_total['totals']


# %% Figure 4 barchart option: Median MR, OHR, and OFP (% distribution bar chart)  # updated w/ full dataset

totals = {}

'''
Calculating total OH for all compounds:
'''
#df_oh = common_cmpds_ppb.copy()
df_oh = full_cmpd_dataset.copy()

starttime = df_oh['start_time'] # storing for later
hour = df_oh['hour'] # storing for later
df_oh.drop(columns = ['start_time', 'hour'], inplace = True)

cmpd_list = df_oh.columns # need list of compounds for filtering the ref_data
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# need compound names as index
df_oh = df_oh.transpose()
for index, row in df_oh.iterrows(): # applying number density function 
    for col in row.index:
     if pd.api.types.is_numeric_dtype(df_oh[col]):
         df_oh.at[index, col] = number_density(
             row[col],
         )
# converting to reactivities against OH (1/s)
for col in df_oh:
    df_oh[col] = df_oh[col] * ref_data['k_oh_298']

df_oh = df_oh.transpose() # flipping compounds back up to columns and putting back datetime column
units_oh = 'sec$^-$$^1$'
df_oh = pd.DataFrame(data = df_oh.median(), columns = ['median_oh'])

'''
calculating OFP for all pmf compounds
'''
#df_ofp = common_cmpds_ppb.copy()
df_ofp = full_cmpd_dataset.copy()

df_ofp.drop(columns = ['hour'], inplace = True)
df_ofp.set_index('start_time', inplace = True)
# want cmpds as row names and samples as columns:
df_ofp = df_ofp.transpose()
# want to only look at compounds included in the PMF input:
factor_df = profile_conc.copy() 
factor_df.set_index('compounds', inplace = True)
factor_cmpd_list = factor_df.index

df_ofp = df_ofp.loc[factor_cmpd_list] # filtering
cmpd_list = df_ofp.index # list of cmpds in the ground samples

# filtering the reference data for those cmpds
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# currently, df = matrix of ppb values, samples as columns and cmpds as rows
# converting to OFP (ppb)
for value in df_ofp:
    df_ofp[value] = df_ofp[value] * ref_data['MIR_ppb']
    
df_ofp = df_ofp.transpose()
df_ofp.reset_index(inplace = True, drop = True)
df_ofp = pd.DataFrame(data = df_ofp.median(), columns = ['median_ofp'])
units_ofp = '$ppb_v$'

'''
MR df too 
'''
#df_mr = common_cmpds_ppb.copy()
df_mr = full_cmpd_dataset.copy()

df_mr.drop(columns = ['start_time', 'hour'], inplace = True)
variable_count = len(df_mr.index) 
compounds = df_mr.columns
units_mr = '$ppb_v$'
df_mr = pd.DataFrame(data = df_mr.median(), columns = ['median_mr'])
 
'''
final dataframes
'''
df = pd.concat([df_mr, df_oh, df_ofp], axis = 1)
df_pct = df.div(df.sum(axis=0), axis=1) * 100
df_total = df.sum()
df = df.transpose()
df_pct = df_pct.transpose()
df_total = df_total.transpose()

units = [units_mr, units_oh, units_ofp]

'''
Making the figure
'''

df = df_pct.copy()
df.index = ['Mixing Ratio', 'OHR', 'OFP']

# need list of compounds
variable_count = len(df.index) # of factors

df.rename(columns = {'4-tert-butylcyclohexyl acetate peak 1' : '4-tert-butylcyclohexyl acetate 1',
                       '4-tert-butylcyclohexyl acetate peak 2' : '4-tert-butylcyclohexyl acetate 2',
                       '2-tert-butylcyclohexyl acetate peak 1' : '2-tert-butylcyclohexyl acetate 1'}, 
                 inplace = True)

compounds = df.columns

fontsize = 26
fontweight = 'normal'


fig, ax = plt.subplots(figsize = (20, 20), dpi = 300)

values = df.to_numpy()
x = np.arange(len(df))
width = 0.97

values = df.to_numpy()
x = np.arange(len(df))
width = 0.97

n_rows = len(df)
n_cols = len(df.columns)
top_n = 7

top_cols_global = set()
bar_patches = {}

# Plot each bar, sorting its segments from smallest to largest
for bar_idx, (index, row) in enumerate(df.iterrows()):
    row_values = pd.to_numeric(row, errors='coerce').fillna(0)

    top_cols = set(row_values.nlargest(top_n).index)
    top_cols_global.update(top_cols)

    order = row_values.sort_values(ascending=True, kind='stable').index
    bottom = 0

    for col in order:
        height = row_values[col]

        container = ax.bar(
            x[bar_idx],
            height,
            bottom=bottom,
            width=width,
            color=cmpd_colors_dict.get(col, 'gray'),
            edgecolor='white',
            linewidth=1.5,
        )

        # Store the patch using its bar index and original compound name
        bar_patches[(bar_idx, col)] = container.patches[0]

        bar = bar_patches[(bar_idx, col)]
        bar.set_facecolor(cmpd_colors_dict.get(col, 'gray') if col in top_cols else '#87ceeb')
        bar.set_alpha(1 if col in top_cols else 1)
        bar.set_edgecolor('black' if col in top_cols else '#87ceeb')
        bar.set_hatch(hatch_dict.get(col, ''))

        # Label the top 7 compounds at their actual stacked positions
        if col in top_cols and height > 0:
            wrapped_label = '\n'.join(textwrap.wrap(col, width=20.5))

            ax.text(
                x=bar_idx,
                y=bottom + height / 2,
                s=f'{wrapped_label}: {height:.1f}%',
                ha='center',
                va='center',
                fontsize=20 if height < 4 else 26,
                color='black'
            )

        bottom += height

ax.set_xticks(x,
              df.index,
              fontsize = 30)
ax.set_ylim(0, 100)
ax.set_yticks(
    ticks = np.linspace(0, 100, 6),
    labels = np.linspace(0, 100, 6),
    fontsize = 34,
    fontweight = 'bold'
    )

ax.set_ylabel(
    'Compound Distribution (%)',
    fontsize=40,
    fontweight='bold'
)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


labels = ['Industrial', 'Fragrances', 'Biogenic', 'Other']
legend_colors = ['#B0B0B0', 'hotpink', 'Green', '#87ceeb']

hatches = ['', '', '\\', '']

# adding manual source legend
patches = []
for i, color, h in zip(labels, legend_colors, hatches):
    leg = mpatches.Patch(facecolor = color, 
                         edgecolor = 'black',
                         label = i,
                         hatch = h)
    patches.append(leg)
ax.legend(handles = patches, 
          loc = 'lower center', 
          bbox_to_anchor = (0.5, -0.1),
          fontsize = 28,
          ncols = 4, 
         #columnspacing = 15
          )

ug_m3_median_list = [7.3, 0.5, 1.3] # need to manually update this for now 

for i, (total, unit, ug) in enumerate(zip(df_total, units, ug_m3_median_list)):
    ax.text(i, 99 * 1.02, f'{total:0.1f} {unit}\n[{ug} μg m$^-$$^3$]', 
            ha='center', 
            fontsize = 28, 
            fontweight = 'normal')

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.show()
plt.close()

# %% Median MR, OHR, and OFP (% distribution pie charts)  # updated w/ full dataset

totals = {}

'''
Calculating total OH for all PMF compounds:
'''
#df_oh = common_cmpds_ppb.copy()
df_oh = full_cmpd_dataset.copy()

starttime = df_oh['start_time'] # storing for later
hour = df_oh['hour'] # storing for later
df_oh.drop(columns = ['start_time', 'hour'], inplace = True)

cmpd_list = df_oh.columns # need list of compounds for filtering the ref_data
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# need compound names as index
df_oh = df_oh.transpose()
for index, row in df_oh.iterrows(): # applying number density function 
    for col in row.index:
     if pd.api.types.is_numeric_dtype(df_oh[col]):
         df_oh.at[index, col] = number_density(
             row[col],
         )
# converting to reactivities against OH (1/s)
for col in df_oh:
    df_oh[col] = df_oh[col] * ref_data['k_oh_298']

df_oh = df_oh.transpose() # flipping compounds back up to columns and putting back datetime column
units_oh = 'sec$^-$$^1$'
df_oh = pd.DataFrame(data = df_oh.median(), columns = ['median_oh'])

'''
calculating OFP for all pmf compounds
'''
#df_ofp = common_cmpds_ppb.copy()
df_ofp = full_cmpd_dataset.copy()

df_ofp.drop(columns = ['hour'], inplace = True)
df_ofp.set_index('start_time', inplace = True)
# want cmpds as row names and samples as columns:
df_ofp = df_ofp.transpose()
# want to only look at compounds included in the PMF input:
factor_df = profile_conc.copy() 
factor_df.set_index('compounds', inplace = True)
factor_cmpd_list = factor_df.index

df_ofp = df_ofp.loc[factor_cmpd_list] # filtering
cmpd_list = df_ofp.index # list of cmpds in the ground samples

# filtering the reference data for those cmpds
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# currently, df = matrix of ppb values, samples as columns and cmpds as rows
# converting to OFP (ppb)
for value in df_ofp:
    df_ofp[value] = df_ofp[value] * ref_data['MIR_ppb']
    
df_ofp = df_ofp.transpose()
df_ofp.reset_index(inplace = True, drop = True)
ofp_sum = df_ofp.copy()
ofp_sum = pd.DataFrame(data = ofp_sum.sum(axis = 1), columns = ['summed_ofp']) # summing for each date
ofp_sum.set_index(starttime, inplace = True)
df_ofp = pd.DataFrame(data = df_ofp.median(), columns = ['median_ofp'])
units_ofp = '$ppb_v$'

'''
MR df too 
'''
#df_mr = common_cmpds_ppb.copy()
df_mr = full_cmpd_dataset.copy()

df_mr.drop(columns = ['start_time', 'hour'], inplace = True)
variable_count = len(df_mr.index) 
compounds = df_mr.columns
units_mr = '$ppb_v$'
df_mr = pd.DataFrame(data = df_mr.median(), columns = ['median_mr'])
 
'''
final dataframes
'''
df = pd.concat([df_mr, df_oh, df_ofp], axis = 1)
df_pct = df.div(df.sum(axis=0), axis=1) * 100
df_total = pd.DataFrame(data = df.sum(), columns = ['totals'])
#df = df.transpose()
#df_pct = df_pct.transpose()
#df_total = df_total.transpose()

units = [units_mr, units_oh, units_ofp]

'''
Figure
'''

def alternate_order(series):
    s = series.sort_values(ascending=False)
    left = list(s.index)
    
    alt = []
    while left:
        alt.append(left.pop(0))  # largest
        if left:
            alt.append(left.pop(-1))  # smallest
    return s.loc[alt]

def autopct_func(pct):
    return f'{pct:.1f}%' if pct >= cutoff else ''

# Columns and units
columns = df_pct.columns
totals = df_total.values
labels = ['Conc.', 'Reactivity', 'Ozone']
units = [units_mr, units_oh, units_ofp]
subplots = ['A', 'B', 'C']

# Create a figure with 1 row and 3 columns
gridspec = {'wspace' : 0.3}
fig, axes = plt.subplots(
    ncols = 1, 
    nrows = 3, 
    figsize = (12, 22),
    dpi = 300,
    gridspec_kw = gridspec
    )
cutoff = 5.1
all_labels = set()
ug_m3_totals = [8.2, '', 1.5]

for ax, col, unit, ug, l, s, t in zip(axes, columns, 
                                  units, ug_m3_median_list, 
                                  labels, subplots, totals):
    # Sort values descending and take top 5
    top5 = df_pct[col].sort_values(ascending=False)[:7]
    top5 = alternate_order(top5)
    other = df_pct[col].sum() - top5.sum()
    
    # Combine top5 with 'Other' using pd.concat
    pie_values = pd.concat([top5, pd.Series({'other': other})])
    
    # Update set of all labels
    all_labels.update(pie_values.index.tolist())
    
    wedges, texts, autotexts = ax.pie(
        pie_values,
        labels = pie_values.index,
        wedgeprops = {'edgecolor' : 'black', 'linewidth' : 2},
        autopct = autopct_func,
        startangle = 90,
        colors = [cmpd_colors_dict[value] for value in pie_values.index],
        textprops = {'fontsize' : 20, 'fontweight' : 'normal'},
        pctdistance = 0.83,
        radius = 1.25
    )
    ax.set_title(f'{l}: {t.item():.1f} {unit} [{ug} $µg/m^3$]',
                 fontsize = 24,
                 fontweight = 'bold',
                 pad = 20)
    ax.axis('equal')  # ensures pie chart is circular
    
    ax.text(
        x = 0.09,
        y = 1.06, 
        s = f'{s}',
        transform = ax.transAxes,
        fontsize = 24,
        fontweight = 'bold'
        )
    
    total = pie_values.sum()
    
    for wedge, value in zip(wedges, pie_values):
        
        pct = value / total * 100
        
        if pct < cutoff:
            
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            
            ax.annotate(
                f'{pct:.1f}%',
                xy=(x, y),
                xytext=(1.45*np.sign(x), 1.2*y),
                arrowprops=dict(arrowstyle='-', lw=1.5),
                ha='center',
                va='center',
                fontsize=18
            )    

# # --- Create a single legend with all unique categories ---
# all_labels = sorted(all_labels)  # optional: sort alphabetically
# fig.legend(
#     handles=[plt.Line2D([0], [0], marker='o', color='w', label=lbl,
#                         markerfacecolor=cmpd_colors_dict[lbl], markersize=20)
#              for lbl in all_labels],
#     loc = 'center left',
#     ncol = 1,
#     fontsize=18,
#     frameon=True,
#     bbox_to_anchor = (-1, 1)
    
# )


plt.show()

# %% ug/m3, Median MR, OHR, and OFP (% distribution pie charts) # updated w/ full dataset

totals = {}

'''
Calculating total OH for all PMF compounds:
'''

#df_oh = common_cmpds_ppb.copy()
df_oh = full_cmpd_dataset.copy()


starttime = df_oh['start_time'] # storing for later
hour = df_oh['hour'] # storing for later
df_oh.drop(columns = ['start_time', 'hour'], inplace = True)

cmpd_list = df_oh.columns # need list of compounds for filtering the ref_data
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# need compound names as index
df_oh = df_oh.transpose()
for index, row in df_oh.iterrows(): # applying number density function 
    for col in row.index:
     if pd.api.types.is_numeric_dtype(df_oh[col]):
         df_oh.at[index, col] = number_density(
             row[col],
         )
# converting to reactivities against OH (1/s)
for col in df_oh:
    df_oh[col] = df_oh[col] * ref_data['k_oh_298']

df_oh = df_oh.transpose() # flipping compounds back up to columns and putting back datetime column
units_oh = 'sec$^-$$^1$'
df_oh = pd.DataFrame(data = df_oh.median(), columns = ['median_oh'])

'''
calculating OFP for all pmf compounds
'''
#df_ofp = common_cmpds_ppb.copy()
df_ofp = full_cmpd_dataset.copy()

df_ofp.drop(columns = ['hour'], inplace = True)
df_ofp.set_index('start_time', inplace = True)
# want cmpds as row names and samples as columns:
df_ofp = df_ofp.transpose()
# want to only look at compounds included in the PMF input:
factor_df = profile_conc.copy() 
factor_df.set_index('compounds', inplace = True)
factor_cmpd_list = factor_df.index

df_ofp = df_ofp.loc[factor_cmpd_list] # filtering
cmpd_list = df_ofp.index # list of cmpds in the ground samples

# filtering the reference data for those cmpds
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# currently, df = matrix of ppb values, samples as columns and cmpds as rows
df_ofp = df_ofp.transpose()
df_ofp.reset_index(inplace = True)
meta_data_copy = meta_data.copy()
for df in [meta_data_copy, df_ofp]:
    df.set_index('start_time', inplace = True)
    
# NEW: converting to ug/m3
df_ofp_ug_m3 = ug_m3_from_ppb(df_ofp, ref_data, meta_data_copy)

# converting to OFP (ug/m3)
df_ofp_ug_m3 = df_ofp_ug_m3.transpose() # need compounds as index
for value in df_ofp_ug_m3:
    df_ofp_ug_m3[value] = df_ofp_ug_m3[value] * ref_data['MIR_ppb']
# reset 
df_ofp_ug_m3 = df_ofp_ug_m3.transpose()
df_ofp.reset_index(inplace = True, drop = True)
ofp_sum = df_ofp_ug_m3.copy()
ofp_sum = pd.DataFrame(data = ofp_sum.sum(axis = 1), columns = ['summed_ofp']) # summing for each date
ofp_sum.set_index(starttime, inplace = True)
df_ofp = pd.DataFrame(data = df_ofp.median(), columns = ['median_ofp'])
units_ofp = '$µg/m^3$'

'''
MR df too 
'''
#df_mr = common_cmpds_ppb.copy()
df_mr = full_cmpd_dataset.copy()

df_mr.set_index('start_time', inplace = True)
df_mr.drop(columns = ['hour'], inplace = True)
variable_count = len(df_mr.index) 
compounds = df_mr.columns

ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[compounds] # filtering for cmpds included in PMF solutio


# NEW: converting to ug/m3
df_mr_ug_m3 = ug_m3_from_ppb(df_mr, ref_data, meta_data_copy)
                
units_mr = '$µg/m^3$'
df_mr = pd.DataFrame(data = df_mr_ug_m3.median(), columns = ['median_mr'])

df_copy = df_mr.copy()
df_compare = df_copy.transpose()
df_compare = df_compare[['2-methyl-undecanal', 
                         'ethylhexyl benzoate', 'diphenyl ether',
                         'allyl heptanoate', 'phenoxyisopropanol', 'benzyl acetate',
                         'benzyl alcohol']]
df_compare 
 
'''
final dataframes
'''
df = pd.concat([df_mr, df_oh, df_ofp], axis = 1)
df_pct = df.div(df.sum(axis=0), axis=1) * 100
df_total = pd.DataFrame(data = df.sum(), columns = ['totals'])
#df = df.transpose()
#df_pct = df_pct.transpose()
#df_total = df_total.transpose()

units = [units_mr, units_oh, units_ofp]

'''
Figure
'''

def alternate_order(series):
    s = series.sort_values(ascending=False)
    left = list(s.index)
    
    alt = []
    while left:
        alt.append(left.pop(0))  # largest
        if left:
            alt.append(left.pop(-1))  # smallest
    return s.loc[alt]

def autopct_func(pct):
    return f'{pct:.1f}%' if pct >= cutoff else ''

# Columns and units
columns = df_pct.columns
totals = df_total.values
labels = ['Mixing Ratio', 'OHR', 'OFP']
units = [units_mr, units_oh, units_ofp]
subplots = ['A', 'B', 'C']

# Create a figure with 1 row and 3 columns
gridspec = {'wspace' : 0.3}
fig, axes = plt.subplots(
    ncols = 1, 
    nrows = 3, 
    figsize = (12, 22),
    dpi = 300,
    gridspec_kw = gridspec
    )
cutoff = 4.7
all_labels = set()

for ax, col, unit, l, s, t in zip(axes, columns, 
                                  units, labels, subplots, totals):
    # Sort values descending and take top 5
    top5 = df_pct[col].sort_values(ascending=False)[:7]
    top5 = alternate_order(top5)
    other = df_pct[col].sum() - top5.sum()
    
    # Combine top5 with 'Other' using pd.concat
    pie_values = pd.concat([top5, pd.Series({'other': other})])
    
    # Update set of all labels
    all_labels.update(pie_values.index.tolist())
    
    wedges, texts, autotexts = ax.pie(
        pie_values,
      #  labels = pie_values.index,
        wedgeprops = {'edgecolor' : 'black', 'linewidth' : 2},
        autopct = autopct_func,
        startangle = 10,
        colors = [cmpd_colors_dict[value] for value in pie_values.index],
        textprops = {'fontsize' : 20, 'fontweight' : 'normal'},
        pctdistance = 0.83,
        radius = 1.25
    )
    ax.set_title(f'{l}: {t.item():.1f} {unit}',
                 fontsize = 24,
                 fontweight = 'bold',
                 pad = 20)
    ax.axis('equal')  # ensures pie chart is circular
    
    ax.text(
        x = 0.15,
        y = 1.05, 
        s = f'{s}',
        transform = ax.transAxes,
        fontsize = 24,
        fontweight = 'bold'
        )
    
    total = pie_values.sum()
    
    for wedge, value in zip(wedges, pie_values):
        
        pct = value / total * 100
        
        if pct < cutoff:
            
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            
            ax.annotate(
                f'{pct:.1f}%',
                xy=(x, y),
                xytext=(1.45*np.sign(x), 1.2*y),
                arrowprops=dict(arrowstyle='-', lw=1.5),
                ha='center',
                va='center',
                fontsize=18
            )    

# # --- Create a single legend with all unique categories ---
# all_labels = sorted(all_labels)  # optional: sort alphabetically
# fig.legend(
#     handles=[plt.Line2D([0], [0], marker='o', color='w', label=lbl,
#                         markerfacecolor=cmpd_colors_dict[lbl], markersize=20)
#              for lbl in all_labels],
#     loc = 'center left',
#     ncol = 1,
#     fontsize=18,
#     frameon=True,
#     bbox_to_anchor = (-1, 1)
    
# )


plt.show()

# %%  SI Figure 4: MT vs temperature plots

df_residual = scaled_residual_ts.copy()
df_residual.drop(df_residual.tail(1).index,inplace = True) # dropping Q/Qexp row


d = {
     'a-pinene' : full_cmpd_dataset['a-pinene'],
     'limonene' : full_cmpd_dataset['limonene'],
     'pine_res' : df_residual['a-pinene'],
     'lim_res' : df_residual['limonene'],
     'D5-siloxane' : full_cmpd_dataset['decamethylcyclopentasiloxane'],
     'Median Temperature' : meta_data['median_temp_c'],
     'Average Temperature' : meta_data['avg_temp_c'],
     'date_time' : full_cmpd_dataset['start_time'],
     'hour' : full_cmpd_dataset['hour']
     }
df = pd.DataFrame(data = d)
df = df.replace(-9999, np.nan)
df.dropna(inplace = True, ignore_index = True)



df.set_index('date_time', inplace = True)
df_am = df.between_time('05:00', '12:00')
df_pm = df.between_time('12:01', '23:00')

label_fontsize = 14
legend_fontsize = 12
tick_fontsize = 12

fig, ((ax2, ax1), (ax3, ax4)) = plt.subplots(
    ncols = 2,
    nrows = 2,
    sharex = True,
    figsize = (12, 12), 
    dpi = 300)

# ax2 = top left plot, lim vs pine vs temp
simple_corr_plot(
    axis = ax2,
    x = 'Average Temperature',
    y = 'limonene',
    df = df,
    num_dec = 0.3,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = '#C0B141',
    linecolor = '#C0B141',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 10,
    legend_fontsize = legend_fontsize,
    loc = 'upper left',
    x_unit = 'C',
    y_unit = '$ppb_v$'
    )
simple_corr_plot(
    axis = ax2,
    x = 'Average Temperature',
    y = 'a-pinene',
    df = df,
    num_dec = 0.3,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = '#9B287B',
    linecolor = '#9B287B',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 10,
    legend_fontsize = legend_fontsize,
    loc = 'upper left',
    x_unit = 'C',
    y_unit = '$ppb_v$'
    )
ax2.set_ylim([-0.003, df['limonene'].max() + 0.005])
ax2.set_yticks(
    ticks = np.linspace(0, df['limonene'].max(), 5),
    labels = np.linspace(0, df['limonene'].max(), 5)
    )
ax2.legend(
    loc = 'upper right',
    ncols = 1,
    frameon = False,
 #   bbox_to_anchor = (0, 0.93),
    fontsize = 14
    )
ax2.set_ylabel(
    'Mixing Ratio ($ppb_v$)',
    fontsize = 14
    )

# ax1 = top right plot, pine vs temp vs tod 
simple_corr_plot(
    axis = ax1,
    x = 'Average Temperature',
    y = 'a-pinene',
    df = df_am,
    num_dec = 0.3,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'orange',
    linecolor = 'orange',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 10,
    legend_fontsize = legend_fontsize,
    loc = 'upper right',
    x_unit = 'C',
    y_unit = '$ppb_v$'
    )
simple_corr_plot(
    axis = ax1,
    x = 'Average Temperature',
    y = 'a-pinene',
    df = df_pm,
    num_dec = 0.4,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'darkblue',
    linecolor = 'darkblue',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 10,
    legend_fontsize = legend_fontsize,
    loc = 'upper right',
    x_unit = 'C',
    y_unit = '$ppb_v$'
    )
ax1.set_ylim([-0.002, df['a-pinene'].max()+.01])
ax1.set_yticks(
    ticks = np.linspace(0, df['a-pinene'].max() + 0.01, 5),
    labels = np.linspace(0, df['a-pinene'].max() + 0.01, 5)
    )
ax1.set_xticks(
    ticks = np.linspace(df['Average Temperature'].min(), df['Average Temperature'].max(), 5),
    labels = np.linspace(df['Average Temperature'].min(), df['Average Temperature'].max(), 5)
    )

ax1.legend(
    loc = 'upper left',
    ncols = 1,
    frameon = False,
   # bbox_to_anchor = (0.45, 0.88),
    fontsize = 14
    )

# ax3 = bottom left, lim vs temp vs tod 
simple_corr_plot(
    axis = ax3,
    x = 'Average Temperature',
    y = 'limonene',
    df = df_pm,
    num_dec = 0.4,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'darkblue',
    linecolor = 'darkblue',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 10,
    legend_fontsize = legend_fontsize,
    loc = 'upper right',
    x_unit = 'C',
    y_unit = '$ppb_v$'
    )
simple_corr_plot(
    axis = ax3,
    x = 'Average Temperature',
    y = 'limonene',
    df = df_am,
    num_dec = 0.3,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'orange',
    linecolor = 'orange',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 10,
    legend_fontsize = legend_fontsize,
    loc = 'upper right',
    x_unit = 'C',
    y_unit = '$ppb_v$'
    )
ax3.set_ylim([-0.002, df['limonene'].max() + 0.005])
ax3.set_yticks(
    ticks = np.linspace(0, df['limonene'].max(), 5),
    labels = np.linspace(0, df['limonene'].max(), 5)
    )
ax3.legend(
    loc = 'upper left',
    ncols = 1,
    frameon = False,
   # bbox_to_anchor = (0.45, 0.88),
    fontsize = 14
    )
# ax4 = bottom left, scaled residuals vs temp
simple_corr_plot(
    axis = ax4,
    x = 'Average Temperature',
    y = 'lim_res',
    df = df,
    num_dec = 0.2,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = '#C0B141',
    linecolor = '#C0B141',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 10,
    legend_fontsize = legend_fontsize,
    loc = 'upper left',
    x_unit = 'C',
    y_unit = ''
    )
simple_corr_plot(
    axis = ax4,
    x = 'Average Temperature',
    y = 'pine_res',
    df = df,
    num_dec = 0.2,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = '#9B287B',
    linecolor = '#9B287B',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 10,
    legend_fontsize = legend_fontsize,
    loc = 'upper left',
    x_unit = 'C',
    y_unit = ''
    )
ax4.set_ylim([-1, 2])
ax4.set_yticks(
    ticks = np.linspace(-1, 2, 5),
    labels = np.linspace(-1, 2, 5)
    )
ax4.set_ylabel(
    'Scaled Factor Residual',
    fontsize = label_fontsize
    )
ax4.legend(
    loc = 'upper left',
    ncols = 1,
    frameon = False,
   # bbox_to_anchor = (0.45, 0.88),
    fontsize = 14
    )

# loops
for ax in [ax1, ax2]:
    ax.set_xlabel('')
for ax in [ax1, ax3]:    
    ax.text(
            x = 0.15,
            y = 1.03,
            s = '06:00 - 12:00',
            fontsize = 14,
            color = 'orange',
            fontweight = 'bold',
            transform = ax.transAxes
            )
    ax1.text(
            x = 0.55,
            y = 1.03,
            s = '12:00 - 18:00',
            fontsize = 14,
            color = 'darkblue',
            fontweight = 'bold',
            transform = ax.transAxes
            )
for ax in [ax2, ax4]:
    ax.text(
            x = 0.15,
            y = 1.03,
            s = 'limonene',
            color = '#C0B141',
            fontsize = 14,
            fontweight = 'bold',
            transform = ax.transAxes
            )
    ax.text(
            x = 0.55,
            y = 1.03,
            s = 'a-pinene',
            color = '#9B287B',
            fontsize = 14,
            fontweight = 'bold',
            transform = ax.transAxes
            )
for ax in [ax3, ax4]:
    ax.set_xticks(
        ticks = np.linspace(df['Average Temperature'].min(), df['Average Temperature'].max(), 5),
        labels = np.linspace(df['Average Temperature'].min(), df['Average Temperature'].max(), 5)
        )
for ax in [ax1, ax2, ax3, ax4]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
    ax.set_xlim([df['Average Temperature'].min() - 0.5, df['Average Temperature'].max() + 0.5])
for i, ax, in enumerate([ax2, ax1, ax3, ax4]):
    ax.text(
        x = 0.0, 
        y = 1.03,
        s = chr(65 + i),
        fontsize = label_fontsize,
        transform = ax.transAxes,
        fontweight = 'bold'
        )

plt.show()
plt.close()

 # %% SI Figure: Limonene vs. a-pinene (two OLS plots)

import statsmodels.api as sm # re-loading bc dumb new error

# days of high Factor 5: 
    # Factor 5: July 17, ~6 and 10am (indicies 5, 6) and August 2 (index 20)

only_f5_index = [5, 6, 20]

# re-writing previously defined OLS outlier plotting function b/c I need a different legend

# function to create correlation plots, ignoring outlier points
def ols_corr_plot_outliers (axis, x, y, df, index, 
                   color, num_dec, label_fontsize, tick_fontsize, 
                   fontweight, markersize, legend_fontsize, loc):
    
    import statsmodels.api as sm
    
    df_all = df.copy()
    df_no_out =  df_all.drop(index = index) # dropping outliers
    df_no_out.reset_index(inplace = True, drop = True) # dropping old index column
    # OLS model with all points:
    y_variable_all = df_all[y]
    x_variable_all = df_all[x]
    pearson_all = scipy.pearsonr(x_variable_all, y_variable_all).statistic # getting r-value
    x_variable_all = sm.add_constant(x_variable_all)
    model_all = sm.OLS(y_variable_all, x_variable_all).fit() # running ols model
    y_pred_all = model_all.predict(x_variable_all)
    # OLS model with outliers removed
    y_variable_no_out = df_no_out[y]
    x_variable_no_out = df_no_out[x] 
    pearson_no_out = scipy.pearsonr(x_variable_no_out, y_variable_no_out).statistic # getting r-value
    x_variable_no_out = sm.add_constant(x_variable_no_out)
    model_no_out = sm.OLS(y_variable_no_out, x_variable_no_out).fit() # running ols model
    y_pred_no_out = model_no_out.predict(x_variable_no_out)
    
    # creating the plot
    axis.plot(df_all[x], df_all[y],
            linestyle = '',
            marker = 'o',
            markersize = markersize,
            color = color,
            mec = 'black',
            alpha = 0.2)
    axis.plot(df_all[x], y_pred_all,
            color = 'darkgray',
            alpha = 0.5,
            linestyle = '--',
            dashes = (5, 10), # manually changing the dash spacing of the line
           # label = f'All Values\nY = {model_all.params.iloc[1]:0.2f}x\n r: {pearson_all[0]:0.2f}'
           )
    axis.plot(df_no_out[x], df_no_out[y],
            linestyle = '',
            marker = 'o',
            markersize = markersize,
            color = color,
            mec = 'black'
            )
    axis.plot(df_no_out[x], y_pred_no_out,
            color = 'black',
            linestyle = 'solid',
            dashes = (5, 10), # manually changing the dash spacing of the line
            label = f'slope = {model_no_out.params.iloc[1]:{num_dec}f} (±{model_no_out.bse.iloc[1]:{num_dec}f})\n r: {pearson_no_out:0.2f}'
            )
    axis.legend(
        #title = 'OLS Models',
        fontsize = legend_fontsize,
        loc = loc,
        edgecolor = 'black'
        )

    axis.set_xticks(
        ticks = np.linspace(df_all[x].min(), df_all[x].max(), 5),
        labels = np.linspace(df_all[x].min(), df_all[x].max(), 5),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_yticks(
        ticks = np.linspace(df_all[y].min(), df_all[y].max(), 5),
        labels = np.linspace(df_all[y].min(), df_all[y].max(), 5),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_xlabel(f'{x} ($ppb_v$)', 
                  size = label_fontsize, 
                  fontweight = fontweight)
    axis.set_ylabel(f'{y} ($ppb_v$)',
                  size = label_fontsize, 
                  fontweight = fontweight)
    
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# defining general variables 

df = full_cmpd_dataset[['limonene', 'a-pinene']]
df.dropna(inplace = True)

data_df = df
loc = 'center right'
legend_fontsize = 20
label_fontsize = 24
tick_fontsize = 20
fontweight = 'normal'
markersize = 14

x_variable_name = 'a-pinene' # edit !
y_variable_name = 'limonene' # edit !
color = 'darkgreen'

gridspec = {'wspace' : 0.1}
fig, (ax1, ax2) = plt.subplots(
    ncols = 2,
    nrows = 1,
    sharey = True,
    sharex = True,
    figsize = (16, 7), 
    dpi = 300,
    gridspec_kw = gridspec
    )

# first plot: no outliers, all datapoints
simple_corr_plot(ax1, 
                 x=x_variable_name, 
                 y=y_variable_name,
                 df=data_df,
                 num_dec = 0.1,
                 sci_note = False,
                 show_slope = False,
                 show_r = False,
                 show_both = True,
                 color=color,
                 linecolor = 'black',
                 lw = 2,
                 linestyle = 'solid',
                 label_fontsize=label_fontsize, 
                 tick_fontsize=tick_fontsize,
                 fontweight=fontweight, 
                 markersize=markersize,
                 legend_fontsize=legend_fontsize, 
                 loc=loc,
                 x_unit = 'ppbv', 
                 y_unit='ppbv'
                 )

ax1.set_xlabel('')
ax1.set_ylabel('limonene ($ppb_v$)',
               fontweight = fontweight,
               fontsize = label_fontsize,
               labelpad = 9)

# second plot: removed high f4 days
outlier_index = only_f5_index
ols_corr_plot_outliers(ax2, 
                       x_variable_name, 
                       y_variable_name, 
                       data_df, 
                       outlier_index, 
                       color, 
                       0.1,
                       label_fontsize, 
                       tick_fontsize, 
                       fontweight, 
                       markersize, 
                       legend_fontsize, 
                       loc)
ax2.set_xlabel('')
ax2.set_ylabel('')


# setting one x-axis label
ax1.text(
    x = 0.85, 
    y = -0.15, 
    s = 'α-pinene ($ppb_v$)',
    fontsize = label_fontsize,
    fontweight = fontweight,
    transform = ax1.transAxes
    )
# setting the parameters for the plot spines
for ax in [ax1, ax2]:
   for spine in ax.spines.values():
       spine.set_color('black')
       spine.set_linewidth(1)

plt.show()

# %% SI Figure 5: Removing outlier temperatures

import statsmodels.api as sm # re-loading bc dumb new error


# re-writing previously defined OLS outlier plotting function b/c I need a different legend

# function to create correlation plots, ignoring outlier points
def ols_corr_plot_outliers (axis, x, y, df, index, 
                   color, num_dec, label_fontsize, tick_fontsize, 
                   fontweight, markersize, legend_fontsize, loc):
    
    import statsmodels.api as sm
    
    df_all = df.copy()
    df_no_out =  df_all.drop(index = index) # dropping outliers
    df_no_out.reset_index(inplace = True, drop = True) # dropping old index column
    # OLS model with all points:
    y_variable_all = df_all[y]
    x_variable_all = df_all[x]
    pearson_all = scipy.pearsonr(x_variable_all, y_variable_all).statistic # getting r-value
    x_variable_all = sm.add_constant(x_variable_all)
    model_all = sm.OLS(y_variable_all, x_variable_all).fit() # running ols model
    y_pred_all = model_all.predict(x_variable_all)
    # OLS model with outliers removed
    y_variable_no_out = df_no_out[y]
    x_variable_no_out = df_no_out[x] 
    pearson_no_out = scipy.pearsonr(x_variable_no_out, y_variable_no_out).statistic # getting r-value
    x_variable_no_out = sm.add_constant(x_variable_no_out)
    model_no_out = sm.OLS(y_variable_no_out, x_variable_no_out).fit() # running ols model
    y_pred_no_out = model_no_out.predict(x_variable_no_out)
    
    # creating the plot
    axis.plot(df_all[x], df_all[y],
            linestyle = '',
            marker = 'o',
            markersize = markersize,
            color = color,
            mec = 'black',
            alpha = 0.2)
    axis.plot(df_all[x], y_pred_all,
            color = color,
            alpha = 0.3,
            linestyle = '--',
            dashes = (5, 10), # manually changing the dash spacing of the line
           # label = f'All Values\nY = {model_all.params.iloc[1]:0.2f}x\n r: {pearson_all[0]:0.2f}'
           )
    axis.plot(df_no_out[x], df_no_out[y],
            linestyle = '',
            marker = 'o',
            markersize = markersize,
            color = color,
            mec = 'black'
            )
    axis.plot(df_no_out[x], y_pred_no_out,
            color = color,
            linestyle = 'solid',
        #    dashes = (5, 10), # manually changing the dash spacing of the line
            label = f'slope = {model_no_out.params.iloc[1]:{num_dec}f} (±{model_no_out.bse.iloc[1]:{num_dec}f})\n r: {pearson_no_out:0.2f}'
            )
    axis.legend(
        #title = 'OLS Models',
        fontsize = legend_fontsize,
        loc = loc,
        edgecolor = 'black'
        )

    axis.set_xticks(
        ticks = np.linspace(df_all[x].min(), df_all[x].max(), 5),
        labels = np.linspace(df_all[x].min(), df_all[x].max(), 5),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_yticks(
        ticks = np.linspace(df_all[y].min(), df_all[y].max(), 5),
        labels = np.linspace(df_all[y].min(), df_all[y].max(), 5),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_xlabel(f'{x} ($ppb_v$)', 
                  size = label_fontsize, 
                  fontweight = fontweight)
    axis.set_ylabel(f'{y} ($ppb_v$)',
                  size = label_fontsize, 
                  fontweight = fontweight)
    
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# defining general variables 

df_residual = scaled_residual_ts.copy()
df_residual.drop(df_residual.tail(1).index,inplace = True) # dropping Q/Qexp row

d = {
     'a-pinene' : full_cmpd_dataset['a-pinene'],
     'limonene' : full_cmpd_dataset['limonene'],
     'Median Temperature' : meta_data['median_temp_c'],
     'Average Temperature' : meta_data['avg_temp_c'],
     'date_time' : full_cmpd_dataset['start_time'],
     'hour' : full_cmpd_dataset['hour'],
     'pine_res' : df_residual['a-pinene'],
     'lim_res' : df_residual['limonene'],
     }


df = pd.DataFrame(data = d)
df = df.replace(-9999, np.nan)
df.dropna(inplace = True, ignore_index = True)

df_copy = df.copy()

df.set_index('date_time', inplace = True)
df_am = df.between_time('05:00', '12:00')
df_pm = df.between_time('12:01', '23:00')


for df in [df_am, df_pm, df]:
    df.reset_index(inplace = True)
    
outlier_index = [1, 8, 0]

label_fontsize = 14
legend_fontsize = 12
tick_fontsize = 12
markersize = 10

fig, ((ax2, ax1), (ax3, ax4)) = plt.subplots(
    ncols = 2,
    nrows = 2,
    sharex = True,
    figsize = (12, 12), 
    dpi = 300)

# ax2 = top left plot, lim vs pine vs temp
x_variable_name = 'Average Temperature'
y_variable_name = 'limonene'
data_df = df_am.copy()
outlier_index = outlier_index
color = '#C0B141'
ols_corr_plot_outliers(
    ax2,
    x_variable_name, 
    y_variable_name,
    data_df,
    outlier_index,
    color,
    0.4,
    label_fontsize, 
    tick_fontsize, 
    fontweight, 
    markersize, 
    legend_fontsize, 
    'upper right'
    )
y_variable_name = 'a-pinene'
color = '#9B287B'
ols_corr_plot_outliers(
    ax2,
    x_variable_name, 
    y_variable_name,
    data_df,
    outlier_index,
    color,
    0.3,
    label_fontsize, 
    tick_fontsize, 
    fontweight, 
    markersize, 
    legend_fontsize, 
    'upper right'
    )
ax2.set_ylim([-0.003, df['limonene'].max() + 0.005])
ax2.set_yticks(
    ticks = np.linspace(0, df['limonene'].max(), 5),
    labels = np.linspace(0, df['limonene'].max(), 5)
    )
ax2.legend(
    loc = 'upper right',
    ncols = 1,
    frameon = False,
 #   bbox_to_anchor = (0, 0.93),
    fontsize = 14
    )
ax2.set_ylabel(
    'Mixing Ratio ($ppb_v$)',
    fontsize = 14
    )


# ax1 = top right plot, pine vs temp vs tod 
y_variable_name = 'a-pinene'
data_df = df_am
outlier_index = outlier_index
color = 'orange'
ax = ax1
ols_corr_plot_outliers(
    ax,
    x_variable_name, 
    y_variable_name,
    data_df,
    outlier_index,
    color,
    0.3,
    label_fontsize, 
    tick_fontsize, 
    fontweight, 
    markersize, 
    legend_fontsize, 
    'upper right'
    )
data_df = df_pm
color = 'darkblue'
ols_corr_plot_outliers(
    ax,
    x_variable_name, 
    y_variable_name,
    data_df,
    outlier_index,
    color,
    0.3,
    label_fontsize, 
    tick_fontsize, 
    fontweight, 
    markersize, 
    legend_fontsize, 
    'upper right'
    )
ax1.set_ylim([-0.003, df['a-pinene'].max()+.01])
ax1.set_yticks(
    ticks = np.linspace(0, df['a-pinene'].max() + 0.01, 5),
    labels = np.linspace(0, df['a-pinene'].max() + 0.01, 5)
    )
ax1.set_xticks(
    ticks = np.linspace(df['Average Temperature'].min(), df['Average Temperature'].max(), 5),
    labels = np.linspace(df['Average Temperature'].min(), df['Average Temperature'].max(), 5)
    )

ax1.legend(
    loc = 'upper left',
    ncols = 1,
    frameon = False,
   # bbox_to_anchor = (0.45, 0.88),
    fontsize = 14
    )

# ax3 = bottom left, lim vs temp vs tod 
y_variable_name = 'limonene'
data_df = df_am
outlier_index = outlier_index
color = 'orange'
ax = ax3
ols_corr_plot_outliers(
    ax,
    x_variable_name, 
    y_variable_name,
    data_df,
    outlier_index,
    color,
    0.4,
    label_fontsize, 
    tick_fontsize, 
    fontweight, 
    markersize, 
    legend_fontsize, 
    'upper right'
    )
data_df = df_pm
color = 'darkblue'
ols_corr_plot_outliers(
    ax,
    x_variable_name, 
    y_variable_name,
    data_df,
    outlier_index,
    color,
    0.4,
    label_fontsize, 
    tick_fontsize, 
    fontweight, 
    markersize, 
    legend_fontsize, 
    'upper right'
    )
ax3.set_ylim([-0.003, df['limonene'].max() + 0.005])
ax3.set_yticks(
    ticks = np.linspace(0, df['limonene'].max(), 5),
    labels = np.linspace(0, df['limonene'].max(), 5)
    )
ax3.legend(
    loc = 'upper left',
    ncols = 1,
    frameon = False,
   # bbox_to_anchor = (0.45, 0.88),
    fontsize = 14
    )


# ax4 = bottom left, scaled residuals vs temp
y_variable_name = 'pine_res'
data_df = df_am
outlier_index = outlier_index
color = '#9B287B'
ax = ax4
ols_corr_plot_outliers(
    ax,
    x_variable_name, 
    y_variable_name,
    data_df,
    outlier_index,
    color,
    0.2,
    label_fontsize, 
    tick_fontsize, 
    fontweight, 
    markersize, 
    legend_fontsize, 
    'upper right'
    )
y_variable_name = 'lim_res'
color = '#C0B141'
ols_corr_plot_outliers(
    ax,
    x_variable_name, 
    y_variable_name,
    data_df,
    outlier_index,
    color,
    0.2,
    label_fontsize, 
    tick_fontsize, 
    fontweight, 
    markersize, 
    legend_fontsize, 
    'upper right'
    )
ax4.set_ylim([-1, 2])
ax4.set_yticks(
    ticks = np.linspace(-1, 2, 5),
    labels = np.linspace(-1, 2, 5)
    )
ax4.set_ylabel(
    'Scaled Factor Residual',
    fontsize = label_fontsize
    )
ax4.legend(
    loc = 'upper left',
    ncols = 1,
    frameon = False,
   # bbox_to_anchor = (0.45, 0.88),
    fontsize = 14
    )


# loops
for ax in [ax3, ax4]:
    ax.set_xticks(
        ticks = np.linspace(df['Average Temperature'].min()-5, df['Average Temperature'].max()+5, 5),
        labels = np.linspace(df['Average Temperature'].min()-5, df['Average Temperature'].max()+5, 5)
        )
for ax in [ax1, ax2]:
    ax.set_xlabel('')
for ax in [ax1, ax3]:    
    ax.text(
            x = 0.15,
            y = 1.03,
            s = '06:00 - 12:00',
            fontsize = 14,
            color = 'orange',
            fontweight = 'bold',
            transform = ax.transAxes
            )
    ax1.text(
            x = 0.55,
            y = 1.03,
            s = '12:00 - 18:00',
            fontsize = 14,
            color = 'darkblue',
            fontweight = 'bold',
            transform = ax.transAxes
            )
for ax in [ax2, ax4]:
    ax.text(
            x = 0.15,
            y = 1.03,
            s = 'limonene',
            color = '#C0B141',
            fontsize = 14,
            fontweight = 'bold',
            transform = ax.transAxes
            )
    ax.text(
            x = 0.55,
            y = 1.03,
            s = 'a-pinene',
            color = '#9B287B',
            fontsize = 14,
            fontweight = 'bold',
            transform = ax.transAxes
            )

for ax in [ax1, ax2, ax3, ax4]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
    ax.set_xlim([df['Average Temperature'].min() - 0.5, df['Average Temperature'].max() + 0.5])
for i, ax, in enumerate([ax2, ax1, ax3, ax4]):
    ax.text(
        x = 0.0, 
        y = 1.03,
        s = chr(65 + i),
        fontsize = label_fontsize,
        transform = ax.transAxes,
        fontweight = 'bold'
        )

plt.show()
plt.close()




# %% Figure 2: Monoterpene distribution

df = pd.DataFrame()
df = full_cmpd_dataset[['limonene', 'a-pinene']].copy()
df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['hour'] = full_cmpd_dataset['hour'].copy()
df['lim_pine'] = df['limonene']/ df['a-pinene']
df['average_temp'] = meta_data['avg_temp_c']
df['toluene'] = full_cmpd_dataset['toluene']
df.set_index('date_time', inplace = True)


df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x >= 6
                                   else 
                                   'am2' if x <= 12 and x >= 10
                                   else
                                   'pm1' if x <= 16 and x > 13
                                   else 
                                   'pm2'
                                   )

# grouping by hour for diel plot
grouped = df.groupby('timeblock')['lim_pine'].describe()

# splitting am and pm for OLS
df_copy = df.copy()
df_copy.dropna(inplace = True)
df_am = df_copy.between_time('05:00', '12:00')
df_pm = df_copy.between_time('12:00', '23:00')

# defining general variables 
loc = 'lower right'
legend_fontsize = 11
label_fontsize = 14
tick_fontsize = 12
fontweight = 'normal'
markersize = 10
lw = 3

pm_color = '#083D77' # pm color!
am_color = '#B76D68' # am color!

gridspec = {'hspace' : 0.5, 'wspace' : 0.3}
fig, ((ax3, ax4), (ax1, ax2)) = plt.subplots(
    ncols = 2,
    nrows = 2,
    figsize = (10.5, 6), 
    dpi = 300,
    gridspec_kw = gridspec
    )


# ax3 = diel plot
ax3.plot(grouped.index, grouped['50%'], label = 'Median',
         color = 'black',
         marker = 'o', 
         markersize = markersize)
ax3.fill_between(grouped.index, grouped['25%'], grouped['75%'],
                alpha = 0.2, color = 'gray', label = 'IQR')
ax3.set_xlabel('Time of Day',
               fontsize = label_fontsize,
               )
ax3.set_ylabel('limonene : α-pinene',
               fontsize = label_fontsize,
               labelpad = 15
               )
# adding lines at urban-rural cutoffs
ax3.hlines(y = 0.21, 
           xmin = grouped.index.min(), 
           xmax = grouped.index.max(),
           linewidth = 1,
           color = 'darkgray',
           linestyle = '--')
ax3.hlines(y = 0.68, 
           xmin = grouped.index.min(), 
           xmax = grouped.index.max(),
           linewidth = 1,
           color = 'darkgray',
           linestyle = '--')
# adding ratio cut-off boxes
box_start = -0.5
box_end = 4 + 0.5
# forest vs urban boxes and labels
box_color = '#7A9B76'

ax3.axhspan(
    ymin = 0.68,
    ymax = 1.7,
    color = box_color,
    alpha = 0.25,
    hatch = '/'
    )
ax3.axhspan(
    ymin = 0,
    ymax = 0.21,
    color = box_color,
    alpha = 0.6,
    hatch = '/'
    )
ax3.text( 
    x = 1.8, 
    y = 1.2, 
    s = 'Urban Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'normal'
    )
ax3.text( 
    x = 1.8, 
    y = 0.05, 
    s = 'Forested Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'normal'
    )
#ax3.set_xlim([grouped.index.min() - 0.5, grouped.index.max() + 0.5])
ax3.set_xticks(
    ticks = range(len(grouped.index)),
    labels = ['Morning', 'Mid-Morning', 'Afternoon', 'Early Evening'],
    fontsize = tick_fontsize
    )
ax3.set_ylim([0, 2])
ax3.set_yticks(
    ticks = np.linspace(0, 2, 5),
    labels = np.linspace(0, 2, 5),
    fontsize = tick_fontsize
    )
ax3.legend(fontsize = legend_fontsize,
           loc = 'upper left',
           bbox_to_anchor = (0.1, 1.01))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

#ax4 = OLS plot
x_variable_name = 'a-pinene' # edit !
y_variable_name = 'limonene' # edit !
simple_corr_plot(ax4, 
                 x=x_variable_name, 
                 y=y_variable_name,
                 df=df_pm,
                 num_dec = 0.2,
                 sci_note = False,
                 show_slope = False,
                 show_r = False,
                 show_both = True,
                 color=pm_color,
                 linecolor = pm_color,
                 lw = lw,
                 linestyle = 'solid',
                 label_fontsize=label_fontsize, 
                 tick_fontsize=tick_fontsize,
                 fontweight=fontweight, 
                 markersize=markersize,
                 legend_fontsize=legend_fontsize, 
                 loc=loc,
                 x_unit = 'ppbv', 
                 y_unit='ppbv'
                 )
simple_corr_plot(ax4, 
                 x=x_variable_name, 
                 y=y_variable_name,
                 df=df_am,
                 num_dec = 0.1, # formatting for decimals
                 sci_note = False,
                 show_slope = False,
                 show_r = False,
                 show_both = True,
                 color=am_color,
                 linecolor = am_color,
                 lw = lw,
                 linestyle = 'solid',
                 label_fontsize=label_fontsize, 
                 tick_fontsize=tick_fontsize,
                 fontweight=fontweight, 
                 markersize=markersize,
                 legend_fontsize=legend_fontsize, 
                 loc=loc,
                 x_unit = 'ppbv', 
                 y_unit='ppbv'
                 )
ax4.set_xlabel('α-pinene ($ppb_v$)',
               fontweight = fontweight,
               fontsize = label_fontsize,
               )
ax4.set_xlim([-0.002, df['a-pinene'].max() + 0.01])
ax4.set_xticks(
    ticks = np.linspace(0, df['a-pinene'].max() + 0.01, 5),
    labels = np.linspace(0, df['a-pinene'].max() + 0.01, 5),
    fontsize = tick_fontsize
    )
ax4.legend(
    loc = 'upper left',
    fontsize = legend_fontsize,
    ncols = 1,
    frameon = False,
    reverse = False,
    bbox_to_anchor = (0.0, 0.93),
 #   columnspacing = 0.4
    labelspacing = 1
    )


# ax1 = limonene vs toluene
simple_corr_plot(
    axis = ax1,
    x = 'toluene',
    y = 'limonene',
    df = df_pm,
    num_dec = 0.3,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = pm_color,
    linecolor = pm_color,
    lw = lw,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = fontweight,
    markersize = markersize,
    legend_fontsize = legend_fontsize,
    loc = 'upper right',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )
simple_corr_plot(
    axis = ax1,
    x = 'toluene',
    y = 'limonene',
    df = df_am,
    num_dec = 0.2,
    sci_note = False,
    show_slope = True,
    show_r = False,
    show_both = True,
    color = am_color,
    linecolor = am_color,
    lw = lw,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = fontweight,
    markersize = markersize,
    legend_fontsize = legend_fontsize,
    loc = 'upper left',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )
ax1.legend(
    loc = 'upper left',
    fontsize = legend_fontsize,
    ncols = 1,
    frameon = False,
    reverse = False,
    bbox_to_anchor = (0.0, 0.93),
    #   columnspacing = 0.4
    labelspacing = 1
    )

# ax 2 = apinene vs toluene
simple_corr_plot(
    axis = ax2,
    x = 'toluene',
    y = 'a-pinene',
    df = df_pm,
    num_dec = 0.3,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = pm_color,
    linecolor = pm_color,
    lw = lw,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = fontweight,
    markersize = markersize,
    legend_fontsize = legend_fontsize,
    loc = 'upper right',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )
simple_corr_plot(
    axis = ax2,
    x = 'toluene',
    y = 'a-pinene',
    df = df_am,
    num_dec = 0.3,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = am_color,
    linecolor = am_color,
    lw = lw,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = fontweight,
    markersize = markersize,
    legend_fontsize = legend_fontsize,
    loc = 'upper left',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )

ax2.set_ylabel('α-pinene ($ppb_v$)',
               fontweight = fontweight,
               fontsize = label_fontsize,
               )
ax2.set_ylim([-0.003, df['a-pinene'].max() + 0.01])
ax2.set_yticks(
    ticks = np.linspace(0, df['a-pinene'].max() + 0.01, 5),
    labels = np.linspace(0, df['a-pinene'].max() + 0.01, 5),
    fontsize = tick_fontsize
    )
ax2.legend(
    loc = 'upper left',
    fontsize = legend_fontsize,
    ncols = 1,
    frameon = False,
    reverse = False,
    bbox_to_anchor = (0.0, 1),
    #   columnspacing = 0.4
    labelspacing = 1
    )
# loops for consistency
for ax in [ax4, ax1]:
    ax.set_ylabel('limonene ($ppb_v$)',
                   fontweight = fontweight,
                   fontsize = label_fontsize,
                   )
    ax.set_ylim([-0.005, df['limonene'].max() + 0.01])
    ax.set_yticks(
        ticks = np.linspace(0, df['limonene'].max() + 0.01, 5),
        labels = np.linspace(0, df['limonene'].max() + 0.01, 5),
        fontsize = tick_fontsize
        )
for ax in [ax1, ax2]:
    ax.set_xlabel('toluene ($ppb_v$)',
                   fontweight = fontweight,
                   fontsize = label_fontsize,
                   )
    ax.set_xlim([-0.1, df['toluene'].max() + 0.1])
    ax.set_xticks(
        ticks = np.linspace(0, df['toluene'].max() + 0.1, 5),
        labels = np.linspace(0, df['toluene'].max() + 0.1, 5),
        fontsize = tick_fontsize
        )
for ax in [ax4, ax1, ax2]:
    ax.text(
        x = 0.45,
        y = 1.02,
        s = '12:00 - 18:00',
        fontsize = tick_fontsize,
        fontweight = 'bold',
        color = '#052241',
        transform = ax.transAxes, # relative to axis, not data
        )
    ax.text(
        x = 0.05,
        y = 1.02,
        s = '06:00 - 12:00',
        fontsize = tick_fontsize,
        color = '#580C1F',
        fontweight = 'bold',
        transform = ax.transAxes, # relative to axis, not data,
        )
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
for ax, t in zip([ax3, ax4, ax1, ax2], ['A', 'B', 'C', 'D']):
    ax.text(
        x = -0.05,
        y = 1.1,
        s = t,
        fontsize = 14,
        color = 'black',
        fontweight = 'bold',
        transform = ax.transAxes, # relative to axis, not data,
        )
    
plt.show()

# %% SI Figure: Camphor vs a-Pine vs camphor vs lim 

# re-writing previously defined OLS outlier plotting function b/c I need a different legend
# function to create correlation plots, ignoring outlier points
def ols_corr_plot_outliers (axis, x, y, df, index, 
                   color, label_fontsize, tick_fontsize, 
                   fontweight, markersize, num_dec, sci_note, 
                   legend_fontsize, loc):
    
    import statsmodels.api as sm
    
    df_all = df.copy()
    df_no_out =  df_all.drop(index = index) # dropping outliers
    df_no_out.reset_index(inplace = True, drop = True) # dropping old index column
    # OLS model with all points:
    y_variable_all = df_all[y]
    x_variable_all = df_all[x]
    pearson_all = scipy.pearsonr(x_variable_all, y_variable_all).statistic # getting r-value
    x_variable_all = sm.add_constant(x_variable_all)
    model_all = sm.OLS(y_variable_all, x_variable_all).fit() # running ols model
    y_pred_all = model_all.predict(x_variable_all)
    # OLS model with outliers removed
    y_variable_no_out = df_no_out[y]
    x_variable_no_out = df_no_out[x] 
    pearson_no_out = scipy.pearsonr(x_variable_no_out, y_variable_no_out).statistic # getting r-value
    x_variable_no_out = sm.add_constant(x_variable_no_out)
    model_no_out = sm.OLS(y_variable_no_out, x_variable_no_out).fit() # running ols model
    y_pred_no_out = model_no_out.predict(x_variable_no_out)
    
    if sci_note == True:
        note = 'E'
    else:
        note = 'f'
    
    # creating the plot
    axis.plot(df_all[x], df_all[y],
            linestyle = '',
            marker = 'o',
            markersize = markersize,
            color = color,
            mec = 'black',
            alpha = 0.2)
    axis.plot(df_all[x], y_pred_all,
            color = 'darkgray',
            alpha = 0.5,
            linestyle = '--',
            dashes = (5, 10), # manually changing the dash spacing of the line
           # label = f'All Values\nY = {model_all.params.iloc[1]:0.2f}x\n r: {pearson_all[0]:0.2f}'
           )
    axis.plot(df_no_out[x], df_no_out[y],
            linestyle = '',
            marker = 'o',
            markersize = markersize,
            color = color,
            mec = 'black'
            )
    axis.plot(df_no_out[x], y_pred_no_out,
            color = 'black',
            linestyle = 'solid',
         #   dashes = (5, 10), # manually changing the dash spacing of the line
            label = f'slope = {model_no_out.params.iloc[1]:{num_dec}{note}} (±{model_no_out.bse.iloc[1]:{num_dec}{note}})\nr: {pearson_no_out:.2f}'
            )
    axis.legend(
        #title = 'OLS Models',
        fontsize = legend_fontsize,
        loc = loc,
        edgecolor = 'black'
        )

    axis.set_xticks(
        ticks = np.linspace(df_all[x].min(), df_all[x].max(), 5),
        labels = np.linspace(df_all[x].min(), df_all[x].max(), 5),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_yticks(
        ticks = np.linspace(df_all[y].min(), df_all[y].max(), 5),
        labels = np.linspace(df_all[y].min(), df_all[y].max(), 5),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_xlabel(f'{x} ($ppb_v$)', 
                  size = label_fontsize, 
                  fontweight = fontweight)
    axis.set_ylabel(f'{y} ($ppb_v$)',
                  size = label_fontsize, 
                  fontweight = fontweight)
    
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

df = pd.DataFrame()

df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['lim_pine'] = full_cmpd_dataset['limonene'].copy()/ full_cmpd_dataset['a-pinene'].copy()
df['a-pinene'] = full_cmpd_dataset['a-pinene']
df['campor_pine'] = full_cmpd_dataset['camphor'].copy()/ full_cmpd_dataset['a-pinene'].copy()
df['camphor'] = full_cmpd_dataset['camphor']
df['limonene'] = full_cmpd_dataset['limonene']
df['f1'] = factor_ts['Factor 1']
df['f2'] = factor_ts['Factor 2']
df['f3'] = factor_ts['Factor 3']
df['f4'] = factor_ts['Factor 4']
# can't have NaNs for a corr plot
df.dropna(inplace = True)
df.set_index('date_time', inplace = True)

# defining general variables 
data_df = df
loc = 'lower right'
legend_fontsize = 20
label_fontsize = 24
tick_fontsize = 20
fontweight = 'normal'
markersize = 14

gridspec = {'hspace' : 0.25, 
            'wspace' : 0.12}
fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(
    ncols = 2,
    nrows = 2,
    sharey = True,
    sharex = False,
    figsize = (16, 10), 
    dpi = 300,
    gridspec_kw = gridspec
    )

# first plot: no outliers, all datapoints
y_variable_name = 'camphor' # edit !
x_variable_name = 'a-pinene' # edit !
color = 'purple'
simple_corr_plot(ax1, x=x_variable_name, 
                 y=y_variable_name, 
                 df=df,
                 num_dec = 0.2,
                 sci_note = False,
                 show_slope = False,
                 show_r = False,
                 show_both = True,
                 color=color,
                 linecolor = 'black',
                 lw = 2,
                 linestyle = 'solid',
                 label_fontsize=label_fontsize,
                 tick_fontsize=tick_fontsize,
                 fontweight=fontweight, 
                 markersize=markersize,
                 legend_fontsize=legend_fontsize, 
                 loc=loc, 
                 x_unit = '$ppb_v$', 
                 y_unit = '$ppb_v$')

y_variable_name = 'camphor' # edit !
x_variable_name = 'limonene' # edit !
color = 'darkslategrey'
simple_corr_plot(ax2,
                 x=x_variable_name, 
                 y=y_variable_name, 
                 df=df,
                 num_dec = 0.2, 
                 sci_note = False,
                 show_slope = False,
                 show_r = False,
                 show_both = True,
                 color=color,
                 linecolor = 'black',
                 lw = 2,
                 linestyle = 'solid',
                 label_fontsize=label_fontsize,
                 tick_fontsize=tick_fontsize,
                 fontweight=fontweight, 
                 markersize=markersize,
                 legend_fontsize=legend_fontsize, 
                 loc=loc, 
                 x_unit = '$ppb_v$', 
                 y_unit = '$ppb_v$')
  
# bottom plots: removed high f4 days
df = pd.DataFrame()
df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['a-pinene'] = full_cmpd_dataset['a-pinene']
df['camphor'] = full_cmpd_dataset['camphor']
df['limonene'] = full_cmpd_dataset['limonene']
# can't have NaNs for a corr plot
df.dropna(inplace = True)
# globabl variables
only_f4_index = [5, 6, 20]
outlier_index = only_f4_index
data_df = df
loc = 'lower right'
legend_fontsize = 20
label_fontsize = 24
tick_fontsize = 20
fontweight = 'normal'
markersize = 14
sci_note = False

x_variable_name = 'a-pinene' # edit !
y_variable_name = 'camphor' # edit !
color = 'plum'
num_dec = 0.1
ols_corr_plot_outliers(ax3, x_variable_name, y_variable_name, data_df, 
                       outlier_index, color, label_fontsize, tick_fontsize, 
                       fontweight, markersize, num_dec, sci_note, 
                       legend_fontsize, loc)

x_variable_name = 'limonene' # edit !
y_variable_name = 'camphor' # edit !
color = 'slategrey'
num_dec = 0.1
ols_corr_plot_outliers(ax4, x_variable_name, y_variable_name, data_df, 
                       outlier_index, color, label_fontsize, tick_fontsize, 
                       fontweight, markersize, num_dec, sci_note, 
                       legend_fontsize, loc)


# loops
for ax in [ax1, ax3]:
    ax.set_xlim([-0.003, df['a-pinene'].max() + 0.005])
ax3.set_xticks(ticks = np.linspace(0, df['a-pinene'].max() + 0.005, 4),
               labels = np.linspace(0, df['a-pinene'].max() + 0.005, 4),
               fontsize = tick_fontsize
               )
for ax in [ax2, ax4]:
    ax.set_xlim([-0.005, df['limonene'].max() + 0.005])
ax4.set_xticks(ticks = np.linspace(0, df['limonene'].max() + 0.005, 4),
               labels = np.linspace(0, df['limonene'].max() + 0.005, 4),
               fontsize = tick_fontsize
               )

for ax in [ax1, ax2]:
    ax.set(xticklabels = [], xlabel = None)
for ax in [ax3, ax4]:
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
for ax in [ax2, ax4]:
    ax.set_ylabel('')
    
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_ylim([-0.001, df['camphor'].max() + 0.002])
    ax.set_yticks(ticks = np.linspace(0, df['camphor'].max(), 4),
                   labels = np.linspace(0, df['camphor'].max(), 4),
                   fontsize = tick_fontsize
                   )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    ax.text(
        x = 0.02, 
        y = 1.05,
        s = chr(65 + i),
        fontsize = 24,
        transform = ax.transAxes,
        fontweight = 'bold'
        )
    

plt.show()

# %% SI Figure: MT % breakdown

df = percent_SppSum.copy()
df.set_index('compounds', inplace = True)
df = df.transpose()

df = df[['limonene', 'a-pinene']]

mt_colors = {
    'limonene' : '#C0B141',
    'a-pinene' : '#9B287B'
    }

fig, ax = plt.subplots(figsize = (8, 5), 
                       dpi = 300)
df.plot(ax = ax, 
        stacked = False,
        kind = 'bar', 
        subplots = False, 
        sharex = True, 
        use_index = True, 
        width = 0.95,
        ylim = [0, 100],
        fontsize = 14,
        color = [mt_colors[col] for col in df.columns],
        edgecolor = 'black', 
        linewidth = 1.5,
        rot = 360
        )
ax.set_ylabel('% Factor Contribution',
              fontsize = 16)

plt.show()




# %% SI Figure 16: Temp + Fragrances 

d = {
      'tricyclodecenyl acetate' : full_cmpd_dataset['tricyclodecenyl acetate'],
      'diphenyl ether' : full_cmpd_dataset['diphenyl ether'],
      '4-tert-butylcyclohexyl acetate 2' : full_cmpd_dataset['4-tert-butylcyclohexyl acetate 2'],
      '4-tert-butylcyclohexyl acetate 1' : full_cmpd_dataset['4-tert-butylcyclohexyl acetate 1'],
      'benzyl acetate' : full_cmpd_dataset['benzyl acetate'],
      'toluene' : full_cmpd_dataset['toluene'],
      'Average Temperature' : meta_data['avg_temp_c'],
      'date_time' : full_cmpd_dataset['start_time'],
      'hour' : full_cmpd_dataset['hour']
      }
df = pd.DataFrame(data = d)
df = df.replace(-9999, np.nan)
df.dropna(inplace = True, ignore_index = True)

df.set_index('date_time', inplace = True)
df_am = df.between_time('05:00', '12:00')
df_pm = df.between_time('12:01', '23:00')



# setting the ind sizes of the two subplots + the white space inbetween
gridspec = {'hspace' : 0.12, 
            'wspace' : 0.25}
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    ncols = 2,
    nrows = 2,
    sharex = True,
    figsize = (12, 12), 
    dpi = 300,
    gridspec_kw = gridspec)

cmpd_list = ['benzyl acetate', 'diphenyl ether', 'tricyclodecenyl acetate', 'toluene']
ax_list = [ax1, ax2, ax3, ax4]
am_dec_list = [0.2, 0.3, 0.3, 0.1]

for cmpd, ax, f in zip(cmpd_list, ax_list, am_dec_list):
    x = 'Average Temperature'
    y = cmpd
    simple_corr_plot( # am plots
        axis = ax,
        x = x,
        y = y,
        df = df_am,
        num_dec = f, 
        sci_note = False,
        show_slope = False,
        show_r = False,
        show_both = True,
        color = 'orange',
        linecolor = 'orange',
        lw = 2,
        linestyle = 'solid',
        label_fontsize = 14,
        tick_fontsize = 12,
        fontweight = 'normal',
        markersize = 8,
        legend_fontsize = 12,
        loc = 'upper right',
        x_unit = 'C',
        y_unit = '$ppb_v$'
        )
    simple_corr_plot( # adding pm plots on top
        axis = ax,
        x = x,
        y = y,
        df = df_pm,
        num_dec = 0.2, 
        sci_note = False,
        show_slope = False,
        show_r = False,
        show_both = True,
        color = 'darkblue',
        linecolor = 'darkblue',
        lw = 2,
        linestyle = 'solid',
        label_fontsize = 14,
        tick_fontsize = 12,
        fontweight = 'normal',
        markersize = 8,
        legend_fontsize = 12,
        loc = 'upper left',
        x_unit = 'C',
        y_unit = '$ppb_v$'
        )

    ax.set_xlim([df['Average Temperature'].min() - 0.5, df['Average Temperature'].max() + 0.5])
    ax.set_xticks(
        ticks = np.linspace(df['Average Temperature'].min(), df['Average Temperature'].max(), 5),
        labels = np.linspace(df['Average Temperature'].min(), df['Average Temperature'].max(), 5)
        )
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
    ax.set_xlabel('')

    ax.set_ylim([0 - 0.001, df[cmpd].max() + 0.05])
    ax.set_yticks(
        ticks = np.linspace(0, df[cmpd].max(), 5),
        labels = np.linspace(0, df[cmpd].max(), 5)
        )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    ax.text(
        x = -0.05, 
        y = 1.02,
        s = chr(65 + i),
        fontsize = 14,
        transform = ax.transAxes,
        fontweight = 'bold'
        )

for ax in [ax1, ax2]:
    ax.text(
            x = 0.08,
            y = 1.02,
            s = '06:00 - 12:00',
            fontsize = 14,
            fontweight = 'bold',
            color = 'orange',
            transform = ax.transAxes, # relative to axis, not data
            )
    ax.text(
            x = 0.51,
            y = 1.02,
            s = '12:00 - 18:00',
            fontsize = 14,
            color = 'darkblue',
            fontweight = 'bold',
            transform = ax.transAxes, # relative to axis, not data
            )

for ax in [ax3, ax4]:
    ax.set_xlabel('Average Temperature (C)',
                  fontsize = 14)
  
# manually adjusting ax4
ax4.set_ylim([0 - 0.1, df[cmpd].max() + 0.1])
ax4.set_yticks(
    ticks = np.linspace(0, df[cmpd].max(), 5),
    labels = np.linspace(0, df[cmpd].max(), 5)
    )
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.show()
plt.close()

# %% SI Figure: Correlation Plot w/ ground data

meta_data_copy = meta_data.copy() # no overwriting !
#cmpds_ppb_copy = ground_cmpds_ppb.copy()

starttime = meta_data_copy['start_time']

meta_data_copy.drop(columns = ['start_time', 
                               'avg_rain_in/s',
                               'median_temp_c',
                               'max_temp_c',
                               'avg_RH_pct',
                               'avg_press_inHg',
                               'median_WindDir',
                               'avg_no3_ug_m3',
                               'avg_so4_ug_m3',
                               'avg_nh4_ug_m3',
                               'avg_total_aerosol_ug_m3'
                               ], inplace = True)

meta_data_copy.set_index('hour', inplace = True)
#cmpds_ppb_copy.set_index('hour', inplace = True)

#meta_data_concat = pd.concat([meta_data_copy, cmpds_ppb_copy], axis = 1)

df = factor_ts.copy()  # no overwriting! 

df.drop(columns = ['datetime'], inplace = True)
df.set_index('hour', inplace = True)

# adding metadata to each factor df to run the corr matrix
df = pd.concat([df, meta_data_copy], axis = 1)
df.rename(columns = {
    'avg_temp_c' : 'Average\nTemperature (C)',
    'median_SolarRad_W/m2' : 'Median Solar\nRadiation (W/$m^2$)',
    'avg_WindSpeed_m/s' : 'Average Wind\nSpeed (m/s)',
    'avg_o3_ppb' : 'Average $O_3$\n($ppb_v$)',
    'avg_no_ppb' : 'Average NO ($ppb_v$)',
    'avg_no2_ppb' : 'Average $NO_2$ ($ppb_v$)',
    'avg_nox_ppb' : 'Average $NO_x$ ($ppb_v$)',
    'avg_oa_ug_m3' : 'Average OA (μg/$m^3$)',
    'median_RH_pct' : 'Median RH (%)'
    }, inplace = True)

df = df.corr() # getting pearson coeficcients
# half matrix plot
lower_mask = np.tri(df.shape[0], df.shape[1], k = -1)
upper_mask = lower_mask.T
matrix_mask = np.ma.array(df, mask = upper_mask) # only plotting bottom half of matrix plot


fig, ax = plt.subplots(figsize = (24, 20), dpi = 300)

# need to center where 0 = white
divnorm = colors.TwoSlopeNorm(vmin = matrix_mask.min(), vcenter = 0, vmax = matrix_mask.max())

cax = ax.imshow(matrix_mask, cmap = 'coolwarm',  # plot matrix w/ chosen colormap
                  interpolation = 'nearest', # round to nearest decimal place for heatmap value
                  norm = divnorm
                  ) 
cbar = plt.colorbar(cax, shrink = 0.95, pad = 0.03)
cbar.set_label(label = 'Pearson Coefficient Correlations', 
               weight = 'normal', fontsize = 24, labelpad = 20) # default labelpad = 4
cbar.ax.tick_params(labelsize = 24)

ax.set_xticks(range(len(df.columns))) # set xticks based on the columns in the matrix
ax.set_xticklabels(df.columns, 
                 rotation = 90, fontsize = 24) # set x values based on columns and rotate for visibility
ax.set_yticks(range(len(df.columns))) # repeat for yaxis
ax.set_yticklabels(df.columns, fontsize = 24)
for (i, j), value in np.ndenumerate(df): # (coordinates) & r-value in loop in values within the mask:
  if i > j: # ensure annotated values are in the bottom half of the plot
      if float(f'{value:.2}') == 1.0 : 
          ax.text(j, i, f'{value:.1f}', ha = 'center', va = 'center', 
                  color = 'black', fontsize = 24)    
      elif float(f'{value:.2}') >= 0.3 : # only displaying r higher than 0.3
             ax.text(j, i, f'{value:.2f}', ha = 'center', va = 'center', 
                     color = 'black', fontsize = 24, fontweight = 'bold') 
      elif float(f'{value:.2}') <= - 0.3 : # adjusting the value:# will change the sigfigs
         ax.text(j, i, f'{value:.2f}', ha = 'center', va = 'center', 
                 color = 'black', fontsize = 24, fontweight = 'bold')
      else: 
         ax.text(j, i, f'{value:.2f}', ha = 'center', va = 'center', 
                 color = 'black', fontsize = 24, fontweight = 'normal')
      

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

plt.show()
plt.close()

# %% SI Figure: Correlation Plot w/ back trajectories

full_path = drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / '2025' / 'ground_data_workup' / 'meta_data' / '20250613_back_traj_full.xlsx'
#file = '20250613_back_traj_full.xlsx'

back_traj =  pd.read_excel(full_path, 
                           sheet_name = 'Full_Trajectory_withPop_240129', 
                           header = 0,
                           usecols = [
                               'date_arrival',
                               'Traj_Age_hr_c',
                               'people_per_mi2_c',
                              # 'Dist_Traveled_m',
                               'land area mi2',
                               'on_land',
                               'Rel_Humidity',
                               'Ambient_Temp',
                               'Height_m',
                               'Mixed_Layer_Depth_m',
                              # 'windspeed_m_s',
                               'Pop_ed'
                               ],
                           skiprows = lambda x: x not in [0] and x < 25
                           ) 
#back_traj['pop_x_temp'] = back_traj['Pop_ed'] * back_traj['Ambient_Temp']
# Renaming columns nicely for the plot
back_traj.rename(columns = {
                       'Traj_Age_hr_c' : 'traj_age_hr',
                   #    'Dist_Traveled_m' : 'Distance\nTraveled (m)',
                       'land area mi2' :'Land Area ($mi^2$)',
                       'Ambient_Temp' : 'Temperature (K)',
                       'Rel_Humidity' : 'Relative\nHumidity (%)',
                       'Height_m' : 'Airmass \nHeight (m)',
                       'Mixed_Layer_Depth_m' : 'Mixed Layer\nDepth (m)',
                       'Pop_ed' : 'Population (#)',
                 #      'windspeed_m_s' : 'Windspeed (m/s',
                       'people_per_mi2_c' : 'Population Density\n(people $mi^{-2}$)',
              #         'pop_x_temp' : 'Population x Ambient Temperature)'
    }, inplace = True)
back_traj.set_index('date_arrival', inplace = True)

df = factor_ts.copy()
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



# making the correlation plot
df = concat.corr() # getting pearson coeficcients
# half matrix plot
lower_mask = np.tri(df.shape[0], df.shape[1], k = -1)
upper_mask = lower_mask.T
matrix_mask = np.ma.array(df, mask = upper_mask) # only plotting bottom half of matrix plot

plt.rcParams["mathtext.default"] = 'regular'
fig, ax = plt.subplots(figsize = (24, 20), 
                       dpi = 300)

# need to center where 0 = white
divnorm = colors.TwoSlopeNorm(vmin = matrix_mask.min(), vcenter = 0, vmax = matrix_mask.max())

cax = ax.imshow(matrix_mask, cmap = 'coolwarm',  # plot matrix w/ chosen colormap
                  interpolation = 'nearest', # round to nearest decimal place for heatmap value
                  norm = divnorm
                  ) 
 
cbar = plt.colorbar(cax, shrink = 0.95)
cbar.set_label(label = 'Pearson Coefficient Correlations',
               weight = 'normal', 
               fontsize = 24,
               labelpad = 20)

cbar.ax.tick_params(labelsize = 24)
ax.set_xticks(range(len(df.columns))) # set xticks based on the columns in the matrix
ax.set_xticklabels(df.columns, 
                 rotation = 90, 
                 fontsize = 24) # set x values based on columns and rotate for visibility
ax.set_yticks(range(len(df.columns))) # repeat for yaxis
ax.set_yticklabels(df.columns, fontsize = 24)

for (i, j), value in np.ndenumerate(df): # (coordinates) & r-value in loop in values within the mask:
  if i > j: # ensure annotated values are in the bottom half of the plot
      if float(f'{value:.2}') == 1.0 : 
          ax.text(j, i, f'{value:.1f}', ha = 'center', va = 'center', 
                  color = 'black', fontsize = 24)    
      elif float(f'{value:.2}') >= 0.3 : # only displaying r higher than 0.3
             ax.text(j, i, f'{value:.2f}', ha = 'center', va = 'center', 
                     color = 'black', fontsize = 24, fontweight = 'bold') 
      elif float(f'{value:.2}') <= - 0.3 : # adjusting the value:# will change the sigfigs
         ax.text(j, i, f'{value:.2f}', ha = 'center', va = 'center', 
                 color = 'black', fontsize = 24, fontweight = 'bold')
      else: 
         ax.text(j, i, f'{value:.2f}', ha = 'center', va = 'center', 
                 color = 'black', fontsize = 24, fontweight = 'normal')
    

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

plt.show()
plt.close()

# %% Toluene Diel Plot

df = pd.DataFrame()
df['toluene'] = full_cmpd_dataset['toluene']
df['date_time'] = full_cmpd_dataset['start_time']
df['hour'] = full_cmpd_dataset['hour']
df.set_index('date_time', inplace = True)


df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x >= 6
                                   else 
                                   'am2' if x <= 12 and x >= 10
                                   else
                                   'pm1' if x <= 16 and x > 13
                                   else 
                                   'pm2'
                                   )

# grouping by hour for diel plot
grouped = df.groupby('timeblock')['toluene'].describe()


# defining general variables 
loc = 'lower right'
legend_fontsize = 11
label_fontsize = 14
tick_fontsize = 12
fontweight = 'normal'
markersize = 10
lw = 3



fig, ax3 = plt.subplots(
    ncols = 1,
    nrows = 1,
   # figsize = (10.5, 6), 
    dpi = 300,

    )


# ax3 = bottom plot
ax3.plot(grouped.index, grouped['50%'], label = 'Median',
         color = 'black',
         marker = 'o', 
         markersize = markersize)
ax3.fill_between(grouped.index, grouped['25%'], grouped['75%'],
                alpha = 0.2, color = 'gray', label = 'IQR')
ax3.set_ylim([0, grouped['75%'].max()])
ax3.set_yticks(
    ticks = np.linspace(0, grouped['75%'].max(), 5),
    labels = np.linspace(0, grouped['75%'].max(), 5),
    fontsize = tick_fontsize
    )
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


ax3.set_ylabel('toluene ($ppb_v$)',
               fontsize = label_fontsize,
               labelpad = 15
               )

#ax3.set_xlim([grouped.index.min() - 0.5, grouped.index.max() + 0.5])
ax3.set_xticks(
    ticks = range(len(grouped.index)),
    labels = ['Morning', 'Mid-Morning', 'Afternoon', 'Early Evening'],
    fontsize = tick_fontsize
    )

ax3.legend(fontsize = legend_fontsize,
           loc = 'upper right',
       #   bbox_to_anchor = (0.1, 1.01)
          )

plt.show()
plt.close()

# %% SI Figure: MT vs camphor vs toluene by TOD

d = {
      'a-pinene' : full_cmpd_dataset['a-pinene'],
      'limonene' : full_cmpd_dataset['limonene'],
      'camphor' : full_cmpd_dataset['camphor'],
      'toluene' : full_cmpd_dataset['toluene'],
      'Average Temperature' : meta_data['avg_temp_c'],
      'date_time' : full_cmpd_dataset['start_time'],
      'hour' : full_cmpd_dataset['hour']
      }
df = pd.DataFrame(data = d)
df = df.replace(-9999, np.nan)
df['limonene : a-pinene'] = df['limonene'] / df['a-pinene']
df.dropna(inplace = True, ignore_index = True)

df.set_index('date_time', inplace = True)
df_am = df.between_time('05:00', '12:00')
df_pm = df.between_time('12:01', '23:00')

# setting the ind sizes of the two subplots + the white space inbetween
gridspec = {'hspace' : 0.4, 
            'wspace' : 0.12}
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
    ncols = 2,
    nrows = 3,
    figsize = (10.5, 8), 
    dpi = 300,
    gridspec_kw = gridspec)

# ax1 = uppper left
simple_corr_plot(
    axis = ax1,
    x = 'limonene',
    y = 'camphor',
    df = df_am,
    num_dec = 0.2, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'orange',
    linecolor = 'orange',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 8,
    legend_fontsize = 12,
    loc = 'upper right',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )
# ax1 = uppper right
simple_corr_plot(
    axis = ax2,
    x = 'limonene',
    y = 'camphor',
    df = df_pm,
    num_dec = 0.1, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'darkblue',
    linecolor = 'darkblue',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 8,
    legend_fontsize = 12,
    loc = 'upper right',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )

simple_corr_plot(
    axis = ax3,
    x = 'a-pinene',
    y = 'camphor',
    df = df_am,
    num_dec = 0.1, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'orange',
    linecolor = 'orange',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 8,
    legend_fontsize = 12,
    loc = 'upper right',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )
simple_corr_plot(
    axis = ax4,
    x = 'a-pinene',
    y = 'camphor',
    df = df_pm,
    num_dec = 0.1, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'darkblue',
    linecolor = 'darkblue',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 8,
    legend_fontsize = 12,
    loc = 'upper right',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )

simple_corr_plot(
    axis = ax5,
    x = 'toluene',
    y = 'camphor',
    df = df_am,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'orange',
    linecolor = 'orange',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 8,
    legend_fontsize = 12,
    loc = 'upper right',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )
# ax1 = uppper right
simple_corr_plot(
    axis = ax6,
    x = 'toluene',
    y = 'camphor',
    df = df_pm,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'darkblue',
    linecolor = 'darkblue',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 8,
    legend_fontsize = 12,
    loc = 'upper right',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )

# ax1.set(xticklabels = [],
#         xlabel = None)
# ax2.set(xticklabels = [],
#         xlabel = None)
ax2.set(ylabel = None)
ax4.set(ylabel = None)
ax6.set(ylabel = None)

for ax in [ax2, ax4, ax6]:
    ax.set_yticklabels([''] * len(ax.get_yticks()))


for i, ax, in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
    ax.text(
        x = 0.03, 
        y = 1.05,
        s = chr(65 + i),
        fontsize = 12,
        transform = ax.transAxes,
        fontweight = 'bold'
        )

ax1.set_title(
    '06:00 - 12:00',
     fontsize = 14,
     color = 'orange'
    )
ax2.set_title(
    '12:00 - 18:00',
    fontsize = 14,
    color = 'darkblue'
    )

plt.show()

# %% SI Figure: ethylhexyl vs homosalate


d = {
     'Homosalate' : full_cmpd_dataset['homosalate'],
     'Ethylhexyl benzoate': full_cmpd_dataset['ethylhexyl benzoate'],
     'Median Relative Humidity' : meta_data['median_RH_pct'],
     'Median Temperature' : meta_data['median_temp_c'],
     'Phenoxyisopropanol' : full_cmpd_dataset['phenoxyisopropanol'],
     }
df = pd.DataFrame(data = d)
df = df.replace(-9999, np.nan)
df.dropna(inplace = True, ignore_index = True)

# removing values set to DL/2 for correlation
df['Homosalate'] = df['Homosalate'].replace(4.47618e-07, np.nan)
df['Ethylhexyl benzoate'] = df['Ethylhexyl benzoate'].replace(6.97335e-08, np.nan)

df = df.replace(-9999, np.nan)
df.dropna(inplace = True, ignore_index = True)

fig, ax = plt.subplots(
    figsize = (8, 8), 
    dpi = 300)
simple_corr_plot(axis = ax, 
                 x = 'Ethylhexyl benzoate', 
                 y = 'Homosalate', 
                 df = df, 
                 num_dec = 0.4,
                 sci_note = False,
                 show_slope = False,
                 show_r = True,
                 show_both = False,
                 color = '#C95B0C', 
                 linecolor = 'black',
                 lw = 2,
                 linestyle = 'solid',
                 label_fontsize = 16, 
                 tick_fontsize = 14, 
                 fontweight = 'normal',
                 markersize = 10, 
                 legend_fontsize = 14, 
                 loc = 'upper left', 
                 x_unit = '$ppb_v$', 
                 y_unit = '$ppb_v$')

ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.show()

# %% SI Figure 10: legend  # uncomment to print

# df = common_cmpds_ppb.copy()
# df.rename(columns = {'start_time_edt' : 'datetime'}, inplace = True)

# df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x > 6
#                                    else 
#                                    'am2' if x <= 12 and x > 10
#                                    else
#                                    'pm1' if x <= 16 and x > 13
#                                    else 
#                                    'pm2'
#                                    )

# start_time = df['start_time']
# df.drop(columns = ['start_time', 'hour'], inplace = True)
     
# am1 = df[df['timeblock'] == 'am1'] # filtering by timeblock
# am2 = df[df['timeblock'] == 'am2']
# pm1 = df[df['timeblock'] == 'pm1']
# pm2 = df[df['timeblock'] == 'pm2']


# block_dfs = [am1, am2, pm1, pm2]
# block_strings = ['am1', 'am2', 'pm1', 'pm2']

# block_dfs = [df.median(numeric_only = True) for df in block_dfs]  # getting median values
# block_dfs = [pd.DataFrame(df) for df in block_dfs]
# for df, value in zip(block_dfs, block_strings):
#     df.rename(columns = {0 : value}, inplace = True)

# concat = pd.concat(block_dfs, axis = 1)
# df = concat.transpose()


# fig, ax = plt.subplots(figsize = (12, 12), dpi = 200)
# bars = df.plot(ax = ax, 
#                stacked = True,
#                  kind = 'bar', 
#                  subplots = False, 
#                  sharex = True, 
#                  use_index = True, 
#          # ylim = [0, 105],
#           fontsize = 14,
#           color = [cmpd_colors_dict[col] for col in df.columns],
#           edgecolor = 'white', 
#           linewidth = 0.75,
#           rot = 360
#           )
# for col_idx, container in enumerate(ax.containers):
#     compound = df.columns[col_idx]
#     hatch_style = hatch_dict.get(compound, '')

#     for bar in container:
#         bar.set_hatch(hatch_style)

# legend = plt.legend(ncols = 1, fontsize = 'large', loc = 'best',
#            bbox_to_anchor = (1, 1)
#            )
# # coping legend into its own figure
# legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
# # Convert pixel coordinates to inches
# legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())

# # Create a separate figure for the legend
# legend_fig, legend_ax = plt.subplots(figsize = (legend_bbox.width, legend_bbox.height))

# # Recreate the legend on the separate figure/axis
# legend_squared = legend_ax.legend(
#     *ax.get_legend_handles_labels(), 
#     loc = 'center left',
#     bbox_to_anchor = (-1, 0, 1, 1),
#     bbox_transform = legend_fig.transFigure,
#     frameon = False,
#     fancybox = None,
#     shadow = False,
#     ncol = 1,
#     mode = 'expand',
#     fontsize = 20,
#     reverse = True
    
# )

# # Remove everything else from the legend's figure
# legend_ax.axis('off')
# %%

fig_sum = 0.075528 + 1.015054 +  0.000128 + 2.024672 + 0.432694 + 2.618399 + 3.545046 + 0.896958 + 0.050024 + 11.256619 + 2.133390
