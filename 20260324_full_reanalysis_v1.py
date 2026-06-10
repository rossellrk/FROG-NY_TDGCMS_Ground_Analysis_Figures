# -*- coding: utf-8 -*-
"""
Re-analysis of the ground data (re-analyazed via TDGMS Macro) and new PMF 5-factor solution.
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
cmpd_refs = cmpd_refs[['cmpd_name', 'CAS', 'Molecular Weight (g/mol)', 'Molecular Formula', 'MIR_best',
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
cmpd_refs['MIR_best'] = pd.to_numeric(cmpd_refs['MIR_best'], errors = 'coerce')


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

# reading in ACSM OA Data
file = drive_folder / 'CSU' / 'Research' / 'Research Projects' / 'Field Campaigns' / 'Frog 2023' / '2025' / 'ground_data_workup' / 'meta_data' / 'FROG_ACSM_NR-PM1.csv'
col_names = ['datetime_local', 'total_ug_m3', 'OA_ug_m3', 'no3_ug_m3', 'nh4_ug_m3', 'so4_ug_m3', 'chl_ug_m3']
oa_data = pd.read_csv(file, skiprows = [0], 
                      names = col_names, 
                      parse_dates = ['datetime_local'])


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
    'Total OH Reactivity' : 'dashed'
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
    'sabinene' : '#e04471C'
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


# %% Factors by total compund mass

df = profile_conc.copy()
df = df.sum(axis = 0, numeric_only = True)  # summing each factor total
df = pd.DataFrame(data = df)

fig, ax = plt.subplots(figsize = (5, 8), dpi = 300)
bar_plot = df.plot(ax = ax, kind = 'bar', sharex = True,
                #   title = f'{len(df)}' + ' Factor Solution',
                   ylabel = 'Total Factor Mixing Ratio ($ppb_v$)',
                   rot = 45,
                   edgecolor = 'gray', 
                   color = 'cadetblue',
                   linewidth = 0.75,
                   )

units = '$ppb_v$'
add_bar_totals(range(len(df.index)), 0.8, df[0], units, 8, 'normal')

ax.get_legend().remove()

plt.show() 

# %% T:B Ratios for Factor Solutions

df = profile_conc.copy()

df.set_index('compounds', inplace = True)
df = df.loc[['benzene', 'toluene']]
df.replace(0, np.nan, inplace = True)

df.loc['t_b'] = df.loc['toluene'] / df.loc['benzene']


row = df.loc['t_b']
fig, ax = plt.subplots(figsize = (5, 8), dpi = 300)
bar_plot = row.plot(ax = ax, kind = 'bar', 
                  #title = f'{len(df.columns)}' + ' Factor Solution',
                   ylabel = 'Toluene : Benzene',
                   rot = 45,
                   color = 'lightgray',
                   edgecolor = 'black', 
                   linewidth = 0.75,
                   )
ax.set

units = ''
add_bar_totals(range(len(row)), 150, row, units, 12, 'normal')
plt.show() 


# %% Bar Charts of Compound Distributions in Factors

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

n_rows = len(df)
n_cols = len(df.columns)

# labeling the top 5 compounds making up each factor:
for bar_idx, (index, row) in enumerate(df.iterrows()):
    top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(6).index.tolist()

    for col_idx, col in enumerate(df.columns):
        # Find the correct bar (patch)
        bar = ax.patches[col_idx * n_rows + bar_idx]
        
        if col in top_cols:
            color = cmpd_colors_dict.get(col, 'gray')
            alpha = 1
        else:
            color = cmpd_colors_dict.get(col, 'gray')
            alpha = 0.2
        
        bar.set_color(color)
        bar.set_alpha(alpha)
        bar.set_edgecolor('white')
        

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

# Build legend with matching alpha
# handles = []
# for col in df.columns:
#     alpha = 1 if col in top_cols_global else 0.2
#     wrapped_label = '\n'.join(textwrap.wrap(col, width = 22))  # Adjust width as needed
#     handles.append(
#         mpatches.Patch(
#             color=cmpd_colors_dict[col],
#             label=wrapped_label,
#             alpha=alpha
#         )
#     )
# # Add legend
# ax.legend(handles = handles, 
#           ncols = 1,
#           fontsize = 16.5, 
#           loc = 'best',
#           bbox_to_anchor = (1,1),
#           reverse = True
#           )
ax.get_legend().remove()

# plt.legend(ncols = 1, fontsize = 18, loc = 'best',
#            bbox_to_anchor = (1, 1))

# adding total labels on top 
fontsize = 24
add_percent_bar_totals(ax, range(len(df)), 99, df_total, units, fontsize, fontweight)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.show()
plt.close()




# %% Full Factor TS w/ 3 subplots

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
    
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows = 3,
    ncols = 1,
    sharex = True,
    figsize = (8, 6),
    dpi = 300)

axis1 = ['Factor 1', 'Factor 2']
axis2 = ['Factor 4', 'Factor 5']
axis3 = ['Factor 3']
  
# plotting factor solns
for col in df.columns:
    if col in axis1:
        plt1 = ts_plot2(df, ax1, color = color_dict[col], linestyle = linestyle_dict[col])
    elif col in axis2:
       plt2 = ts_plot2(df, ax2, color = color_dict[col], linestyle = linestyle_dict[col])
    elif col in axis3:
        plt3 = ts_plot2(df, ax3, color = color_dict[col], linestyle = linestyle_dict[col])

ax1.set(
       xlim = (df.index[0] - timedelta(hours = 6), df.index[-1] + timedelta(hours = 6)), # adjusting x-lim
       xlabel = 'Local Date/Time',  
       #title = f'{len(df.columns)} Factor Solution' 
       )


for ax, t in zip([ax1, ax2, ax3], ['A', 'B', 'C']):
    ax.legend(
        ncols = 1,
        loc = 'upper right',
        fontsize = 'small'
        )
    ax.set_ylabel(
        'Normalized\nConc.'
        )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.text(
        x = -0.1,
        y = 0.9, 
        s = t,
        transform = ax.transAxes,
        weight = 'bold',
        fontsize = 14
        )

fig.autofmt_xdate()

ax1.xaxis.set_major_formatter(md.DateFormatter('%D'))


plt.show()
plt.close()


# %% Factor Full TS against Temp and Smoke Events

df = factor_ts.copy() # no overwriting! 
df.set_index('datetime', inplace = True)
df.drop(columns = ['hour'], inplace = True)
    
fig, (ax1, ax2) = plt.subplots(
    nrows = 2,
    ncols = 1,
    sharex = True,
    figsize = (8, 6), 
    dpi = 300)

# plotting against temperature
ax1.plot(weather['dt'], weather['temp_c'],
  color = 'maroon',
  linestyle = '--', 
  linewidth = 1,
  alpha = 0.5)
ax1.set(
        ylim  = [0, 40],
      )
ax1.set_ylabel('Median Temperature (°C)', color = 'maroon')
# changing axis color for plot readibility
ax1.spines['left'].set_color('maroon')
ax1.tick_params(axis = 'y', colors = 'maroon')
[t.set_color('maroon') for t in ax1.yaxis.get_ticklines()]
[t.set_color('maroon') for t in ax1.yaxis.get_ticklabels()]

# plotting against OA conc:
ax3 = ax1.twinx()
ax3.plot(oa_data['datetime_local'], oa_data['OA_ug_m3'],
         color = 'black',
         linestyle = '--', 
         linewidth = 1,
         alpha = 0.5)
ax3.set(
       ylim  = [0, 35]
                 )
ax3.set_ylabel('Organic Aerosol Mass\n(μg $m^-3$)', color = 'darkslategray')

# changing axis color for plot readibility
[t.set_color('darkslategray') for t in ax3.yaxis.get_ticklines()]
[t.set_color('darkslategray') for t in ax3.yaxis.get_ticklabels()]
ax3.tick_params(axis = 'y', colors = 'darkslategray')
ax3.spines['right'].set_color('darkslategray')
 
# plotting factor solns
for col in df.columns:
     ax2.plot(df.index, df[col],
             label = col,
             linewidth = 2,
             marker = 'o',
             markeredgecolor = 'black',
             markerfacecolor = None,
             markersize = 4,
             color = color_dict[col]
             ) 
ax2.set(
       ylabel = 'Normalized Cmpd Conc.',
       xlim = (df.index[0] - timedelta(hours = 6), df.index[-1] + timedelta(hours = 6)), # adjusting x-lim
       xlabel = 'Local Date/Time',
       ylim = [0, 13]
   #    title = f'{len(df.columns)} Factor Solution'
       )
legend = ax2.legend(
    loc = 'best',
    shadow = True,
    framealpha = 1.0,
    facecolor = 'white',
    labelcolor = 'black',
    edgecolor = 'black',
    ncols = 5
    )
legend.get_frame().set_alpha(1.0)

 
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
rect_box(heatwave_start, heatwave_end, ax2, '') # adding heatwave boxes
rect_box(heatwave_start, heatwave_end, ax1, '') # adding heatwave box

box_color = 'gray'
rect_box(smoke1_start, smoke1_end, ax2, '') # adding smoke boxes
rect_box(smoke2_start, smoke2_end, ax2, '') 
rect_box(smoke1_start, smoke1_end, ax1, '')
rect_box(smoke2_start, smoke2_end, ax1, '') 

fig.autofmt_xdate()
ax.xaxis.set_major_formatter(md.DateFormatter('%D %H:%M'))

for ax, t in zip([ax1, ax2], ['A', 'B']):
    ax.text(
            x = -0.1,
            y = 1.05,
            s = t,
            transform = ax.transAxes,
            fontsize = 12,
            fontweight = 'bold'
            )

plt.show()
plt.close()

# %% Full Factor TS w/ 2 subplots and smoke/heat events

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
for ax in [ax1, ax2, ax3, ax4]:
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


# %% Factor Soln Diel Plot - line

df = factor_ts.copy() # no overwriting! 

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

plt.show()
plt.close()   

# %% Factor Soln Diel Plot - bar chart

df = factor_ts.copy() # no overwriting! 


temp_list = []
for col in df.columns:
    if col not in ignore:
        grouped = df.groupby('hour')[col].mean()
        temp_list.append(grouped)
df = pd.DataFrame(data = temp_list)
df = df.transpose()



fig, ax = plt.subplots(
    figsize = (8, 5),
    dpi = 300
    ) 
df.plot(ax = ax, 
        stacked = True,
        kind = 'bar', 
        subplots = False, 
        sharex = True, 
        use_index = True,
        color = [color_dict[col] for col in df.columns],
        edgecolor = 'white',
        linewidth = 2,
        rot = 360,
        width = 0.85
        )

ax.set_xlabel('Hour of Day')
ax.set_ylabel('Median Normalized Cmpd Conc.')

ax.legend(ncols = 4,
          loc = 'upper center')

plt.show()
plt.close() 

# %% both diel plot styles, one plot

gridspec = {'hspace' : 0.2, 
            'wspace' : 0.2,
            'height_ratios' : [0.25, 0.5, 2],
            }
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows = 3,
    ncols = 1,
    sharex = True,
    figsize = (8, 6),
    dpi = 300,
    gridspec_kw = gridspec
    )

# ax1 = top line plot, top of broken axis
df = factor_ts.copy() # no overwriting! 
for col in df.columns:
    if col not in ignore:
        grouped = df.groupby('hour')[col].agg(
            mean = 'mean',
            q25 = lambda x: x.quantile(0.25),
            q75 = lambda x: x.quantile(0.75))
        ax1.plot(grouped.index, grouped['mean'], label = col,
                color = color_dict[col],
                 linestyle = 'solid', 
                 linewidth = 0.75,
                 marker = 'o',
                 markersize = 7)
        ax1.fill_between(grouped.index, grouped['q25'], grouped['q75'],
                        alpha = 0.1, color = color_dict[col])
ax1.set_ylim([5, 6])
ax1.set_yticks(
    ticks = np.linspace(5, 6, 2),
    labels = np.linspace(5, 6, 2),
    fontsize = 12
    )
ax1.set_xticks(
    ticks = np.linspace(6, 18, 7)
    )
ax1.text(
    x = -0.1,
    y = 1.2, 
    s = 'A',
    fontsize = 14,
    fontweight = 'bold',
    transform = ax1.transAxes
    )
#ax1.get_legend().remove()

# ax2 = top line plot, bottom of broken axis
df = factor_ts.copy() # no overwriting! 
df = df[['datetime', 'Factor 1', 'Factor 3', 'Factor 2', 'Factor 4', 'hour']]
for col in df.columns:
    if col not in ignore:
        grouped = df.groupby('hour')[col].agg(
            mean = 'mean',
            q25 = lambda x: x.quantile(0.25),
            q75 = lambda x: x.quantile(0.75))
        ax2.plot(grouped.index, grouped['mean'], label = col,
                color = color_dict[col],
                 linestyle = 'solid', 
                 linewidth = 0.75,
                 marker = 'o',
                 markersize = 7)
        ax2.fill_between(grouped.index, grouped['q25'], grouped['q75'],
                        alpha = 0.1, color = color_dict[col])
ax2.set_ylim([0, 3.5])
ax2.set_yticks(
    ticks = np.linspace(0, 3.5, 3),
    labels = np.linspace(0, 3.5, 3),
    fontsize = 12
    )
ax2.set_xticks(
    ticks = np.linspace(6, 18, 7)
    )
ax2.text(
    x = 3.5,
    y = 0.8,
    s = 'Factor Conc.',
    fontsize = 14,
    rotation = 90
    )
#ax2.get_legend().remove()

ax1.xaxis.set_visible(False)
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)

# setting those horizonantal lines
# source code for the split axis: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = {'marker' : [(-1, -d), (1, d)], 'markersize' : 12,
              'linestyle' : "none", 'color' : 'k', 'mec' : 'k', 'mew' : 1, 'clip_on' : False}
ax1.plot([0, 1], [0, 0], transform = ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform = ax2.transAxes, **kwargs)

# ax3 = bottom bar plot
df1 = factor_ts.copy() # no overwriting! 
df1 = df1[['datetime', 'Factor 1', 'Factor 3', 'Factor 2', 'Factor 4', 'hour']]
temp_list = []
for col in df1.columns:
    if col not in ignore:
        grouped = df1.groupby('hour')[col].mean()
        temp_list.append(grouped)
temp_df = pd.DataFrame(data = temp_list)
temp_df = temp_df.transpose()
temp_df.index = temp_df.index.astype(float)

x = temp_df.index.values
bottom = np.zeros(len(x))
for col in temp_df.columns:
    ax3.bar(
        x,
        temp_df[col].values,
        bottom=bottom,
        width=0.8,
        color=color_dict[col],
        edgecolor='white',
        linewidth=2,
        label=col
    )
    bottom += temp_df[col].values
ax3.set_yticks(
    ticks = np.linspace(0, 7, 4),
    labels = np.linspace(0, 7, 4),
    fontsize = 12
    )

ax3.set_xlabel('Hour of Day',
               fontsize = 14)
ax3.set_ylabel('∑Factor Conc.',
               fontsize = 14,
               labelpad = 15)
ax3.legend(ncols = 4,
          loc = 'upper center',
          bbox_to_anchor = (0.5, 1.7),
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
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    

plt.show()

# %% both diel plot styles, one plot, F1/F3 and F2/F4

frag_factors = ['Factor 1', 'Factor 3']
ind_factors = ['Factor 2', 'Factor 4']

gridspec = {'hspace' : 0.2, 
            'wspace' : 0.2,
            'height_ratios' : [0.25, 0.5, 0.75, 2],
            }
fig, (ax1, ax2, ax4, ax3) = plt.subplots(
    nrows = 4,
    ncols = 1,
    sharex = True,
    figsize = (8, 6),
    dpi = 300,
    gridspec_kw = gridspec
    )

# ax1 = top line plot, F1/F3, top of broken axis
df = factor_ts.copy() # no overwriting! 
df = df[['datetime', 'Factor 1', 'Factor 3', 'Factor 2', 'Factor 4', 'hour']]
for col in df.columns:
    if col not in ignore:
        if col in frag_factors:
            grouped = df.groupby('hour')[col].agg(
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
ax1.set_ylim([3.5, 6])
ax1.set_yticks(
    ticks = np.linspace(3.5, 6, 2),
    labels = np.linspace(3.5, 6, 2),
    fontsize = 12
    )
ax1.set_xticks(
    ticks = np.linspace(6, 18, 7)
    )
ax1.text(
    x = -0.1,
    y = 1.2, 
    s = 'A',
    fontsize = 14,
    fontweight = 'bold',
    transform = ax1.transAxes
    )
#ax1.get_legend().remove()
ax1.xaxis.set_visible(False)
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)

# setting those horizonantal lines
# source code for the split axis: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = {'marker' : [(-1, -d), (1, d)], 'markersize' : 12,
              'linestyle' : "none", 'color' : 'k', 'mec' : 'k', 'mew' : 1, 'clip_on' : False}
ax1.plot([0, 1], [0, 0], transform = ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform = ax2.transAxes, **kwargs)

# ax2 = top line plot, F1/F3, bottom of broken axis
# ax4 = top line plot, F2/F4, bottom of broken axis
df = factor_ts.copy() # no overwriting! 
df = df[['datetime', 'Factor 1', 'Factor 3', 'Factor 2', 'Factor 4', 'hour']]
for col in df.columns:
    if col not in ignore:
        if col in frag_factors:
            grouped = df.groupby('hour')[col].agg(
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
        if col in ind_factors:
            grouped = df.groupby('hour')[col].agg(
                median = 'median',
                q25 = lambda x: x.quantile(0.25),
                q75 = lambda x: x.quantile(0.75))
            ax4.plot(grouped.index, grouped['median'], label = col,
                    color = color_dict[col],
                     linestyle = 'solid', 
                     linewidth = 0.75,
                     marker = 'o',
                     markersize = 7)
            ax4.fill_between(grouped.index, grouped['q25'], grouped['q75'],
                            alpha = 0.1, color = color_dict[col])
ax2.set_ylim([0, 2])
ax2.set_yticks(
    ticks = np.linspace(0, 2, 3),
    labels = np.linspace(0, 2, 3),
    fontsize = 12
    )
ax4.set_ylim([0, 6])
ax4.set_yticks(
    ticks = np.linspace(0, 6, 4),
    labels = np.linspace(0, 6, 4),
    fontsize = 12
    )
ax2.text(
    x = 3.5,
    y = -1.5,
    s = 'Factor Conc.',
    fontsize = 14,
    rotation = 90
    )
#ax2.get_legend().remove()


# ax3 = bottom bar plot
df1 = factor_ts.copy() # no overwriting! 
df1 = df1[['datetime', 'Factor 1', 'Factor 3', 'Factor 2', 'Factor 4', 'hour']]
temp_list = []
for col in df1.columns:
    if col not in ignore:
        grouped = df1.groupby('hour')[col].median()
        temp_list.append(grouped)
temp_df = pd.DataFrame(data = temp_list)
temp_df = temp_df.transpose()
temp_df.index = temp_df.index.astype(float)

x = temp_df.index.values
bottom = np.zeros(len(x))
for col in temp_df.columns:
    ax3.bar(
        x,
        temp_df[col].values,
        bottom=bottom,
        width=0.8,
        color=color_dict[col],
        edgecolor='white',
        linewidth=2,
        label=col
    )
    bottom += temp_df[col].values
ax3.set_yticks(
    ticks = np.linspace(0, 7, 4),
    labels = np.linspace(0, 7, 4),
    fontsize = 12
    )

ax3.set_xlabel('Hour of Day',
               fontsize = 14)
ax3.set_ylabel('∑Factor Conc.',
               fontsize = 14,
               #labelpad = 9
               )
ax3.legend(ncols = 4,
          loc = 'upper center',
          bbox_to_anchor = (0.5, 2.2),
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
  
for ax in [ax1, ax2, ax3, ax4]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    

plt.show()

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

# %% Weekday vs Weekend factor trends

df = factor_ts.copy() # no overwriting! 

    
weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] # set days of week in a weekday
weekend = ['Saturday', 'Sunday']  # set days in a weekend
grouping = ['weekend', 'weekday'] # we want to sort data based on weekday or weekend status

ignore = ['datetime', 'hour', 'dow', 'weekday_weekend'] 

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
            y = 0.7,
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
    y = 12, 
    s = 'Median Cmpd Conc. (Normalized ppb)',
    rotation = 'vertical'
    )
plt.xlabel('Hour of Day')# setting x-axis label on bottom plot only
plt.show()
plt.close()
   
# %% Corr Matrix (R2) w/ MetaVariables + Common Compounds Over Time

# getting the big ass meta data df ready

meta_data_copy = meta_data.copy() # no overwriting !
cmpds_ppb_copy = common_cmpds_ppb.copy()

starttime = meta_data_copy['start_time']

meta_data_copy.drop(columns = ['start_time', 'avg_rain_in/s'], inplace = True)
cmpds_ppb_copy.drop(columns = ['start_time'], inplace =  True)
meta_data_copy.set_index('hour', inplace = True)
cmpds_ppb_copy.set_index('hour', inplace = True)

meta_data_concat = pd.concat([meta_data_copy, cmpds_ppb_copy], axis = 1)

df = factor_ts.copy() # no overwriting! 

df.drop(columns = ['datetime'], inplace = True)
df.set_index('hour', inplace = True)

# adding metadata to each factor df to run the corr matrix
df = pd.concat([df, meta_data_concat], axis = 1) 

df = df.corr() # getting r values
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

# %% Corr Matrix (r) w/ All Ground Compounds + PMF Factors 

cmpds = full_cmpd_dataset.copy()
cmpds.set_index('start_time', inplace = True)
df = factor_ts.copy() # no overwriting! 

df.drop(columns = ['hour'], inplace = True)
df.set_index('datetime', inplace = True)

# adding metadata to each factor df to run the corr matrix
df = pd.concat([df, cmpds], axis = 1) 


df = df.corr() # getting r values
#df = np.square(df) # getting R2 values

fig, ax = plt.subplots(figsize = (30, 30), dpi = 300)
cax = ax.imshow(df, cmap = 'coolwarm',  # plot matrix w/ chosen colormap
          interpolation = 'nearest') # round to nearest decimal place for heatmap value
  
cbar = plt.colorbar(cax, shrink = 0.95, label = 'R2 Coefficient Correlations')
cbar.ax.tick_params(labelsize = 24)
ax.set_xticks(range(len(df.columns))) # set xticks based on the columns in the matrix
ax.set_xticklabels(df.columns, 
         rotation = 90, fontsize = 14) # set x values based on columns and rotate for visibility
ax.set_yticks(range(len(df.columns))) # repeat for yaxis
ax.set_yticklabels(df.columns, fontsize = 14)
ax.set_yticklabels(df.columns, fontsize = 14)
for (i, j), value in np.ndenumerate(df): # (coordinates) & r-value in loop in values within the mask:
  #     if i > j: # ensure annotated values are in the bottom half of the plot
  if float(f'{value:.2}') >= 0.3 : # only displaying R higher than 0.5
     ax.text(j, i, f'{value:.2}', ha = 'center', va = 'center', color = 'black', fontsize = 7) # adjusting the value:# will change the sigfigs
  if float(f'{value:.1}') <= - 0.3 : 
     ax.text(j, i, f'{value:.1}', ha = 'center', va = 'center', color = 'black', fontsize = 7)

plt.tight_layout()

plt.show()
plt.close()



# %% Corr Matrix (r) w/ All MetaVariables + Common Compounds

# getting the big ass meta data df ready

meta_data_copy = meta_data.copy() # no overwriting !
cmpds_ppb_copy = common_cmpds_ppb.copy()

starttime = meta_data_copy['start_time']

meta_data_copy.drop(columns = ['start_time', 'avg_rain_in/s'], inplace = True)
cmpds_ppb_copy.drop(columns = ['start_time'], inplace =  True)
meta_data_copy.set_index('hour', inplace = True)
cmpds_ppb_copy.set_index('hour', inplace = True)

meta_data_concat = pd.concat([meta_data_copy, cmpds_ppb_copy], axis = 1)

df = factor_ts.copy()  # no overwriting! 

df.drop(columns = ['datetime'], inplace = True)
df.set_index('hour', inplace = True)

# adding metadata to each factor df to run the corr matrix
df = pd.concat([df, meta_data_concat], axis = 1)
df = df.corr() # getting pearson coeficcients

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

plt.show()
plt.close()

# %% Corr Matrix (r) w/ Specific Meta Tracers and PMF Factors

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
                     color = 'black', fontsize = 24) 
      elif float(f'{value:.2}') <= - 0.3 : # adjusting the value:# will change the sigfigs
         ax.text(j, i, f'{value:.2f}', ha = 'center', va = 'center', 
                 color = 'black', fontsize = 24)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

plt.show()
plt.close()


# %% Bar Charts of OH Reactivities by Factor

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

plt.show()
plt.close()

# %% Total mixing ratios as pie chart

df = common_cmpds_ppb.copy()
df.drop(columns = ['hour'], inplace = True)
df.set_index('start_time', inplace = True)
# need cmpds as index
df = df.transpose() 
df.index.rename('compounds', inplace = True)

# only looking at compounds included in the PMF 
factor_df = profile_conc.copy()
factor_df.set_index('compounds', inplace = True)
factor_cmpd_list = factor_df.index
df = df.loc[factor_cmpd_list]

# adding total mixing ratios across the samples
conc_totals = df.sum(axis = 1, numeric_only = True)
total_conc_ppb = conc_totals.sum()
conc_totals = conc_totals.to_frame()
conc_totals.rename(columns = {0 : 'summed_conc_ppb'}, inplace = True)
conc_totals['%_conc'] = conc_totals['summed_conc_ppb'] / total_conc_ppb * 100

# getting top 5 cmpds + summing the rest into 'other'
conc_totals.sort_values(by = '%_conc', ascending = False, inplace = True)
top_number = 5
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
    f'Total Mixing Ratio: {total_conc_ppb:0.1f} $ppb_v$', 
    x = 0.48,
    y = 0.99,
    fontsize = 20,
    fontweight = 'normal'
    )

plt.show()

# %% OH reactivities of measured data + pie chart

df = common_cmpds_ppb.copy()
df.drop(columns = ['hour'], inplace = True)
df.set_index('start_time', inplace = True)
# need cmpds as index
df = df.transpose() 
df.index.rename('compounds', inplace = True)

# only looking at compounds included in the PMF 
factor_df = profile_conc.copy()
factor_df.set_index('compounds', inplace = True)
factor_cmpd_list = factor_df.index
df = df.loc[factor_cmpd_list]

# need list of compounds to filter the reference data by
cmpd_list = df.index

ref_data = cmpd_refs.copy()
# need index names to match
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# applying number density function to convert ppb to molc/cm3
for index, row in df.iterrows(): 
    for col in row.index:
     if pd.api.types.is_numeric_dtype(df[col]):
         df.at[index, col] = number_density(
             row[col],
         )

# converting to reactivities against OH (1/s)
for col in df:
    df[col] = df[col] * factor_ref_data['k_oh_298']

# adding total OH values across the samples
oh_totals = df.sum(axis = 1, numeric_only = True)
total_measured_oh = oh_totals.sum()
oh_totals = oh_totals.to_frame()
oh_totals.rename(columns = {0 : 'summed_oh'}, inplace = True)
oh_totals['%_oh'] = oh_totals['summed_oh'] / total_measured_oh * 100

# getting top 5 cmpds + summing the rest into 'other'
oh_totals.sort_values(by = '%_oh', ascending = False, inplace = True)
top_number = 5
others = pd.DataFrame(data = {
    'other' : oh_totals[top_number:].sum()
    })
others = others.transpose()
df = oh_totals[:top_number].copy()
df = pd.concat([df, others], axis = 0)


# making a pie chart
fig, ax = plt.subplots(figsize = (16, 16), dpi = 300)

values = df['summed_oh'].value_counts(dropna = True)

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = pct * total / 100.0
        return '{:.1f}%\n({v:0.1f} sec$^-$$^1$)'.format(pct, v=val)
    return my_format

df.plot(ax = ax,
        kind = 'pie',
        y = '%_oh',
        use_index = 'True',
        colors = [cmpd_colors_dict[value] for value in df.index],
        wedgeprops = {'edgecolor' : 'white', 'linewidth' : 3},
        autopct = autopct_format(df['summed_oh']),
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
    f'Total OH: {total_measured_oh:0.1f} sec$^-$$^1$', 
    x = 0.48,
    y = 0.99,
    fontsize = 20,
    fontweight = 'normal'
    )

plt.show()

# %% Wind Rose w/ wind speed

# averaged to sorbent tube times:
#fig, ax = plt.subplots(figsize = (12, 12), dpi = 200)
ax = WindroseAxes.from_ax()
ax.bar(meta_data['median_WindDir'], meta_data['avg_WindSpeed_m/s'],
       normed = True,
       opening = 0.8,
       edgecolor = 'white',
       cmap = cm.viridis_r)
ax.set_legend(
    title = 'Wind Speed (m/s)\nAveraged by Sampling Blocks'
    )

# full wind rose
#fig, ax = plt.subplots(figsize = (12, 12), dpi = 200)
ax = WindroseAxes.from_ax()
windchart = ax.bar(weather['wind_dir_deg'], weather['wind_speed_m/s'],
       normed = True,
       opening = 0.8,
       edgecolor = 'white',
       cmap = cm.viridis_r)
ax.set_legend(
    title = 'Wind Speed (m/s)\n(Full DataSet')
plt.show()
plt.close()

# %% Wind Rose against Conc.

df = factor_ts.copy()
dt_cols_ignore = ['datetime', 'hour']

# averaged to sorbent tube times:

for col in df.columns:
    if col not in dt_cols_ignore:
        plt.style.use('default')
        fig = plt.figure()

        ax = WindroseAxes.from_ax()
        ax.bar(meta_data['median_WindDir'], df[col],
               nsector = 36, #each bar is 10 degrees wide (360/36)
               calm_limit = 0.0, # adds extra space to the ends of the bars
               normed = True, # converts from frequency to prob. density plot
               opening = 0.8, # size of the little hole in the middle
               edgecolor = 'black',
               cmap = cm.Reds,
               facecolor = 'whitecapitalize'
               )
        ax.set_title(f'{col}',
                     fontsize = 26,
                     fontweight = 'bold',
                     color = 'black',                #color_dict[col],
                     loc = 'center',
                     pad = 50 # increasing space between title and top of graph
                     )
        legend = ax.set_legend(
            loc = 'center right',
            fontsize = 20
            )
        ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'],
                           fontsize = 24,
                           )
        ax.tick_params(axis = 'both', which='major', pad = 15)
        
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_fontsize(20)
        plt.show()
        
# %% pollution Roses w/ conc. and wind speed

df = factor_ts.copy()
dt_cols_ignore = ['datetime', 'hour']

# Parameters
num_bins = 36  # Number of directional bins (e.g., 15 degrees per bin)
bin_edges = np.linspace(0, 360, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

wind_spd = meta_data['avg_WindSpeed_m/s']
wind_dir = meta_data['median_WindDir']


# Plot
fig, axs = plt.subplots(2, 2, 
                       subplot_kw = {'projection': 'polar'}, 
                       figsize = (10, 8),
                       dpi = 300
                       )
axs = axs.flatten()
cols_to_plot = [col for col in df.columns if col not in dt_cols_ignore]
plt.subplots_adjust(hspace = 0.3, wspace = 0.25) # making the subplots closer together

for idx, col in enumerate(cols_to_plot[:5]):
    ax = axs[idx] 
    
    # Subplot label (C, D, E, F...)
    # chr(65) is 'A' in ASCII, B = chr(66)
    ax.text(x = 0, y = 1.3, 
            s = chr(65 + idx), 
            transform = ax.transAxes,
            fontsize = 16, fontweight = 'bold', 
            va = 'top', ha = 'right')

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
                               norm = plt.Normalize(vmin = np.nanmin(mean_voc), vmax = np.nanmax(mean_voc)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = ax, pad = 0.1, 
                        orientation = 'vertical',
                        shrink = 0.9,
                        format = lambda x, _: f"{x:.1f}")
    # cbar.set_label(f'{col} Mixing Ratio ($ppb_v$)',
    #                fontsize = 12)
    
    # Set the zero location and direction
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Set radial grid and labels
    ax.set_ylim(0, 1.75)
    ax.set_rticks([0.25, 0.75, 1.25, 1.75])
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

# %% pollution roses by TOD

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

# %% Temp-dependence of biogenic residuals and average temp

import statsmodels.api as sm # re-loading bc dumb new error

df = scaled_residual_ts.copy()
df.drop(df.tail(1).index,inplace = True) # dropping Q/Qexp row

x_variable = meta_data['median_temp_c'] # edit!
x_name = 'Average Temperature' # edit!
units = '(° C)' # edit!
bios = ['limonene', 'a-pinene']

x_linear = x_variable.copy()
x_linear = sm.add_constant(x_linear) 

color_list = ['#9B287B', '#9B287B', '#C0B141', '#C0B141'] # edit if needed.

fig, ax = plt.subplots(
    figsize = (8, 8)
    )
ax.set_prop_cycle(color = color_list)
    
for col in df.columns:
    if col in bios:
        y_linear = df[col]
        y_linear = y_linear.loc[x_linear.index]
        model = sm.OLS(y_linear, x_linear).fit()
        y_pred = model.predict(x_linear)
        pearson = scipy.pearsonr(x_variable, y_linear)
        
        ax.plot(x_variable, df[col], #label = col,
                linestyle = '', marker = 'o')
        ax.plot(x_variable, y_pred,
                #label = f'{col}: $R^2$ = {model.rsquared:0.4f}',
                label = f'{col}: $r$ = {pearson[0]:0.2f}',
                marker = '',
                alpha = 0.75)
        ax.set_xlabel(f'{x_name} {units}',
                      fontsize = 14)
        ax.set_ylabel('Scaled Factor Residual',
                      fontsize = 14)
        ax.legend(fontsize = 14)
        # ax.set_title(
        #     f'{len(percent_factor_total.columns) - 1} Factor Solution'
        #              )
             
plt.show()
plt.close()

# %% Temp-dependence of biogenic residuals and max temp

import statsmodels.api as sm # re-loading bc dumb new error

df = scaled_residual_ts.copy()
df.drop(df.tail(1).index,inplace = True) # dropping Q/Qexp row

x_variable = meta_data['max_temp_c'] # edit !
x_name = 'Max Temperature' # edit!
units = '(° C)' # edit!

bios = ['limonene', 'a-pinene']

x_linear = x_variable.copy()
x_linear = sm.add_constant(x_linear)

color_list = ['#9B287B', '#9B287B', '#C0B141', '#C0B141'] # edit if needed


fig, ax = plt.subplots()
ax.set_prop_cycle(color = color_list)

for col in df.columns:
    if col in bios:
        y_linear = df[col]
        y_linear = y_linear.loc[x_linear.index]
        model = sm.OLS(y_linear, x_linear).fit()
        y_pred = model.predict(x_linear)
        pearson = scipy.pearsonr(x_variable, y_linear)
        
        ax.plot(x_variable, df[col], #label = col,
                linestyle = '', marker = 'o')
        ax.plot(x_variable, y_pred,
                # label = f'{col}: $R^2$ = {model.rsquared:0.4f}',
                label = f'{col}: $r$ = {pearson[0]:0.4f}',
                marker = '',
                alpha = 0.75)
        ax.set_xlabel(f'{x_name} {units}')
        ax.set_ylabel('Scaled Factor Residual')
        ax.legend()
        # ax.set_title(
        #     f'{len(percent_factor_total.columns) - 1} Factor Solution'
        #              )
 
plt.show()
plt.close()

# %% Corr Matrix (r) w/ Residuals Over Time

#getting the big ass meta data df ready

meta_data_copy = meta_data.copy() # no overwriting !

starttime = meta_data_copy['start_time']

meta_data_copy.drop(columns = ['start_time', 'avg_rain_in/s'], inplace = True)
meta_data_copy.set_index('hour', inplace = True)

df = residual_ts.copy()
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
plt.title('Residuals',
          fontsize = 20)
plt.show()
plt.close()

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
    

# %% Bar charts of original ground data by TOD

df = common_cmpds_ppb.copy()
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
    ticks = np.linspace(0, 4, 5),
    labels = np.linspace(0, 4, 5),
    fontweight = 'normal',
    fontsize = 16)

#labeling the top 5 compounds making up each time block:
# for bar_idx, (index, row) in enumerate(df.iterrows()):
#     top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(5).index.tolist()
    
#     cum_height = 0
#     for col in df.columns:
#         height = row[col]
#         mid_point = cum_height + height / 2
#         if col in top_cols and height > 0:
            
#                     wrapped_label = '\n'.join(textwrap.wrap(col, width = 25))  # Adjust width as needed
                
#                     ax.text(
#                     x = bar_idx,
#                     y = mid_point,
#                     s = f'{wrapped_label}' + ': ' + f'{height:.2f}',
#                     ha = 'center',
#                     va = 'center',
#                     fontsize = 12,
#                     color = 'black'  # or a color that contrasts the bar
#                     )
#         cum_height += height

units = '$ppb_v$'
add_bar_totals(range(len(df)), 4.1, df_total, units, 16, 'normal')

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.get_legend().remove()

plt.show()
plt.close()

# %% Diel plots of specific compounds

df = full_cmpd_dataset.copy()
df.rename(columns = {'start_time' : 'datetime'}, inplace = True)

df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x >= 6
                                   else 
                                   'am2' if x <= 12 and x >= 10
                                   else
                                   'pm1' if x <= 16 and x > 13
                                   else 
                                   'pm2'
                                   )

start_time = df['datetime']
df.drop(columns = ['datetime', 'hour'], inplace = True)
     
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

xlabels = [
    'Morning',
    'Mid-Morning',
    'Afternoon',
    'Early Evening'
    ]

    
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
    nrows = 5,
    ncols = 1,
    sharex = True,
    figsize = (8, 8),
    dpi = 300)

ax3.text( # making a y-axis label for all subplots
    x = -0.6,
    y = -0.13, 
    s = 'Median Mixing Ratio ($ppb_v$)',
    rotation = 'vertical',
    fontsize = 14,
    fontweight = 'bold'
    )

tracer = 'toluene'
plt1 = ts_plot(chosen_df = df, 
               tracer = tracer,
               axis = ax1)

tracer = 'benzyl acetate'
plt2 = ts_plot(chosen_df = df, 
               tracer = tracer,
               axis = ax2)

tracer = 'acetophenone'
plt3 = ts_plot(chosen_df = df, 
               tracer = tracer,
               axis = ax3)

tracer = 'o-xylene'
plt4 = ts_plot(chosen_df = df, 
               tracer = tracer,
               axis = ax4)

tracer = '3-ethyl toluene'
plt5 = ts_plot(chosen_df = df, 
               tracer = tracer,
               axis = ax5)

ax5.set_xticks(
    ticks = range(len(df)),
    labels = xlabels,
    fontsize = 13,
    fontweight = 'bold'
    )


plt.show()

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

# %% OFP for top compounds from original data (pie chart)

'''
Only using compounds with published MIR values 
'''

df = common_cmpds_ppb.copy()
df.drop(columns = ['hour'], inplace = True)
df.set_index('start_time', inplace = True)
# want cmpds as row names and samples as columns:
df = df.transpose()

# want to only look at compounds included in the PMF input:
factor_df = profile_conc.copy() 
factor_df.set_index('compounds', inplace = True)
factor_cmpd_list = factor_df.index

df = df.loc[factor_cmpd_list] # filtering
cmpd_list = df.index # list of cmpds in the ground samples

# filtering the reference data for those cmpds
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# currently, df = matrix of ppb values, samples as columns and cmpds as rows

# converting to OFP (ppb)
for value in df:
    df[value] = df[value] * ref_data['MIR_best']
    
df = df.transpose()
# now we have a matrix, with compounds as the column and each sample as the index



df = pd.DataFrame(df.median())
df.rename(columns = {0 : 'compound_median_ofp_ppb'}, inplace = True)
df_total = df['compound_median_ofp_ppb'].sum()
df.sort_values(by = 'compound_median_ofp_ppb', ascending = False, inplace = True)
top_number = 5
others = pd.DataFrame(data = {
    'other' : df[top_number:].sum()
    })
others = others.transpose()
df = df[:top_number].copy()
df = pd.concat([df, others], axis = 0)

fig, ax = plt.subplots(figsize = (16, 16), dpi = 300)

values = df['compound_median_ofp_ppb'].value_counts(dropna = True)

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = pct * total / 100.0
        return '{:.1f}%\n({v:0.1f} $ppb_v$)'.format(pct, v=val)
    return my_format

df.plot(ax = ax,
        kind = 'pie',
        y = 'compound_median_ofp_ppb',
        use_index = 'True',
        colors = [cmpd_colors_dict[value] for value in df.index],
        wedgeprops = {'edgecolor' : 'white', 'linewidth' : 3},
        autopct = autopct_format(df['compound_median_ofp_ppb']),
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
    f'Total OFP: {df_total:0.1f} $ppb_v$', # based on all median values
    x = 0.48,
    y = 0.93,
    fontsize = 20,
    fontweight = 'normal'
    )

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
    df[col] = df[col] * ref_data['MIR_best']
    
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
    axis.set_yticks(ticks = np.linspace(0, 1.75, 5),
                    labels = np.linspace(0, 1.75, 5),
                    fontsize = 12)
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.set_title(f'Factor {i + 1}',
                   x = 0.85, 
                   y = 0.88,
                   fontweight = 'bold',
                   transform = axis.transAxes
                   )
    axis.set_ylim(0, 1.75)
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

# %% Factor 3 OFP insert
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
    df[col] = df[col] * ref_data['MIR_best']
    
df_factor_totals = df.sum()
f1 = df['Factor 1']
f2 = df['Factor 2']
f3 = df['Factor 3']
f4 = df['Factor 4']
f5 = df['Factor 5']

factor_dfs = [f1, f2, f3, f4, f5] 
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
for x in range(0, 5):
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
        axis.text(x = i, y = y[i] + 0.01,  # setting x and y positions of labels - centering with bar and then slightly above top of bar
                 s = f'{value[i] / total * 100:0.1f}%', # setting and formatting what appears in the label
                 ha = 'center',
                 fontsize = 20
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
    axis.set_xticks(ticks = range(len(labels)), 
                    labels = '', 
                    fontsize = 9,
                    rotation = 0
                    )
    axis.set_yticks(ticks = np.linspace(0, 0.2, 3),
                    labels = np.linspace(0, 0.2, 3),
                    fontsize = 18)
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.set_title('Factor 3',
                   x = 0.85, 
                   y = 0.88,
                   fontweight = 'bold',
                   transform = axis.transAxes,
                   fontsize = 24
                   )
    axis.set_ylim(0, 0.2)
    add_percent_bar_totals(axis, df.index, values, values, total)

fig, ax = plt.subplots(
     nrows = 1,
     ncols = 1,
     sharey = True,
     dpi = 300,
     figsize = (12, 5)
     )
bar_graph_subplot(df = factor_dfs[2], axis = ax)
ax.set_ylabel(
    'OFP ($ppb_v$)',
    fontsize = 24,
    labelpad = 10
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

# %% Van Krevveln plots vs C* (all compounds)

#cstar equation from SI, Donahue et al (2006); https://doi.org/10.1021/es052297c


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
axes[0].scatter(df['cstar_ug_m3'], df['O:C'], 
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
axes[1].set_xlabel('Log C* (μg/$m^3$)', fontweight = fontweight)


cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), ax = axes.ravel().tolist(), pad = 0.03)
cbar.set_label(label = '$k_{OH}$ (cm$^3$ molecules$^{-1}$ sec$^{-1}$)', 
                   fontsize = 11) # set label of the colorbar on the top plot

# setting stuff to bold
for x in [0, 1]:
    axes[x].set_yticklabels(axes[x].get_yticks(), weight = fontweight) # setting y-ticks to bold
    for label in axes[x].get_xticklabels(): # setting axis labels to bold
        label.set_fontweight(fontweight)

axes[0].set_ylim([-0.2, 1.5])
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[1].set_ylim([-0.2, 4.5])
axes[1].set_xlim([df['cstar_ug_m3'].min() - 30, df['cstar_ug_m3'].max() + 2E9])
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#axes[0].set_title('All Quantified Compounds (sized by MW)')

plt.show()

# %% Legend only plot  # uncomment to print

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
    
# )

# # Remove everything else from the legend's figure
# legend_ax.axis('off')

# %% Timeseries of summed OHR and OFP
'''
OHR
'''
df = common_cmpds_ppb.copy()
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
df_ofp = common_cmpds_ppb.copy()
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
    df_ofp[value] = df_ofp[value] * ref_data['MIR_best']
    
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

# %%  MR, OHR, and OFP by TOD (% distribution)

xlabels = [
    'Morning',
    'Mid-Morning',
    'Afternoon',
    'Early Evening'
    ]
totals = {}

'''
Calculating total OH for all PMF compounds:
'''
df_oh = common_cmpds_ppb.copy()
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
df_oh.insert(loc = 0, column = 'hour', value = hour, allow_duplicates = False)
units_oh = 'sec$^-$$^1$'

'''
calculating OFP for all pmf compounds
'''
df_ofp = common_cmpds_ppb.copy()
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
    df_ofp[value] = df_ofp[value] * ref_data['MIR_best']
    
df_ofp = df_ofp.transpose()
df_ofp.reset_index(inplace = True)
df_ofp.insert(loc = 0, column = 'hour', value = hour, allow_duplicates = False)
units_ofp = '$ppb_v$'

'''
MR df too 
'''
df_mr = common_cmpds_ppb.copy()
df_mr.drop(columns = ['start_time'], inplace = True)
variable_count = len(df_mr.index) 
compounds = df_mr.columns
units_mr = '$ppb_v$'

'''
all together now
'''
dfs = {'mr': df_mr, 'oh': df_oh, 'ofp': df_ofp}

for name, df in dfs.items():

    df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x <= 9 and x >= 6
                                       else 'am2' if x <= 12 and x >= 10
                                       else 'pm1' if x <= 16 and x > 13
                                       else 'pm2')

    df.drop(columns=['hour'], inplace=True)

    am1 = df[df['timeblock'] == 'am1']
    am2 = df[df['timeblock'] == 'am2']
    pm1 = df[df['timeblock'] == 'pm1']
    pm2 = df[df['timeblock'] == 'pm2']

    block_dfs = [am1, am2, pm1, pm2]
    block_strings = ['am1', 'am2', 'pm1', 'pm2']

    block_dfs = [x.median(numeric_only=True) for x in block_dfs]
    block_dfs = [pd.DataFrame(x) for x in block_dfs]

    for block_df, value in zip(block_dfs, block_strings):
        block_df.rename(columns={0: value}, inplace=True)

    concat = pd.concat(block_dfs, axis=1)
    df_new = concat.transpose()   # final timeblock dataframe
    df_total = concat.sum(axis=0, numeric_only=True)
    totals[name] = df_total
    dfs[name] = df_new   # overwrite dataframe
    
df_mr = dfs['mr']
df_oh = dfs['oh']
df_ofp = dfs['ofp']    



'''
Making the figure
'''
fig, (ax1, ax2, ax3) = plt.subplots(
    ncols = 1, nrows = 3,
    sharex = True,
    figsize = (13, 15), dpi = 300
    )
for ax, name, df in zip(
        [ax1, ax2, ax3],
        ['mr', 'oh', 'ofp'],
        [df_mr, df_oh, df_ofp]):
    # get correct totals + units
    df_total = totals[name]
    units = {'mr': units_mr, 'oh': units_oh, 'ofp': units_ofp}[name]
    df_pct = df.div(df.sum(axis=1), axis=0) * 100  # convert to %
    df_pct.plot(ax = ax, 
                   stacked = True,
                     kind = 'bar', 
                     subplots = False, 
                     sharex = True, 
                     use_index = True, 
                     width = 0.95,
              ylim = [0, 100],
              fontsize = 14,
              color = [cmpd_colors_dict[col] for col in df.columns],
              edgecolor = 'white', 
              linewidth = 0.75,
              rot = 360
              )
    ax.get_legend().remove()
    ax.set_yticks(
        ticks = np.linspace(0, 100, 5),
        labels = np.linspace(0, 100, 5),
        fontweight = 'normal',
        fontsize = 16)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    add_percent_bar_totals(ax, x = range(len(df)), y = 105, total = df_total,
                           units = units, fontsize = 16, fontweight = 'normal')

ax1.set_ylabel(
    f'Median Mixing Ratios ({units_mr})',
    fontsize = 16,
    fontweight = 'normal',
    labelpad = 6
    )
ax2.set_ylabel(
    f'Median OHR ({units_oh})',
    fontsize = 16,
    fontweight = 'normal',
    labelpad = 6
    )
ax3.set_ylabel(
    f'Median OFP ({units_ofp})',
    fontsize = 16,
    fontweight = 'normal',
    labelpad = 6
    )
ax1.set_xticks(
    ticks = range(len(df_mr)),
    labels = xlabels,
    fontsize = 16,
    fontweight = 'normal'
    )




# # labeling the top 5 compounds making up each time block:
# for bar_idx, (index, row) in enumerate(df.iterrows()):
#     top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(5).index.tolist()
    
#     cum_height = 0
#     for col in df.columns:
#         height = row[col]
#         mid_point = cum_height + height / 2
#         if col in top_cols and height > 0:
            
#                     wrapped_label = '\n'.join(textwrap.wrap(col, width = 25))  # Adjust width as needed
                
#                     ax.text(
#                     x = bar_idx,
#                     y = mid_point,
#                     s = f'{wrapped_label}' + ': ' + f'{height:.2f}',
#                     ha = 'center',
#                     va = 'center',
#                     fontsize = 12,
#                     color = 'black'  # or a color that contrasts the bar
#                     )
#         cum_height += height




plt.show()
plt.close()

# %% Median MR, OHR, and OFP (% distribution bar chart)

totals = {}

'''
Calculating total OH for all PMF compounds:
'''
df_oh = common_cmpds_ppb.copy()
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
df_ofp = common_cmpds_ppb.copy()
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
    df_ofp[value] = df_ofp[value] * ref_data['MIR_best']
    
df_ofp = df_ofp.transpose()
df_ofp.reset_index(inplace = True, drop = True)
df_ofp = pd.DataFrame(data = df_ofp.median(), columns = ['median_ofp'])
units_ofp = '$ppb_v$'

'''
MR df too 
'''
df_mr = common_cmpds_ppb.copy()
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
gridspec = {'wspace' : 0.1, 'width_ratios' : [3, 1]}
fig, (ax) = plt.subplots(
    ncols = 1, nrows = 1,
    figsize = (16, 20), dpi = 300,
  #  gridspec_kw = gridspec
    )
df_pct.plot(ax = ax, 
               stacked = True,
                 kind = 'bar', 
                 subplots = False, 
                 sharex = True, 
                 use_index = True, 
                 width = 0.95,
          ylim = [0, 100],
          fontsize = 14,
          color = [cmpd_colors_dict[col] for col in df_pct.columns],
          edgecolor = 'white', 
          linewidth = 0.75,
          rot = 360
          )
ax.set_yticks(
    ticks = np.linspace(0, 100, 5),
    labels = np.linspace(0, 100, 5),
    fontweight = 'normal',
    fontsize = 14)
ax.set_xticks(
    ticks = range(len(df_pct)),
    labels = ['Mixing Ratio', 'OHR', 'OFP'],
    fontsize = 16,
    fontweight = 'normal'
    )
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.get_legend().remove()
# handles, labels = ax.get_legend_handles_labels()
# ax2.axis('off')  # hide the empty axis
# ax2.legend(
#     handles, labels,
#     loc='center left',      # center the legend in the subplot
#     ncol=1,            # number of columns in the legend
#     fontsize=12,
#     frameon=False      # no box around the legend
# )
ax.set_ylabel(
    'Percent Distribution (%)',
    fontsize = 16
    )


import textwrap



for bar_idx, (index, row) in enumerate(df_pct.iterrows()):  # bar_idx is 0,1,2,...
    row_numeric = row.astype(float)
    
    # Top 5 compounds
    top_cols = row_numeric.nlargest(5).index.tolist()
    
    # Compute cumulative heights
    cum_heights = row_numeric.cumsum()
    
    for col in top_cols:
        bottom = cum_heights[col] - row_numeric[col]
        mid_point = bottom + row_numeric[col] / 2

        # Wrap long compound names
        wrapped_label = '\n'.join(textwrap.wrap(col, width=25))
        num_lines = wrapped_label.count('\n') + 1
#        font_size = min(20, max(8, row_numeric[col] * 0.3 / num_lines + 8))

        ax.text(
            x=bar_idx,           # numeric position 0,1,2...
            y=mid_point,
            s=f'{wrapped_label} {row_numeric[col]:.1f}%',
            ha='center',
            va='center',
            fontsize= 12,
            color='black'
        )




plt.show()
plt.close()

# %% Median MR, OHR, and OFP (% distribution pie charts)

totals = {}

'''
Calculating total OH for all PMF compounds:
'''
df_oh = common_cmpds_ppb.copy()
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
df_ofp = common_cmpds_ppb.copy()
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
    df_ofp[value] = df_ofp[value] * ref_data['MIR_best']
    
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
df_mr = common_cmpds_ppb.copy()
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
ug_m3_totals = [8.2, '', 1.5]

for ax, col, unit, ug, l, s, t in zip(axes, columns, 
                                  units, ug_m3_totals, 
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
      #  labels = pie_values.index,
        wedgeprops = {'edgecolor' : 'black', 'linewidth' : 2},
        autopct = autopct_func,
        startangle = 10,
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

# %% ug/m3, Median MR, OHR, and OFP (% distribution pie charts)

totals = {}

'''
Calculating total OH for all PMF compounds:
'''

df_oh = common_cmpds_ppb.copy()
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
df_ofp = common_cmpds_ppb.copy()
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
    df_ofp_ug_m3[value] = df_ofp_ug_m3[value] * ref_data['MIR_best']
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
df_mr = common_cmpds_ppb.copy()
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

# %% ppb to ug/m3

df_mr = full_cmpd_dataset.copy()
meta_data_copy = meta_data.copy()

for df in [df_mr, meta_data_copy]:
    df.set_index('start_time', inplace = True)

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
df_compare = df_compare[['2-methyl-undecanal', 'carbitol', 'homosalate',
                         'component of texanol',
                         'ethylhexyl benzoate', 'diphenyl ether',
                         'allyl heptanoate', 'phenoxyisopropanol', 'benzyl acetate',
                         'benzyl alcohol', 'tricyclodecenyl acetate']]
df_compare = df_compare.transpose()

# %% Median OHR by TOD

# Calculating total OH for all PMF compounds:
df = common_cmpds_ppb.copy()
starttime = df['start_time'] # storing for later
hour = df['hour'] # storing for later

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

df = df.transpose() # flipping compounds back up to columns and putting back datetime column
df.insert(loc = 0, column = 'hour', value = hour, allow_duplicates = False)
#df.insert(loc = 0, column = 'datetime', value = starttime, allow_duplicates = False)

# df_OH_timeseries = pd.DataFrame()
# df_OH_timeseries.insert(loc = 0, column = 'datetime', value = starttime, allow_duplicates = False)

df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x >= 6
                                   else 
                                   'am2' if x <= 12 and x >= 10
                                   else
                                   'pm1' if x <= 16 and x > 13
                                   else 
                                   'pm2'
                                   )


#df.drop(columns = ['datetime', 'hour'], inplace = True)
df.drop(columns = ['hour'], inplace = True)
     
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

units = 'sec$^-$$^1$'
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
          ylim = [0, 1.25],
          fontsize = 14,
          color = [cmpd_colors_dict[col] for col in df.columns],
          edgecolor = 'white', 
          linewidth = 0.75,
          rot = 360
          )

ax.set_ylabel(
    f'Median OHR ({units})',
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
    ticks = np.linspace(0, 1.25, 5),
    labels = np.linspace(0, 1.25, 5),
    fontweight = 'normal',
    fontsize = 16)

#labeling the top 5 compounds making up each time block:
# for bar_idx, (index, row) in enumerate(df.iterrows()):
#     top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(5).index.tolist()
    
#     cum_height = 0
#     for col in df.columns:
#         height = row[col]
#         mid_point = cum_height + height / 2
#         if col in top_cols and height > 0:
            
#                     wrapped_label = '\n'.join(textwrap.wrap(col, width = 25))  # Adjust width as needed
                
#                     ax.text(
#                     x = bar_idx,
#                     y = mid_point,
#                     s = f'{wrapped_label}' + ': ' + f'{height:.2f}',
#                     ha = 'center',
#                     va = 'center',
#                     fontsize = 12,
#                     color = 'black'  # or a color that contrasts the bar
#                     )
#         cum_height += height


add_bar_totals(range(len(df)),1.27, df_total, units, 16, 'normal')

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.get_legend().remove()

plt.show()
plt.close()

# %% Median OFP by TOD

'''
Only using compounds with published MIR values 
'''

df = common_cmpds_ppb.copy()
hour = df['hour']
df.drop(columns = ['hour'], inplace = True)
df.set_index('start_time', inplace = True)
# want cmpds as row names and samples as columns:
df = df.transpose()

# want to only look at compounds included in the PMF input:
factor_df = profile_conc.copy() 
factor_df.set_index('compounds', inplace = True)
factor_cmpd_list = factor_df.index

df = df.loc[factor_cmpd_list] # filtering
cmpd_list = df.index # list of cmpds in the ground samples

# filtering the reference data for those cmpds
ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# currently, df = matrix of ppb values, samples as columns and cmpds as rows

# converting to OFP (ppb)
for value in df:
    df[value] = df[value] * ref_data['MIR_best']
    
df = df.transpose()
df.reset_index(inplace = True)

# now we have a matrix, with compounds as the column and each sample as the index
df.insert(loc = 0, column = 'hour', value = hour, allow_duplicates = False)
df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x >= 6
                                   else 
                                   'am2' if x <= 12 and x >= 10
                                   else
                                   'pm1' if x <= 16 and x > 13
                                   else 
                                   'pm2'
                                   )


df.drop(columns = ['start_time', 'hour'], inplace = True)
#df.drop(columns = ['hour'], inplace = True)
     
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

units = '$ppb_v$'
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
          ylim = [0, df_total.max() + 0.3],
          fontsize = 14,
          color = [cmpd_colors_dict[col] for col in df.columns],
          edgecolor = 'white', 
          linewidth = 0.75,
          rot = 360
          )

ax.set_ylabel(
    f'Median OFP ({units})',
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
    ticks = np.linspace(0, df_total.max() + 0.3, 5),
    labels = np.linspace(0, df_total.max() + 0.3, 5),
    fontweight = 'normal',
    fontsize = 16)

#labeling the top 5 compounds making up each time block:
# for bar_idx, (index, row) in enumerate(df.iterrows()):
#     top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(5).index.tolist()
    
#     cum_height = 0
#     for col in df.columns:
#         height = row[col]
#         mid_point = cum_height + height / 2
#         if col in top_cols and height > 0:
            
#                     wrapped_label = '\n'.join(textwrap.wrap(col, width = 25))  # Adjust width as needed
                
#                     ax.text(
#                     x = bar_idx,
#                     y = mid_point,
#                     s = f'{wrapped_label}' + ': ' + f'{height:.2f}',
#                     ha = 'center',
#                     va = 'center',
#                     fontsize = 12,
#                     color = 'black'  # or a color that contrasts the bar
#                     )
#         cum_height += height


add_bar_totals(range(len(df)),df_total.max()+0.5, df_total, units, 16, 'normal')

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.get_legend().remove()

plt.show()
plt.close()




# %% Total OH timeseries plot against factors TS 

# Calculating total OH for all PMF compounds:
df = common_cmpds_ppb.copy()
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

df = df.transpose() # flipping compounds back up to columns and putting back datetime column
df_OH_timeseries = pd.DataFrame()
df_OH_timeseries.insert(loc = 0, column = 'datetime', value = starttime, allow_duplicates = False)
df_OH_timeseries['Total OH Reactivity'] = df.sum(axis = 1, numeric_only = True) 
df_OH_timeseries.set_index('datetime', inplace = True)


df = factor_ts.copy() # no overwriting! 
df.set_index('datetime', inplace = True)
df.drop(columns = ['hour'], inplace = True)
    
mc = 10

fig, (ax1, ax2, ax4, ax3) = plt.subplots(
    nrows = 4,
    ncols = 1,
    sharex = True,
    figsize = (12, 9),
    dpi = 300)

axis1 = ['Factor 1', 'Factor 5']
axis2 = ['Factor 3', 'Factor 4']
axis4 = ['Factor 2']
  
# plotting factor solns
for col in df.columns:
    if col in axis1:
        plt1 = general_ts_plot(df, ax1, color = color_dict[col], 
                               linestyle = linestyle_dict[col],  mc = mc)
    elif col in axis2:
       plt2 = general_ts_plot(df, ax2, color = color_dict[col], 
                              linestyle = linestyle_dict[col], mc = mc)
    elif col in axis4:
       plt4 = general_ts_plot(df, ax4, color = color_dict[col], 
                              linestyle = linestyle_dict[col], mc = mc)

for col in df_OH_timeseries:
    plt3 = general_ts_plot(df_OH_timeseries, ax3, color = color_dict[col], 
                           linestyle = linestyle_dict[col], mc = mc)

# setting manual y-axis limits
ax1.set_ylim([-0.75, 12]) # Factors 1 and 5
ax1.set_yticks(
    np.linspace(0, 12, 4),
    np.linspace(0, 12, 4)
    )
ax2.set_ylim([-0.75, 9]) # Factors 3 and 4
ax2.set_yticks(
    np.linspace(0, 9, 4),
    np.linspace(0, 9, 4)
    )
ax4.set_ylim([-0.75, 9]) # Factor 2
ax4.set_yticks(
    np.linspace(0, 9, 4),
    np.linspace(0, 9, 4)
    )
ax3.set_ylim([-0.75, 12]) # OH
ax3.set_yticks(
    np.linspace(0, 12, 4),
    np.linspace(0, 12, 4)
    )

for ax in [ax1, ax2, ax3, ax4]:
    ax.legend(
        ncols = 1,
        loc = 'upper right',
        fontsize = 14
        )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

for ax in [ax1, ax2, ax4]:
    ax.set_ylabel(
        'Normalized\nConc.($ppb_v$)',
        fontsize = 12
        )
ax3.set_ylabel(
    'Total OH\nReactivity ($sec^{-1}$)',
    fontsize = 12)

ax3.set_xlim(df.index[0] - timedelta(hours = 6), df.index[-1] + timedelta(hours = 6)) 
ax3.xaxis.set_major_formatter(md.DateFormatter('%D'))
ax3.tick_params(axis='x', labelsize = 12)
fig.autofmt_xdate()

plt.show()
plt.close()

# %% Tracer ratios over time # delete from code now that there is no benzene

df = pd.DataFrame()

df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['t_b'] = full_cmpd_dataset['toluene'].copy() / full_cmpd_dataset['benzene'].copy()
df['lim_pine'] = full_cmpd_dataset['limonene'].copy()/ full_cmpd_dataset['a-pinene'].copy()
df.set_index('date_time', inplace = True)
    
axis1 = ['t_b']
axis2 = ['lim_pine']

gridspec = {'hspace' : 0.12, 'height_ratios' : [2, 0.08, 0.5, 2]}

fig, (ax1, ax100, ax2, ax3) = plt.subplots(
    nrows = 4,
    ncols = 1,
    sharex = True,
    figsize = (12, 8),
    dpi = 200,
    gridspec_kw = gridspec)

ax100.set_visible(False)
mc = 10

for col in df.columns:
    if col in axis1: 
        plt1_top = general_ts_plot(chosen_df = df, axis = ax1, 
                                   color = 'dodgerblue', linestyle  =  '--', mc = mc)

    if col in axis2:
        plt2 = general_ts_plot(chosen_df = df, axis = ax2, 
                               color = 'darkgreen', linestyle  =  '--', mc = mc)
        plt2 = general_ts_plot(chosen_df = df, axis = ax3, 
                               color = 'darkgreen', linestyle  =  '--', mc = 10)
        # plt3 = general_ts_plot(chosen_df = df, axis = ax4, 
        #                        color = 'darkgreen', linestyle  =  '--', mc = 10)

# y-axis, top plots
ax1.set_yticks(
    ticks = (np.linspace(0, df['t_b'].max(), 5)),
    labels = (np.linspace(0, df['t_b'].max(), 5)),
    size = 10,
    fontweight = 'bold'
    )
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.set_ylabel(
    'toluene : benzene',
    fontsize = 14,
    weight = 'bold'
    )

# y-axis, bottom plots
ax2.set_ylim(10, 14)
ax2.set_yticks(
    ticks = (np.linspace(10, 14, 3)),
            labels = (np.linspace(10, 14, 3)),
            size = 10,
            fontweight = 'bold'
    )
ax2.xaxis.set_visible(False)
ax2.spines.bottom.set_visible(False)
ax3.set_ylim(0, 3.2)
ax3.set_yticks(
    ticks = np.linspace(0, 3.2, 5),
            labels = np.linspace(0, 3.2, 5),
            size = 10,
            fontweight = 'bold'
    )
ax3.spines.top.set_visible(False)



# setting those horizonantal lines
# source code for the split axis: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = {'marker' : [(-1, -d), (1, d)], 'markersize' : 12,
              'linestyle' : "none", 'color' : 'k', 'mec' : 'k', 'mew' : 1, 'clip_on' : False}
ax2.plot([0, 1], [0, 0], transform = ax2.transAxes, **kwargs)
ax3.plot([0, 1], [1, 1], transform = ax3.transAxes, **kwargs)

for ax in [ax1, ax2, ax3]:
    ax.get_legend().remove()


for ax in [ax2, ax3]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# x-axis
ax3.set(
       xlim = (df.index[0] - timedelta(hours = 6), df.index[-1] + timedelta(hours = 6)), # adjusting x-lim 
       )
plt.xticks(fontsize = 12, fontweight = 'bold')
fig.autofmt_xdate()
ax3.xaxis.set_major_formatter(md.DateFormatter('%D'))
 

# adding ratio cut-off boxes
box_start = df.index.min()
box_end = df.index.max()

# traffic vs industry boxes
start_y_point = 0
height = 2
box_color = 'blue' # edit ! 
set_alpha = 0.1 # edit ! 
hatch = '/'
rect_box(box_start, box_end, ax1, hatch) # adding box
start_y_point = 2
height = df['t_b'].max()
box_color = 'lightgray' # edit ! 
set_alpha = 0.2 # edit ! 
hatch = '/'
rect_box(box_start, box_end, ax1, hatch) # adding box

# forest vs urban boxes
start_y_point = 0.68
height = 14
box_color = 'olive' # edit ! 
set_alpha = 0.1 # edit ! 
hatch = '/'
rect_box(box_start, box_end, ax2, hatch) # adding box
rect_box(box_start, box_end, ax3, hatch) # adding box
start_y_point = 0
height = 0.21
box_color = 'green' # edit ! 
set_alpha = 0.1 # edit ! 
hatch = '/'
rect_box(box_start, box_end, ax2, hatch) # adding box
rect_box(box_start, box_end, ax3, hatch) # adding box

# adding manual labels
x = md.date2num(datetime.datetime(2023, 7, 8, 18, 0, 0)) # have to have x as datetime
plt.text( # y-axis label for broken axis
    x = x, 
    y = 0.7, 
    s = 'limonene : α-pinene',
    rotation = 'vertical',
    fontsize = 14,
    fontweight = 'bold'
    )

# # adding labels for ratio boxes
label_fontsize = 10
x = md.date2num(datetime.datetime(2023, 8, 8, 12, 0, 0)) # have to have x as datetime
ax2.text( 
    x = x, 
    y = df['lim_pine'].max(), 
    s = 'Urban Regime',
    rotation = 'horizontal',
    fontsize = label_fontsize,
    fontweight = 'bold'
    )
ax3.text( 
    x = x, 
    y = df['lim_pine'].min() - 0.01, 
    s = 'Forested Regime',
    rotation = 'horizontal',
    fontsize = label_fontsize,
    fontweight = 'bold'
    )
ax1.text( 
    x = x, 
    y = df['t_b'].max() - 5, 
    s = 'Industrial Regime',
    rotation = 'horizontal',
    fontsize = label_fontsize,
    fontweight = 'bold'
    )
y = df['t_b'].min() - 7.5 # same y for both the label and the arrow
ax1.text( 
    x = x, 
    y = y, 
    s = 'Traffic Regime',
    rotation = 'horizontal',
    fontsize = label_fontsize,
    fontweight = 'bold'
    )
x = md.date2num(datetime.datetime(2023, 8, 7, 16, 0, 0))
ax1.text(
    x = x,
    y = y,
    s = r'$\hookleftarrow$',
    size = 18,
    rotation = 270
    )


plt.show()

# %% VCP Tracer Ratios - Not Finished

# df = pd.DataFrame()

# df['date_time'] = ground_cmpds_ppb['start_time_edt'].copy()
# df['D5_benz'] = ground_cmpds_ppb['decamethylcyclopentasiloxane'].copy() / ground_cmpds_ppb['benzene'].copy()
# df['benzyl_acetate_benzene'] = ground_cmpds_ppb['benzyl acetate'].copy()/ ground_cmpds_ppb['benzene'].copy()
# df['benzyl_alcohol_benzene'] = ground_cmpds_ppb['benzyl alcohol'].copy()/ ground_cmpds_ppb['benzene'].copy()
# df['acetophenone_benzene'] = ground_cmpds_ppb['acetophenone'].copy()/ ground_cmpds_ppb['benzene'].copy()
# df.set_index('date_time', inplace = True)
    
# axis1 = ['D5_benz']
# axis2 = ['benzyl_acetate_benzene']
# axis3 = ['benzyl_alcohol_benzene']
# axis4 = ['acetophenone_benzene']

# #gridspec = {'hspace' : 0.12, 'height_ratios' : [2, 0.08, 0.5, 2]}

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(
#     nrows = 4,
#     ncols = 1,
#     sharex = True,
#     figsize = (12, 8),
#     dpi = 200,
#     #gridspec_kw = gridspec
#     )


# mc = 10

# for col in df.columns:
#     if col in axis1: 
#         plt1_top = general_ts_plot(chosen_df = df, axis = ax1, 
#                                    color = 'dodgerblue', linestyle  =  '--', mc = mc)

#     if col in axis2:
#         plt2 = general_ts_plot(chosen_df = df, axis = ax2, 
#                                color = 'darkgreen', linestyle  =  '--', mc = mc)
#     if col in axis3:
#         plt3 = general_ts_plot(chosen_df = df, axis = ax3, 
#                                color = 'darkgreen', linestyle  =  '--', mc = 10)
        
#     if col in axis4: 
#         plt4 = general_ts_plot(chosen_df = df, axis = ax4, 
#                                color = 'darkgreen', linestyle  =  '--', mc = 10)

# # y-axis, top plots
# ax1.set_yticks(
#     ticks = (np.linspace(0, df['t_b'].max(), 5)),
#     labels = (np.linspace(0, df['t_b'].max(), 5)),
#     size = 10,
#     fontweight = 'bold'
#     )
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax1.set_ylabel(
#     'toluene : benzene',
#     fontsize = 14,
#     weight = 'bold'
#     )

# # y-axis, bottom plots
# ax2.set_ylim(10, 14)
# ax2.set_yticks(
#     ticks = (np.linspace(10, 14, 3)),
#             labels = (np.linspace(10, 14, 3)),
#             size = 10,
#             fontweight = 'bold'
#     )
# ax2.xaxis.set_visible(False)
# ax2.spines.bottom.set_visible(False)
# ax3.set_ylim(0, 3.2)
# ax3.set_yticks(
#     ticks = np.linspace(0, 3.2, 5),
#             labels = np.linspace(0, 3.2, 5),
#             size = 10,
#             fontweight = 'bold'
#     )
# ax3.spines.top.set_visible(False)


# %% Wind Roses w/ conc. and wind speed (one subplot, one colorbar)

df = factor_ts.copy()
dt_cols_ignore = ['datetime', 'hour']

cols_to_plot = [col for col in df.columns if col not in dt_cols_ignore]

num_bins = 30
bin_edges = np.linspace(0, 360, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
theta = np.deg2rad(bin_centers)

wind_spd = meta_data['avg_WindSpeed_m/s']
wind_dir = meta_data['median_WindDir']

# Compute global min/max of mean_voc for color scaling
all_mean_voc = []
for col in cols_to_plot:
    voc_conc = df[col]
    bin_indices = np.digitize(wind_dir, bin_edges) - 1
    mean_voc = np.zeros(num_bins)
    for i in range(num_bins):
        mask = bin_indices == i
        mean_voc[i] = np.mean(voc_conc[mask]) if np.any(mask) else np.nan
    all_mean_voc.append(mean_voc)
all_mean_voc = np.array(all_mean_voc)
vmin = np.nanmin(all_mean_voc)
vmax = np.nanmax(all_mean_voc)

# Create figure with 4 subplots
fig, axs = plt.subplots(3, 2, subplot_kw = {'projection':'polar'}, 
                        figsize = (10, 16), dpi = 300)
axs = axs.flatten()

for idx, col in enumerate(cols_to_plot[:5]):
    voc_conc = df[col]
    bin_indices = np.digitize(wind_dir, bin_edges) - 1

    mean_spd = np.zeros(num_bins)
    mean_voc = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = bin_indices == i
        mean_spd[i] = np.mean(wind_spd[mask]) if np.any(mask) else 0
        mean_voc[i] = np.mean(voc_conc[mask]) if np.any(mask) else np.nan

    bars = axs[idx].bar(theta, 
                        mean_spd, 
                        width=np.deg2rad(360/num_bins),
                        bottom=0.0,
                        color = plt.cm.ocean((mean_voc - vmin)/(vmax - vmin)),
                        edgecolor='black',
                        alpha=0.8)
    
    axs[idx].set_theta_zero_location('N')
    axs[idx].set_theta_direction(-1)
    axs[idx].set_rlabel_position(135)
    axs[idx].set_title(col, fontsize=18)

# Add a single colorbar for all subplots
sm = plt.cm.ScalarMappable(cmap='ocean', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
fig.colorbar(sm, ax = axs, orientation = 'vertical', 
             fraction = 0.05, pad = 0.1, label = 'Factor Mixing Ratio ($ppb_v$)')

plt.show()



# %% Ind. Tracer TS Plot (Not Finished)

# df = ground_cmpds_ppb.copy()
# df.rename(columns = {'start_time_edt' : 'datetime'}, inplace = True)
# df.set_index('datetime', inplace = True)
# df.drop(columns = 'hour', inplace = True)

    
# fig, (ax1) = plt.subplots(
#     nrows = 1,
#     ncols = 1,
#     sharex = True,
#     figsize = (8, 3),
#     dpi = 300)


# tracer1 = ['benzyl acetate']
# tracer_axes = {
#     ax1: tracer1,
#    }     

# mc = 12

# # plotting 
# for col in df.columns:
#     for ax, tracers in tracer_axes.items():
#         if col in tracers:
#             general_ts_plot(df, ax, color = cmpd_colors_dict[col], 
#                                      linestyle = 'solid',  mc = mc) 
# ax1.set_ylabel(
#     'Mixing Ratio ($ppb_v$)',
#     fontsize = 12,
#     fontweight = 'bold'
#     )

   
# ax1.set_yticks(np.linspace(df['benzyl acetate'].min() - (df['benzyl acetate'].min() * 0.15), 
#                             (df['benzyl acetate'].max() + (df['benzyl acetate'].max() * 0.15)), 5),
#                 labels = np.linspace(df['benzyl acetate'].min(), df['benzyl acetate'].max(), 5),
#                 fontsize = 13,
#                 ) 
# for axis in [ax1]:
#     axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))



# ax1.set_xticks(pd.date_range(start = df.index[0], end = df.index[-1], periods = 10),
#                labels = pd.date_range(start = df.index[0], end = df.index[-1], periods = 10),
#                fontsize = 12
#                 ) 


# ax1.yaxis.set_minor_locator(AutoMinorLocator())


# fig.autofmt_xdate()
# ax1.xaxis.set_major_formatter(md.DateFormatter('%D'))




# plt.show()
# plt.close()

# %% Box plots of tracer ratios

df = pd.DataFrame()

df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['D5_benz'] = full_cmpd_dataset['decamethylcyclopentasiloxane'].copy() / full_cmpd_dataset['benzene'].copy()
df['benzyl_acetate_benzene'] = full_cmpd_dataset['benzyl acetate'].copy()/ full_cmpd_dataset['benzene'].copy()
df['benzyl_alcohol_benzene'] = full_cmpd_dataset['benzyl alcohol'].copy()/ full_cmpd_dataset['benzene'].copy()
df['acetophenone_benzene'] = full_cmpd_dataset['acetophenone'].copy()/ full_cmpd_dataset['benzene'].copy()
df.set_index('date_time', inplace = True)
 
df = df.between_time('6:00','12:00') # only want morning values to match Coggons. et al (2021) enhancement ratios

medians = df.median()
df_stats = {
    'D5_benz' : medians.iloc[0],
    'benzyl_acetate_benzene' : medians.iloc[1],
    'benzyl_alcohol_benzene' : medians.iloc[2],
    'acetophenone_benzene' : medians.iloc[3],
    }

gridspec = {'wspace' : 0.4, 'width_ratios' : [1, 1, 1, 1]}

fig, axes = plt.subplots(
    nrows = 1,
    ncols = 4,
    figsize = (8, 4),
    dpi = 300,
    gridspec_kw = gridspec)

def boxplot(df, cmpd_ratio, ax, cmpd_name):
    df.boxplot(
        column = cmpd_ratio,
        ax = ax,
        grid = True,
        patch_artist = True,
        boxprops = {
            'facecolor' : cmpd_colors_dict[cmpd_name],
            'color' : 'black'
            },
        medianprops = {
            'color' : 'white'
            },
        whiskerprops = {
            'color' : 'black'
            }
        )
    ax.set_xticklabels([f'{cmpd_name}'])
    ax.set_ylim([-0.1, df[cmpd_ratio].max() + (df[cmpd_ratio].max() * 0.05)])
    ax.set_yticks(
        np.linspace(0, df[cmpd_ratio].max(), 8),
        np.linspace(0, df[cmpd_ratio].max(), 8)
        )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.text(
        x = 0.58,
        y = df_stats[cmpd_ratio] - (df_stats[cmpd_ratio] * 0.08),
        s = f'{df_stats[cmpd_ratio]:2.2f}', # total characters + precision for float objects
        bbox = {'facecolor': cmpd_colors_dict[cmpd_name],
                'alpha' : 0.5,
                'edgecolor' : None,
                'boxstyle' : 'round'},
        horizontalalignment = 'left',
        verticalalignment = 'bottom'
        )

    
# putting each variable on seperate subplot

boxplot( # calling boxplot plotting function
        df = df,
        cmpd_ratio = 'benzyl_acetate_benzene',
        ax = axes[0],
        cmpd_name = 'benzyl acetate'
        )
boxplot(
        df = df,
        cmpd_ratio = 'D5_benz',
        ax = axes[1],
        cmpd_name = 'decamethylcyclopentasiloxane'
        )
boxplot(
        df = df,
        cmpd_ratio = 'benzyl_alcohol_benzene',
        ax = axes[2],
        cmpd_name = 'benzyl alcohol'
        )
boxplot(
        df = df,
        cmpd_ratio = 'acetophenone_benzene',
        ax = axes[3],
        cmpd_name = 'acetophenone'
        )


# setting overall y label
axes[0].set_ylabel(
    'Compound ($ppb_v$) : Benzene ($ppb_v$)'
    )

plt.show()

# %% O3 vs NO2 Data, colored by [total VOC]

# setting up the trace gas data - only want daytime values
o3_copy = ground_o3.copy()
o3_copy.set_index('dt', inplace = True)
nox_copy = ground_nox.copy()
nox_copy.set_index('dt', inplace = True)

o3_daytime = o3_copy['O3_Corrected'].between_time('07:00', '19:00')
nox_daytime = nox_copy.between_time('07:00', '19:00')
trace_gas_daytime = pd.concat([o3_daytime, nox_daytime], axis = 1)


# setting up the dataframe of VOC values
df = full_cmpd_dataset.copy()
starttime = df['start_time'] # storing for later
df.drop(columns = ['start_time', 'hour'], inplace = True)
df['total_vocs'] = df.sum(axis = 1, numeric_only = True)
df.insert(loc = 0, column = 'start_time', value = starttime, allow_duplicates = False)
# now we have a matrix, with compounds as the column and each sample as the index
# adding in the metadata for the colorbar
df['o3'] = meta_data['avg_o3_ppb']
df['no2'] = meta_data['avg_no2_ppb']


# making the plot
fig, ax = plt.subplots(
    figsize = (8, 8),
    dpi = 300,
    )

cmap = plt.get_cmap('gist_earth_r') # choose the colormap
color_norm = plt.Normalize(df['total_vocs'].min(), df['total_vocs'].max())


ax.plot(trace_gas_daytime['NO2_Corrected_Flagged'], trace_gas_daytime['O3_Corrected'],
         label = 'All Daytime Data', linestyle = '', marker = 'o', 
         markeredgecolor = 'lightgray', markerfacecolor = 'white', alpha = 0.4,
         zorder = 1)
ax.scatter(df['no2'], df['o3'],
        label = 'Sampling Blocks', linestyle = '', marker = 'o',
        edgecolor = 'black', s = 60,
        color = cmap(color_norm(df['total_vocs'])),
        zorder = 2)

cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), ax = ax, pad = 0.03)
cbar.set_label(label = 'Total ∑VOCs ($ppb_v$)', 
                   fontsize = 11) # set label of the colorbar on the top plot

ax.set_ylabel('$O_3$ Mixing Ratio ($ppb_v$)', fontsize = 14)
ax.set_xlabel('$NO_2$ Mixing Ratio ($ppb_v$)', fontsize = 14)
ax.set_xlim([-5, 70])
ax.legend(loc = 'upper right')

plt.show()

# %% O3 vs NO2 Data, colored by OFP

# setting up the trace gas data - only want daytime values
o3_copy = ground_o3.copy()
o3_copy.set_index('dt', inplace = True)
nox_copy = ground_nox.copy()
nox_copy.set_index('dt', inplace = True)

o3_daytime = o3_copy['O3_Corrected'].between_time('07:00', '19:00')
nox_daytime = nox_copy.between_time('07:00', '19:00')
trace_gas_daytime = pd.concat([o3_daytime, nox_daytime], axis = 1)


# setting up the OFP 
df = full_cmpd_dataset.copy()
starttime = df['start_time'] # storing for later
df.drop(columns = ['start_time', 'hour'], inplace = True)
cmpd_list = df.columns
df = df.transpose()

ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# converting to OFP (ppb)
for value in df:
    df[value] = df[value] * ref_data['MIR_best']
df = df.transpose()
df.insert(loc = 0, column = 'start_time', value = starttime, allow_duplicates = False)
# now we have a matrix, with compounds as the column and each sample as the index
df['ofp_sum'] = df.sum(axis = 1, numeric_only = True)
# adding in the metadata for the colorbar
df['o3'] = meta_data['avg_o3_ppb']
df['no2'] = meta_data['avg_no2_ppb']


# making the plot
fig, ax = plt.subplots(
    figsize = (8, 8),
    dpi = 300,
    )

cmap = plt.get_cmap('gist_earth_r') # choose the colormap
color_norm = plt.Normalize(df['ofp_sum'].min(), df['ofp_sum'].max())


ax.plot(trace_gas_daytime['NO2_Corrected_Flagged'], trace_gas_daytime['O3_Corrected'],
         label = 'All Daytime Data', linestyle = '', marker = 'o', 
         markeredgecolor = 'lightgray', markerfacecolor = 'white', alpha = 0.4,
         zorder = 1)
ax.scatter(df['no2'], df['o3'],
        label = 'Sampling Blocks', linestyle = '', marker = 'o',
        edgecolor = 'black', s = 60,
        color = cmap(color_norm(df['ofp_sum'])),
        zorder = 2)
ax.set_xlim([-5, 70])

cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), ax = ax, pad = 0.03)
cbar.set_label(label = '∑OFP ($ppb_v$)', 
                   fontsize = 11) # set label of the colorbar on the top plot

ax.set_ylabel('$O_3$ Mixing Ratio ($ppb_v$)', fontsize = 14)
ax.set_xlabel('$NO_2$ Mixing Ratio ($ppb_v$)', fontsize = 14)

ax.legend(loc = 'upper right')

plt.show()

# %% O3 vs NO2 Data, colored by OHR

# setting up the trace gas data - only want daytime values
o3_copy = ground_o3.copy()
o3_copy.set_index('dt', inplace = True)
nox_copy = ground_nox.copy()
nox_copy.set_index('dt', inplace = True)

o3_daytime = o3_copy['O3_Corrected'].between_time('07:00', '19:00')
nox_daytime = nox_copy.between_time('07:00', '19:00')
trace_gas_daytime = pd.concat([o3_daytime, nox_daytime], axis = 1)


# setting up the OFP 
df = full_cmpd_dataset.copy()
starttime = df['start_time'] # storing for later
df.drop(columns = ['start_time', 'hour'], inplace = True)
cmpd_list = df.columns
df = df.transpose()

ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)
ref_data = ref_data.loc[cmpd_list] # filtering for cmpds included in PMF solution # need exact match for loc / list

# applying number density function to convert ppb to molc/cm3
for index, row in df.iterrows(): 
    for col in row.index:
     if pd.api.types.is_numeric_dtype(df[col]):
         df.at[index, col] = number_density(
             row[col],
         )

# converting to reactivities against OH (1/s)
for col in df:
    df[col] = df[col] * ref_data['k_oh_298']
    
df = df.transpose()
df.insert(loc = 0, column = 'start_time', value = starttime, allow_duplicates = False)
# now we have a matrix, with compounds as the column and each sample as the index
df['ohr_sum'] = df.sum(axis = 1, numeric_only = True)
# adding in the metadata for the colorbar
df['o3'] = meta_data['avg_o3_ppb']
df['no2'] = meta_data['avg_no2_ppb']


# making the plot
fig, ax = plt.subplots(
    figsize = (8, 8),
    dpi = 300,
    )

cmap = plt.get_cmap('gist_earth_r') # choose the colormap
color_norm = plt.Normalize(df['ohr_sum'].min(), df['ohr_sum'].max())


ax.plot(trace_gas_daytime['NO2_Corrected_Flagged'], trace_gas_daytime['O3_Corrected'],
         label = 'All Daytime Data', linestyle = '', marker = 'o', 
         markeredgecolor = 'lightgray', markerfacecolor = 'white', alpha = 0.4,
         zorder = 1)
ax.scatter(df['no2'], df['o3'],
        label = 'Sampling Blocks', linestyle = '', marker = 'o',
        edgecolor = 'black', s = 60,
        color = cmap(color_norm(df['ohr_sum'])),
        zorder = 2)
ax.set_xlim([-5, 70])

cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), ax = ax, pad = 0.03)
cbar.set_label(label = '∑OHR ($sec^-1$)', 
                   fontsize = 11) # set label of the colorbar on the top plot

ax.set_ylabel('$O_3$ Mixing Ratio ($ppb_v$)', fontsize = 14)
ax.set_xlabel('$NO_2$ Mixing Ratio ($ppb_v$)', fontsize = 14)

ax.legend(loc = 'upper right')

plt.show()

# %% NO2 vs VOC, colored by [O3]

# setting up the trace gas data - only want daytime values
o3_copy = ground_o3.copy()
o3_copy.set_index('dt', inplace = True)
nox_copy = ground_nox.copy()
nox_copy.set_index('dt', inplace = True)

o3_daytime = o3_copy['O3_Corrected'].between_time('07:00', '19:00')
nox_daytime = nox_copy.between_time('07:00', '19:00')
trace_gas_daytime = pd.concat([o3_daytime, nox_daytime], axis = 1)


# setting up the dataframe of VOC values
df = full_cmpd_dataset.copy()
starttime = df['start_time'] # storing for later
df.drop(columns = ['start_time', 'hour'], inplace = True)
df['total_vocs'] = df.sum(axis = 1, numeric_only = True)
df.insert(loc = 0, column = 'start_time', value = starttime, allow_duplicates = False)
# now we have a matrix, with compounds as the column and each sample as the index
# adding in the metadata for the colorbar
df['o3'] = meta_data['avg_o3_ppb']
df['no2'] = meta_data['avg_no2_ppb']



# making the plot
fig, ax = plt.subplots(
    figsize = (8, 8),
    dpi = 300,
    )

cmap = plt.get_cmap('cividis_r') # choose the colormap
color_norm = plt.Normalize(df['o3'].min(), df['o3'].max())

ax.scatter(df['total_vocs'], df['no2'],
        label = 'Sampling Blocks', linestyle = '', marker = 'o',
        edgecolor = 'black', s = 60,
        color = cmap(color_norm(df['o3'])),
        zorder = 2)

cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), ax = ax, pad = 0.03)
cbar.set_label(label = '$O_3$ Mixing Ratio ($ppb_v$)', 
                   fontsize = 11) # set label of the colorbar on the top plot

ax.set_xlabel('VOC Mixing Ratio ($ppb_v$)', fontsize = 14)
ax.set_ylabel('$NO_2$ Mixing Ratio ($ppb_v$)', fontsize = 14)
ax.set_ylim([0, 30])
ax.legend(loc = 'upper right')

plt.show()

# %% O3 vs NO2 Data, colored by [Factor]

# setting up the trace gas data - only want daytime values
o3_copy = ground_o3.copy()
o3_copy.set_index('dt', inplace = True)
nox_copy = ground_nox.copy()
nox_copy.set_index('dt', inplace = True)

o3_daytime = o3_copy['O3_Corrected'].between_time('07:00', '19:00')
nox_daytime = nox_copy.between_time('07:00', '19:00')
trace_gas_daytime = pd.concat([o3_daytime, nox_daytime], axis = 1)


# setting up the dataframe of VOC values
df = full_cmpd_dataset.copy()
df['Average Temperature'] = meta_data['avg_temp_c']
starttime = df['start_time'] # storing for later
df.drop(columns = ['start_time', 'hour'], inplace = True)
df['total_vocs'] = df.sum(axis = 1, numeric_only = True)
df.insert(loc = 0, column = 'start_time', value = starttime, allow_duplicates = False)
# now we have a matrix, with compounds as the column and each sample as the index
# adding in the metadata for the colorbar
df['o3'] = meta_data['avg_o3_ppb']
df['no2'] = meta_data['avg_no2_ppb']
df['Factor 1'] = factor_ts['Factor 1']
df['Factor 2'] = factor_ts['Factor 2']
df['Factor 3'] = factor_ts['Factor 3']
df['Factor 4'] = factor_ts['Factor 4']
df['Average Temperature'] = meta_data['avg_temp_c']

factors = ['Factor 1', 'Factor 2', 'Average Temperature', 'Factor 3', 'Factor 4']
colorbars = ['pink_r', 'pink_r',  'Reds', 'pink_r', 'pink_r']
units = ['$ppb_v$', '$ppb_v$', '$ppb_v$', 'C', '$ppb_v$', '$ppb_v$']
labels = ['A', 'B', 'C', 'D', 'E', 'F']

gridspec = {'wspace' : 0.15, 'hspace' : 0.12}

fig, axes = plt.subplots(
    ncols = 2,
    nrows = 3,
    figsize = (10, 14),
    dpi = 300,
    sharex = True,
    sharey = True,
    gridspec_kw = gridspec
    )

axes = axes.flatten()

# making the plot

for col, ax, cb, u in zip(factors, axes, colorbars, units):
               
    cmap = plt.get_cmap(cb) # choose the colormap
    color_norm = plt.Normalize(df[col].min(), df[col].max())
            
    ax.plot(trace_gas_daytime['NO2_Corrected_Flagged'], trace_gas_daytime['O3_Corrected'],
             label = 'All Daytime Data', linestyle = '', marker = 'o', 
             markeredgecolor = 'lightgray', markerfacecolor = 'white', alpha = 0.4,
             zorder = 1)
    ax.scatter(df['no2'], df['o3'],
            label = 'Sampling Blocks', linestyle = '', marker = 'o',
            edgecolor = 'black', s = 60,
            color = cmap(color_norm(df[col])),
            zorder = 2)
    
    cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), 
                        ax = ax, pad = 0.03)
    cbar.set_label(label = f'{col} ({u})', 
                       fontsize = 11) # set label of the colorbar on the top plot
    
    ax.set_xlim([-3, 70])
    ax.set_xticks(
        ticks = np.linspace(0, 70, 5),
        labels = np.linspace(0, 70, 5)
        )
    ax.set_ylim([-3, 100])
    ax.set_yticks(
        ticks = np.linspace(0, 100, 5),
        labels = np.linspace(0, 100, 5)
        )
       
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))


axes[2].text(
        x = -0.25, 
        y = 0.2,
        s = '$O_3$ Mixing Ratio ($ppb_v$)', 
        fontsize = 14,
        rotation = 90,
        transform =  axes[2].transAxes
        )
axes[4].text(
        x = 0.9, 
        y = -0.15,
        s = '$NO_2$ Mixing Ratio ($ppb_v$)', 
        fontsize = 14,
        transform = axes[4].transAxes
        )
axes[2].legend(
    loc = 'upper right'
    )

for f, i, l in zip(factors, [0, 1, 2, 3, 4, 5, 6], ['A', 'B', 'C', 'D', 'E', 'F']):
    axes[i].text(
        x = 0,
        y = 1.01,
        s = f'$\\bf{{{l}}}$  {f}',
        fontweight = 'normal',
        fontsize = 14,
        transform = axes[i].transAxes
    )

            
plt.show()

# %% O3 vs NO2 Data, colored by [Factor 1], [Factor 4], and temperature

# setting up the trace gas data - only want daytime values
o3_copy = ground_o3.copy()
o3_copy.set_index('dt', inplace = True)
nox_copy = ground_nox.copy()
nox_copy.set_index('dt', inplace = True)

o3_daytime = o3_copy['O3_Corrected'].between_time('07:00', '19:00')
nox_daytime = nox_copy.between_time('07:00', '19:00')
trace_gas_daytime = pd.concat([o3_daytime, nox_daytime], axis = 1)


# setting up the dataframe of VOC values
df = full_cmpd_dataset.copy()
df['Average Temperature'] = meta_data['avg_temp_c']
starttime = df['start_time'] # storing for later
df.drop(columns = ['start_time', 'hour'], inplace = True)
df['total_vocs'] = df.sum(axis = 1, numeric_only = True)
df.insert(loc = 0, column = 'start_time', value = starttime, allow_duplicates = False)
# now we have a matrix, with compounds as the column and each sample as the index
# adding in the metadata for the colorbar
df['o3'] = meta_data['avg_o3_ppb']
df['no2'] = meta_data['avg_no2_ppb']
df['Factor 1'] = factor_ts['Factor 1']
df['Factor 2'] = factor_ts['Factor 2']
df['Factor 3'] = factor_ts['Factor 3']
df['Factor 4'] = factor_ts['Factor 4']





colorbar_dfs = df[['Factor 2', 'Average Temperature', 'Factor 4']]
units = ['$ppb_v$', '° C', '$ppb_v$']
cbars = ['gist_earth_r', 'Reds', 'gist_earth_r']
labels = ['D', 'E', 'F']

# making the plot
gridspec = {'wspace' : 0.075}
fig, (ax1, ax2, ax3) = plt.subplots(
    ncols = 3,
    nrows = 1,
    sharex = True,
    sharey = True,
    figsize = (18, 8),
    dpi = 300,
    gridspec_kw = gridspec
    )

axes = [ax1, ax2, ax3]

for col, axis, unit, cbar in zip(colorbar_dfs, axes, units, cbars):
    colormap_column = colorbar_dfs[col] # edit!
    cmap = plt.get_cmap(cbar) 
    color_norm = plt.Normalize(colormap_column.min(), colormap_column.max())

    axis.plot(trace_gas_daytime['NO2_Corrected_Flagged'], trace_gas_daytime['O3_Corrected'],
             label = 'All Daytime Data', linestyle = '', marker = 'o', 
             markeredgecolor = 'lightgray', markerfacecolor = 'white', alpha = 0.4,
             markersize = 10,
             zorder = 1)
    axis.scatter(df['no2'], df['o3'],
            label = 'Sampling Blocks', linestyle = '', marker = 'o',
            edgecolor = 'black', 
            s = 150,
            color = cmap(color_norm(colormap_column)),
            zorder = 2)
    # setting the axes
    axis.set_xlim([-5, 70])
    axis.set_xticks(
        ticks = np.linspace(0, 65, 5),
        labels =  np.linspace(0, 65, 5),
        fontsize = 18
        )
    axis.set_ylim(-5, 95)
    axis.set_yticks(
        ticks = np.linspace(0, 85, 5),
        labels = np.linspace(0, 85, 5),
        fontsize = 18
        )
    
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axis.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # setting up the colorbars
    cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, 
                                          cmap = cmap), 
                        location = 'top',
                        shrink = 0.7,
                        ax = axis, 
                        pad = 0.05)
    # setting ticks on the colobar
    cbar_bounds = np.linspace(colormap_column.min(), colormap_column.max(), 4)
    # rounding to the nearest integar for those ticks
    cbar_ticks = [0 if i <= 0 else int(i) for i in cbar_bounds]
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize = 16)
    cbar.set_label(label = f'{col} ({unit})', 
                       fontsize = 20,
                       labelpad = 15
                       )    

# plot specific settings outside of the loop
ax1.set_ylabel('$O_3$ Mixing Ratio ($ppb_v$)', 
               fontsize = 22)
ax2.text(x = 10,
         y = -18,
         s =  '$NO_2$ Mixing Ratio ($ppb_v$)', 
         fontsize = 22)

# setting up manual legend:
from matplotlib.lines import Line2D
patches = [Line2D([], [],
                    marker = 'o',
                    linestyle = '',
                    markeredgecolor = 'black',
                    markerfacecolor = 'white',
                    label = 'All Daytime Data',
                    ),
           Line2D([], [],
                    marker = 'o',
                    linestyle = '',
                    markeredgecolor = 'black',
                    markerfacecolor = 'dimgray',
                    label = 'Sampling Blocks'
                    )]
ax3.legend(handles = patches, fontsize = 16)

for ax, l in zip([ax1, ax2, ax3], ['D', 'E', 'F']):
    ax.text(
        x = 0,
        y = 1.01,
        s = f'{l}',
        fontweight = 'bold',
        fontsize = 16,
        transform = ax.transAxes
    )

plt.show()


# %% Homosalate vs %RH vs Temp vs Ethylhexyl benzoate

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

fig, ax = plt.subplots(
    figsize = (8, 8), 
    dpi = 300)
simple_corr_plot(axis = ax, 
                 x = 'Homosalate', 
                 y = 'Ethylhexyl benzoate', 
                 df = df, 
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

fig, ax = plt.subplots(
    figsize = (8, 8), 
    dpi = 300)
simple_corr_plot(axis = ax, 
                 x = 'Median Relative Humidity', 
                 y = 'Homosalate', 
                 df = df, 
                 color = '#013A63', 
                 linecolor = 'black',
                 lw = 2,
                 linestyle = 'solid',
                 sci_note = False,
                 show_slope = False,
                 show_r = True,
                 show_both = False,
                 label_fontsize = 16, 
                 tick_fontsize = 14, 
                 fontweight = 'normal', 
                 markersize = 10, 
                 legend_fontsize = 14, 
                 loc = 'upper left', 
                 x_unit = '%', 
                 y_unit = '$ppb_v$'
                 )

ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.show()

fig, ax = plt.subplots(
    figsize = (8, 8), 
    dpi = 300)
simple_corr_plot(axis = ax, 
                 x = 'Median Temperature', 
                 y = 'Homosalate', 
                 df = df, 
                 sci_note = False,
                 show_slope = False,
                 show_r = True,
                 show_both = False,
                 color = '#801818', 
                 linecolor = 'black',
                 lw = 2,
                 linestyle = 'solid',
                 label_fontsize = 14, 
                 tick_fontsize = 12, 
                 fontweight = 'normal', 
                 markersize = 10, 
                 legend_fontsize = 12, 
                 loc = 'upper left', 
                 x_unit = 'C', 
                 y_unit = '$ppb_v$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.show()

fig, ax = plt.subplots(
    figsize = (8, 8), 
    dpi = 300)
simple_corr_plot(axis = ax, 
                 x = 'Ethylhexyl benzoate', 
                 y = 'Phenoxyisopropanol', 
                 df = df, 
                 sci_note = False,
                 show_slope = False,
                 show_r = True,
                 show_both = False,
                 color = 'violet', 
                 linecolor = 'black',
                 lw = 2,
                 linestyle = 'solid',
                 label_fontsize = 14, 
                 tick_fontsize = 12, 
                 fontweight = 'normal', 
                 markersize = 10, 
                 legend_fontsize = 12, 
                 loc = 'upper left', 
                 x_unit = '$ppb_v$', 
                 y_unit = '$ppb_v$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.show()


# %% Weekday vs weekend of MT

d = {'datetime' : full_cmpd_dataset['start_time'],
     'a-pinene' : full_cmpd_dataset['a-pinene'],
     'limonene'  : full_cmpd_dataset['limonene'],
     'hour' : full_cmpd_dataset['hour']
     }
df = pd.DataFrame(data = d)


weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] # set days of week in a weekday
weekend = ['Saturday', 'Sunday']  # set days in a weekend
grouping = ['weekend', 'weekday'] # we want to sort data based on weekday or weekend status

ignore = ['datetime', 'hour', 'dow', 'weekday_weekend'] 

df['dow'] = df['datetime'].dt.day_name()
df['weekday_weekend'] = df['dow'].apply(lambda x: 'weekend' if x in weekend else 'weekday') 

df_weekday = df[df['weekday_weekend'] == 'weekday'] # boolean mask to select weekdays
df_weekend = df[df['weekday_weekend'] == 'weekend'] # boolean mask to select weekends


gridspec = {'hspace' : 0.12, 'height_ratios' : [2, 0.08, 0.5, 2]}

# making the plot
fig, (ax1, ax100, ax2, ax3) = plt.subplots(
    nrows = 4,
    ncols = 1,
    sharex = True,
    figsize = (6, 8),
    dpi = 200,
    gridspec_kw = gridspec)

ax100.set_visible(False)
ax100.tick_params(labelbottom = False)

for ax in (ax1,):
    grouped_weekday = df_weekday.groupby('hour')['a-pinene'].describe()
    grouped_weekend = df_weekend.groupby('hour')['a-pinene'].describe()
    
    # plotting weekday day
    ax.plot(grouped_weekday.index, grouped_weekday['50%'], 
            #label = f'{col} - Weekday',
             linestyle = '--', 
              marker = 'o',
             color = 'darkolivegreen',
             label = 'Weekday')
    
    ax.fill_between(grouped_weekday.index, grouped_weekday['25%'], grouped_weekday['75%'],
                    alpha = 0.3, color = 'darkolivegreen')
    # plotting weekend data
    ax.plot(grouped_weekend.index, grouped_weekend['50%'], 
            #label = f'{col} - Weekend',
             linestyle = '--', marker = 'o',
             label = 'Weekend',
             color = 'chocolate')
    ax.fill_between(grouped_weekend.index, grouped_weekend['25%'], grouped_weekend['75%'],
                    alpha = 0.3, color = 'chocolate')
    ax.legend().set_visible(False)
    ax.set_title(
        label = 'a-pinene',
        loc = 'right',
        x = 0.97,
        y = 0.88,
        )

for ax in (ax2, ax3):
    grouped_weekday = df_weekday.groupby('hour')['limonene'].describe()
    grouped_weekend = df_weekend.groupby('hour')['limonene'].describe()
    
    # plotting weekday day
    ax.plot(grouped_weekday.index, grouped_weekday['50%'], 
            #label = f'{col} - Weekday',
             linestyle = '--', 
              marker = 'o',
             color = 'darkolivegreen',
             label = 'Weekday')
    
    ax.fill_between(grouped_weekday.index, grouped_weekday['25%'], grouped_weekday['75%'],
                    alpha = 0.3, color = 'darkolivegreen')
    # plotting weekend data
    ax.plot(grouped_weekend.index, grouped_weekend['50%'], 
            #label = f'{col} - Weekend',
             linestyle = '--', marker = 'o',
             label = 'Weekend',
             color = 'chocolate')
    ax.fill_between(grouped_weekend.index, grouped_weekend['25%'], grouped_weekend['75%'],
                    alpha = 0.3, color = 'chocolate')
    ax.legend().set_visible(False)
    ax2.set_title(
        label = 'limonene',
        loc = 'right',
        x = 0.97,
        y = 0.5,
        )
    # setting up the split axis
    ax2.set_ylim(0.04, 0.1)   # upper range
    ax3.set_ylim(0, 0.03)       # lower range
    ax2.spines.bottom.set_visible(False)
    ax3.spines.top.set_visible(False)

    # setting those horizonantal lines
    # source code for the split axis: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = {'marker' : [(-1, -d), (1, d)], 'markersize' : 12,
                  'linestyle' : "none", 'color' : 'k', 'mec' : 'k', 'mew' : 1, 'clip_on' : False}
    ax2.plot([0, 1], [0, 0], transform = ax2.transAxes, **kwargs)
    ax3.plot([0, 1], [1, 1], transform = ax3.transAxes, **kwargs)
       
         
# setting overall plot title and legend

# create manual symbols for legend
weekday_point = Line2D([0], [0], 
                       label = 'Weekday', 
                       marker = 'o', 
                       linestyle = '--',
                       markersize = 10, 
                       markeredgecolor = 'darkolivegreen', 
                       markerfacecolor = 'darkolivegreen',
                       color = 'darkolivegreen'
                       )
weekend_point = Line2D([0], [0], 
                       label = 'Weekend', 
                       marker = 'o', 
                       linestyle = '--',
                       markersize = 10, 
                       markeredgecolor = 'chocolate', 
                       markerfacecolor = 'chocolate',
                       color = 'chocolate'
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
    x = -0.18,
    y = -0.7, 
    s = 'Median Mixing Ratio ($ppb_v$)',
    rotation = 'vertical',
    transform = ax1.transAxes,
    fontsize = 14
    )
plt.xlabel('Hour of Day', # setting x-axis label on bottom plot only
           fontsize = 14)

plt.show()
plt.close()

# %%  MT vs temperature plots

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
ax2.set_ylim([-0.001, df['limonene'].max() + 0.005])
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
    num_dec = 0.3,
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
ax1.set_ylim([-0.001, df['a-pinene'].max()+.01])
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
    axis = ax3,
    x = 'Average Temperature',
    y = 'limonene',
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
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 10,
    legend_fontsize = legend_fontsize,
    loc = 'upper right',
    x_unit = 'C',
    y_unit = '$ppb_v$'
    )
ax3.set_ylim([0, df['limonene'].max() + 0.005])
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
            transform = ax.transAxes
            )
    ax1.text(
            x = 0.55,
            y = 1.03,
            s = '12:00 - 18:00',
            fontsize = 14,
            color = 'darkblue',
            transform = ax.transAxes
            )
for ax in [ax2, ax4]:
    ax.text(
            x = 0.15,
            y = 1.03,
            s = 'limonene',
            color = '#C0B141',
            fontsize = 14,
            transform = ax.transAxes
            )
    ax.text(
            x = 0.55,
            y = 1.03,
            s = 'a-pinene',
            color = '#9B287B',
            fontsize = 14,
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

# %% Averaging All Back Trajectories over 12 hours

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
                     color = 'black', fontsize = 24) 
      elif float(f'{value:.2}') <= - 0.3 : # adjusting the value:# will change the sigfigs
         ax.text(j, i, f'{value:.2f}', ha = 'center', va = 'center', 
                 color = 'black', fontsize = 24)
      
    

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

plt.show()
plt.close()

# # averaging back traj. by sampling block
# def avg_by_GC_times(hour): # applying function to the dt column within the dataframe
#     if 6 <= hour < 9:
#         return 'Morning'
#     if 12 <= hour < 10:
#         return 'Mid-Morning'
#     if 13 <= hour < 16:
#         return 'Early Evening'
#     if 16 <= hour < 22:
#         return 'Evening'
    
# backtraj_filtered.reset_index(inplace = True, drop = False)
# backtraj_filtered['date'] = backtraj_filtered['date_arrival'].dt.date
# backtraj_filtered['time_block'] = backtraj_filtered['date_arrival'].dt.hour.map(avg_by_GC_times)
# backtraj_filtered_timeblocks = backtraj_filtered.groupby(['date', 'time_block']).mean().reset_index()



# %% Averaged BackTraj Analyses (EBF): 6 + 12 hours 

# 6  hour
path = drive_folder + '\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\meta_data\\'
file = 'Back_Traj_Summaries_6hr_20260117.csv'

backtraj =  pd.read_csv(path + file, 
                           header = 0,
                           parse_dates = ['date_arrival']
                           )
backtraj['date_arrival'] = pd.to_datetime(backtraj['date_arrival'])
hour = 6
backtraj.set_index('date_arrival', inplace = True)
backtraj.drop(columns = ['date_arrival.r',], inplace = True)

# prepping the factor ts df
df = factor_ts.copy()
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
      if float(f'{value:.2}') >= 0.2 : # only displaying R higher than 0.5
         ax.text(j, i, f'{value:.1}', ha = 'center', va = 'center', 
                 color = 'black', fontsize = 24) # adjusting the value:# will change the sigfigs
      if float(f'{value:.2}') <= - 0.2 : 
         ax.text(j, i, f'{value:.1}', ha = 'center', va = 'center', 
                 color = 'black', fontsize = 24)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.title(f'Averaged Back Trajectories:{hour} hours',
          fontsize = 24)

plt.show()
plt.close()



# 12 hour
path = drive_folder + '\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\meta_data\\'
file = 'Back_Traj_Summaries_12hr_20260117.csv'

backtraj =  pd.read_csv(path + file, 
                           header = 0,
                           parse_dates = ['date_arrival']
                           )
backtraj['date_arrival'] = pd.to_datetime(backtraj['date_arrival'])
hour = 12
backtraj.set_index('date_arrival', inplace = True)
backtraj.drop(columns = ['date_arrival.r',], inplace = True)

# prepping the factor TS df
df = factor_ts.copy()
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
      if float(f'{value:.2}') >= 0.2 : # only displaying R higher than 0.5
         ax.text(j, i, f'{value:.2}', ha = 'center', va = 'center', color = 'black', fontsize = 20) # adjusting the value:# will change the sigfigs
      if float(f'{value:.2}') <= - 0.2 : 
         ax.text(j, i, f'{value:.1}', ha = 'center', va = 'center', color = 'black', fontsize = 20)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.title(f'Averaged Back Trajectories: {hour} hours',
          fontsize = 24)

plt.show()
plt.close()

# %% Factor Contributions of Total gC (not finished)

df = profile_conc.copy()
df.set_index('compounds', inplace = True)

refs = cmpd_refs.copy()
refs = refs[['compound_name', 'mw_g_mol', 'formula']]    
refs.set_index('compound_name', inplace = True)

# filtering refs for PMF compounds
refs = refs[refs.index.isin(df.index)]

# check to make sure the indicies match
[x for x in list(df.index.unique()) if x not in list(refs.index.unique())]

concat = pd.concat([df, refs], axis = 1)

# need molar ratio of carbon : compound
def num_carbon(dataframe, formula_col = "formula"):
    c_molar_ratio_list = []

    for x in dataframe[formula_col]:
        counts = parse_formula(x)
        num_c = counts.get("C", 0)
        c_molar_ratio_list.append(num_c)

    dataframe["num_c"] = c_molar_ratio_list
    return dataframe

# applying num_c function
concat = num_carbon(concat)

# now calculating g/m3 from ppb
carbon = Formula('C').mass # storing atmomic mass of C

# %% MT vs toluene by TOD

d = {
     'a-pinene' : full_cmpd_dataset['a-pinene'],
     'limonene' : full_cmpd_dataset['limonene'],
     'D5-siloxane' : full_cmpd_dataset['decamethylcyclopentasiloxane'],
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

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    ncols = 2,
    nrows = 2,
    figsize = (10.5, 6), 
    dpi = 300)

simple_corr_plot(
    axis = ax1,
    x = 'toluene',
    y = 'limonene',
    sci_note = False,
    df = df_am,
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
    loc = 'upper left',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )
simple_corr_plot(
    axis = ax2,
    x = 'toluene',
    y = 'limonene',
    df = df_pm,
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
    x = 'toluene',
    y = 'a-pinene',
    sci_note = False,
    df = df_am,
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
    loc = 'upper left',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )
simple_corr_plot(
    axis = ax4,
    x = 'toluene',
    y = 'a-pinene',
    df = df_pm,
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

ax1.set(xticklabels = [],
        xlabel = None)
ax2.set(xticklabels = [],
        xlabel = None)
ax2.set(ylabel = None)
ax4.set(ylabel = None)

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

# %% MT vs D5 by TOD

d = {
     'a-pinene' : full_cmpd_dataset['a-pinene'],
     'limonene' : full_cmpd_dataset['limonene'],
     'D5-siloxane' : full_cmpd_dataset['decamethylcyclopentasiloxane'],
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

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    ncols = 2,
    nrows = 2,
    figsize = (10.5, 6), 
    dpi = 300)

simple_corr_plot(
    axis = ax1,
    x = 'D5-siloxane',
    y = 'limonene',
    df = df_am,
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
    loc = 'upper left',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )
simple_corr_plot(
    axis = ax2,
    x = 'D5-siloxane',
    y = 'limonene',
    df = df_pm,
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
    x = 'D5-siloxane',
    y = 'a-pinene',
    df = df_am,
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
    loc = 'upper left',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )
simple_corr_plot(
    axis = ax4,
    x = 'D5-siloxane',
    y = 'a-pinene',
    df = df_pm,
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

ax1.set(xticklabels = [],
        xlabel = None)
ax2.set(xticklabels = [],
        xlabel = None)
ax2.set(ylabel = None)
ax4.set(ylabel = None)

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

 # %% MT vs camphor vs toluene by TOD

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

# %% toluene vs camphor by TOD


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
gridspec = {'hspace' : 0.3, 
            'wspace' : 0.18}
fig, ((ax1, ax2)) = plt.subplots(
    ncols = 2,
    nrows = 1,
    figsize = (10.5, 6), 
    dpi = 300,
    gridspec_kw = gridspec)
# ax1 = uppper left
simple_corr_plot(
    axis = ax1,
    x = 'toluene',
    y = 'camphor',
    df = df_am,
    sci_note = True,
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
    x = 'toluene',
    y = 'camphor',
    df = df_pm,
    sci_note = True,
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

# %% L:P vs camphor by TOD


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
gridspec = {'hspace' : 0.3, 
            'wspace' : 0.18}
fig, ((ax1, ax2)) = plt.subplots(
    ncols = 2,
    nrows = 1,
    figsize = (10.5, 6), 
    dpi = 300,
    gridspec_kw = gridspec)
# ax1 = uppper left
simple_corr_plot(
    axis = ax1,
    x = 'limonene : a-pinene',
    y = 'camphor',
    df = df_am,
    sci_note = True,
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
    x_unit = '',
    y_unit = '$ppb_v$'
    )
# ax1 = uppper right
simple_corr_plot(
    axis = ax2,
    x = 'limonene : a-pinene',
    y = 'camphor',
    df = df_pm,
    sci_note = True,
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
# %% Limonene vs. a-pinene (two OLS plots)

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


# %% Lim:aPine over time

df = pd.DataFrame()

df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['lim_pine'] = full_cmpd_dataset['limonene'].copy()/ full_cmpd_dataset['a-pinene'].copy()
df.set_index('date_time', inplace = True)

# setting the ind sizes of the two subplots + the white space inbetween
gridspec = {'hspace' : 0.12, 'height_ratios' : [1, 2]}

fig, (ax1, ax2) = plt.subplots(
    nrows = 2,
    ncols = 1,
    sharex = True,
    figsize = (10, 5),
    dpi = 300,
    gridspec_kw = gridspec
    )

# adding lines to represent campaign average ratios 
ax2.hlines(y = 1.68, 
           xmin = df.index[0] - timedelta(hours = 6), 
           xmax = df.index[-1] + timedelta(hours = 6),
           linewidth = 1,
           color = 'darkgray',
           linestyle = '--')
ax2.hlines(y = 0.50, 
           xmin = df.index[0] - timedelta(hours = 6), 
           xmax = df.index[-1] + timedelta(hours = 6),
           linewidth = 1,
           color = 'darkgray',
           linestyle = 'solid')

mc = 10
for col in df.columns:
    general_ts_plot(chosen_df = df, axis = ax1, 
                           color = 'darkgreen', linestyle  =  '--', mc = mc)
    general_ts_plot(chosen_df = df, axis = ax2, 
                           color = 'darkgreen', linestyle  =  '--', mc = 10)

# y-axis, top subplot
ax1.set_ylim(2.55, 14)
ax1.set_yticks(
    ticks = (np.linspace(3, 14, 4)),
            labels = (np.linspace(3, 14, 4)),
            size = 12,
            fontweight = 'normal'
    )
ax1.xaxis.set_visible(False)
ax1.spines.bottom.set_visible(False)
# y-axis, bottom subplot
ax2.set_ylim(0, 2)
ax2.set_yticks(
    ticks = np.linspace(0, 2, 5),
            labels = np.linspace(0, 2, 5),
            size = 12,
            fontweight = 'normal'
    )
ax2.spines.top.set_visible(False)


# setting those horizonantal lines
# source code for the split axis: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = {'marker' : [(-1, -d), (1, d)], 'markersize' : 12,
              'linestyle' : "none", 'color' : 'k', 'mec' : 'k', 'mew' : 1, 'clip_on' : False}
ax1.plot([0, 1], [0, 0], transform = ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform = ax2.transAxes, **kwargs)

for ax in [ax1, ax2]:
    ax.get_legend().remove()

for ax in [ax1, ax2]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# x-axis
ax2.set(
       xlim = (df.index[0] - timedelta(hours = 6), df.index[-1] + timedelta(hours = 6)), # adjusting x-lim 
       )
plt.xticks(fontsize = 14, fontweight = 'normal')
fig.autofmt_xdate()
ax2.xaxis.set_major_formatter(md.DateFormatter('%D'))

# adding ratio cut-off boxes
box_start = df.index.min()
box_end = df.index.max()

# forest vs urban boxes
start_y_point = 0.68
height = 14
box_color = 'olive' # edit ! 
set_alpha = 0.1 # edit ! 
hatch = '/'
rect_box(box_start, box_end, ax1, hatch) # adding box
rect_box(box_start, box_end, ax2, hatch) # adding box
start_y_point = 0
height = 0.21
box_color = 'green' # edit ! 
set_alpha = 0.1 # edit ! 
hatch = '/'
rect_box(box_start, box_end, ax1, hatch) # adding box
rect_box(box_start, box_end, ax2, hatch) # adding box

# adding manual labels
x = md.date2num(datetime.datetime(2023, 7, 8, 6, 30, 0)) # have to have x as datetime
ax2.text( # y-axis label for broken axis
    x = x, 
    y = 0.65, 
    s = 'limonene : α-pinene',
    rotation = 'vertical',
    fontsize = 16,
    fontweight = 'normal'
    )

# # adding labels for ratio boxes
label_fontsize = 10
x = md.date2num(datetime.datetime(2023, 8, 7, 7, 0, 0)) # have to have x as datetime
ax1.text( 
    x = x, 
    y = df['lim_pine'].max() - 0.05, 
    s = 'Urban Regime',
    rotation = 'horizontal',
    fontsize = label_fontsize,
    fontweight = 'bold'
    )
ax2.text( 
    x = x, 
    y = df['lim_pine'].min() - 0.01, 
    s = 'Forested Regime',
    rotation = 'horizontal',
    fontsize = label_fontsize,
    fontweight = 'bold'
    )

# x = md.date2num(datetime.datetime(2023, 8, 7, 16, 0, 0))
# ax1.text(
#     x = x,
#     y = y,
#     s = r'$\hookleftarrow$',
#     size = 18,
#     rotation = 270
#     )


plt.show()
plt.close()

# %% Lim:aPine over time, colored by TOD

df = pd.DataFrame()

df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['lim_pine'] = full_cmpd_dataset['limonene'].copy()/ full_cmpd_dataset['a-pinene'].copy()
df['hour'] = full_cmpd_dataset['hour']
df.set_index('date_time', inplace = True)

# setting the ind sizes of the two subplots + the white space inbetween
gridspec = {'hspace' : 0.12, 'height_ratios' : [1, 2]}

fig, (ax1, ax2) = plt.subplots(
    nrows = 2,
    ncols = 1,
    sharex = True,
    figsize = (10, 5),
    dpi = 300,
    gridspec_kw = gridspec
    )

# adding lines to represent campaign average ratios 
ax2.hlines(y = 1.68, 
           xmin = df.index[0] - timedelta(hours = 6), 
           xmax = df.index[-1] + timedelta(hours = 6),
           linewidth = 1,
           color = 'darkgray',
           linestyle = '--')
ax2.hlines(y = 0.50, 
           xmin = df.index[0] - timedelta(hours = 6), 
           xmax = df.index[-1] + timedelta(hours = 6),
           linewidth = 1,
           color = 'darkgray',
           linestyle = 'solid')

mc = 75
cmap = plt.cm.gist_earth_r
norm = mcolors.Normalize(vmin  = df['hour'].min(), vmax = df['hour'].max())

ax1.plot(df.index, df['lim_pine'], 
         color='black', linewidth=1.5, alpha=0.6, zorder=1)

ax2.plot(df.index, df['lim_pine'], 
         color='black', linewidth=1.5, alpha=0.6, zorder=1)

sc1 = ax1.scatter(df.index, df['lim_pine'],
            c = df['hour'], cmap = cmap, norm = norm,
            s = mc, zorder = 3,
            edgecolor = 'black'
            )
sc2 = ax2.scatter(df.index, df['lim_pine'],
            c = df['hour'], cmap = cmap, norm = norm,
            s = mc, zorder = 3,
            edgecolor = 'black'
            )

# y-axis, top subplot
ax1.set_ylim(2.55, 14)
ax1.set_yticks(
    ticks = (np.linspace(3, 14, 4)),
            labels = (np.linspace(3, 14, 4)),
            size = 12,
            fontweight = 'normal'
    )
ax1.xaxis.set_visible(False)
ax1.spines.bottom.set_visible(False)
# y-axis, bottom subplot
ax2.set_ylim(0, 2)
ax2.set_yticks(
    ticks = np.linspace(0, 2, 5),
            labels = np.linspace(0, 2, 5),
            size = 12,
            fontweight = 'normal'
    )
ax2.spines.top.set_visible(False)


# setting those horizonantal lines
# source code for the split axis: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = {'marker' : [(-1, -d), (1, d)], 'markersize' : 12,
              'linestyle' : "none", 'color' : 'k', 'mec' : 'k', 'mew' : 1, 'clip_on' : False}
ax1.plot([0, 1], [0, 0], transform = ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform = ax2.transAxes, **kwargs)



for ax in [ax1, ax2]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# x-axis
ax2.set(
       xlim = (df.index[0] - timedelta(hours = 6), df.index[-1] + timedelta(hours = 6)), # adjusting x-lim 
       )
plt.xticks(fontsize = 14, fontweight = 'normal')
fig.autofmt_xdate()
ax2.xaxis.set_major_formatter(md.DateFormatter('%D'))

# adding ratio cut-off boxes
box_start = df.index.min()
box_end = df.index.max()

# forest vs urban boxes
start_y_point = 0.68
height = 14
box_color = '#580C1F' # edit ! 
set_alpha = 0.25 # edit ! 
hatch = '/'
rect_box(box_start, box_end, ax1, hatch) # adding box
rect_box(box_start, box_end, ax2, hatch) # adding box
start_y_point = 0
height = 0.21
box_color = '#BBD1EA' # edit ! 
set_alpha = 0.6 # edit ! 
hatch = '/'
rect_box(box_start, box_end, ax1, hatch) # adding box
rect_box(box_start, box_end, ax2, hatch) # adding box

# adding manual labels
x = md.date2num(datetime.datetime(2023, 7, 8, 6, 30, 0)) # have to have x as datetime
ax2.text( # y-axis label for broken axis
    x = x, 
    y = 0.65, 
    s = 'limonene : α-pinene',
    rotation = 'vertical',
    fontsize = 16,
    fontweight = 'normal'
    )

# # adding labels for ratio boxes
label_fontsize = 10
x = md.date2num(datetime.datetime(2023, 8, 7, 0, 0, 0)) # have to have x as datetime
ax1.text( 
    x = x, 
    y = df['lim_pine'].max() - 0.05, 
    s = 'Urban Regime',
    rotation = 'horizontal',
    fontsize = label_fontsize,
    fontweight = 'normal'
    )
ax2.text( 
    x = x, 
    y = df['lim_pine'].min() - 0.01, 
    s = 'Forested Regime',
    rotation = 'horizontal',
    fontsize = label_fontsize,
    fontweight = 'normal'
    )

# x = md.date2num(datetime.datetime(2023, 8, 7, 16, 0, 0))
# ax1.text(
#     x = x,
#     y = y,
#     s = r'$\hookleftarrow$',
#     size = 18,
#     rotation = 270
#     )

cbar = fig.colorbar(sc2, ax=[ax1, ax2], pad=0.02)
cbar.set_label('Hour of Day')
cbar.set_ticks(np.arange(df['hour'].min(), df['hour'].max(), 3))

plt.show()
plt.close()

# %% lim:apine morning vs night

df = pd.DataFrame()
df = full_cmpd_dataset[['limonene', 'a-pinene']].copy()
df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['hour'] = full_cmpd_dataset['hour'].copy()

df.set_index('date_time', inplace = True)
df.dropna(inplace = True)

df_am = df.between_time('05:00', '12:00')
df_pm = df.between_time('12:01', '23:00')

# defining general variables 
loc = 'lower right'
legend_fontsize = 20
label_fontsize = 24
tick_fontsize = 20
fontweight = 'normal'
markersize = 14

x_variable_name = 'a-pinene' # edit !
y_variable_name = 'limonene' # edit !
color = 'darkgreen'

gridspec = {'wspace' : 0.2}
fig, (ax1, ax2) = plt.subplots(
    ncols = 2,
    nrows = 1,
    sharey = False,
    sharex = False,
    figsize = (16, 7), 
    dpi = 300,
    gridspec_kw = gridspec
    )

# first plot: morning data
simple_corr_plot(ax1, 
                 x=x_variable_name, 
                 y=y_variable_name,
                 df=df_am,
                 sci_note = False,
                 show_slope = False,
                 show_r = False,
                 show_both = True,
                 color = 'orange',
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
ax1.set_title('06:00 - 12:00',
              fontsize = legend_fontsize
              )

# first plot: afternoon data
simple_corr_plot(ax2, 
                 x=x_variable_name, 
                 y=y_variable_name,
                 df=df_pm,
                 sci_note = False,
                 show_slope = False,
                 show_r = False,
                 show_both = True,
                 color='darkblue',
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
ax2.set_xlabel('')
ax2.set_ylabel('limonene ($ppb_v$)',
               fontweight = fontweight,
               fontsize = label_fontsize,
               labelpad = 9)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_title('12:00 - 18:00',
              fontsize = legend_fontsize
              )

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

# %% lim:pine verus temperature, grouped by TOD

d = {
     'a-pinene' : full_cmpd_dataset['a-pinene'],
     'limonene' : full_cmpd_dataset['limonene'],
     'D5-siloxane' : full_cmpd_dataset['decamethylcyclopentasiloxane'],
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

cmap = plt.cm.gist_earth_r
# cutting off the top and bottom 1% of colors b/c they are too dark/light
cmap = mcolors.LinearSegmentedColormap.from_list(
    'earth_r_trim', cmap(np.linspace(0.1, 0.9, 256))
)
norm = mcolors.Normalize(vmin  = df['hour'].min(), vmax = df['hour'].max())

pm_color = '#133378' # pm color!
am_color = '#CBA57F' # am color!

fig, ax = plt.subplots(
    figsize = (8, 8), 
    dpi = 300)
simple_corr_plot(
    axis = ax,
    x = 'Average Temperature',
    y = 'limonene : a-pinene',
    df = df_am,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'white',
    linecolor = am_color,
    lw = 2,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 2,
    legend_fontsize = 12,
    loc = 'upper left',
    x_unit = 'C',
    y_unit = ''
    )
simple_corr_plot(
    axis = ax,
    x = 'Average Temperature',
    y = 'limonene : a-pinene',
    df = df_pm,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'white',
    linecolor = pm_color,
    lw = 2,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 2,
    legend_fontsize = 12,
    loc = 'upper left',
    x_unit = 'C',
    y_unit = ''
    )
ax.scatter(
    x = df['Average Temperature'],
    y = df['limonene : a-pinene'],
    c = df['hour'], cmap = cmap, norm = norm,
    s = 100,
    edgecolor = 'black',
    zorder = 2 
    )

ax.set_ylim([0, df['limonene : a-pinene'].max()+.25])
ax.set_xlim([df['Average Temperature'].min() - 0.2, df['Average Temperature'].max() + 0.2])
ax.set_yticks(
    ticks = np.linspace(0, df['limonene : a-pinene'].max(), 5),
    labels = np.linspace(0, df['limonene : a-pinene'].max(), 5)
    )
ax.set_xticks(
    ticks = np.linspace(df['Average Temperature'].min(), df['Average Temperature'].max(), 5),
    labels = np.linspace(df['Average Temperature'].min(), df['Average Temperature'].max(), 5)
    )
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.set_ylabel(
    'limonene : a-pinene',
    fontsize = 14
    )

cbar = fig.colorbar(sc2, ax=ax, pad=0.02)
cbar.set_label('Hour of Day')
cbar.set_ticks(np.arange(df['hour'].min(), df['hour'].max(), 3))
ax.text(
        x = 0.05,
        y = 0.95,
        s = '06:00 - 12:00',
        fontsize = 14,
        color = am_color,
        transform = ax.transAxes, # relative to axis, not data
        )
ax.text(
        x = 0.39,
        y = 0.95,
        s = '12:00 - 18:00',
        fontsize = 14,
        color = pm_color,
        transform = ax.transAxes, # relative to axis, not data
        )
ax.legend (loc = 'upper left',
           ncols = 2,
           fontsize = 14,
           bbox_to_anchor = (0, 0.95),
           frameon = False
           )

plt.show()
plt.close()

# %% Diel plot of lim:apine

df = pd.DataFrame()

df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['lim_pine'] = full_cmpd_dataset['limonene'].copy()/ full_cmpd_dataset['a-pinene'].copy()
df['hour'] = full_cmpd_dataset['hour']
df.set_index('date_time', inplace = True)

grouped = df.groupby('hour')['lim_pine'].describe()

# function to add rectangular patches to non-TS graphs
def points_rect_box(start, end, axis, color):
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
    ax.add_patch(box)


fig, ax = plt.subplots()     
ax.plot(grouped.index, grouped['50%'], label = 'Median',
        color = 'darkgreen',
         linestyle = '--', marker = 'o')
ax.fill_between(grouped.index, grouped['25%'], grouped['75%'],
                alpha = 0.3, color = 'green', label = 'IQR')
  
   
ax.set_xlabel('Hour of Day')
ax.set_ylabel('limonene : α-pinene')


# adding lines at urban-rural cutoffs
ax.hlines(y = 0.21, 
           xmin = grouped.index.min(), 
           xmax = grouped.index.max(),
           linewidth = 1,
           color = 'darkgray',
           linestyle = '--')
ax.hlines(y = 0.68, 
           xmin = grouped.index.min(), 
           xmax = grouped.index.max(),
           linewidth = 1,
           color = 'darkgray',
           linestyle = '--')


# adding ratio cut-off boxes
box_start = grouped.index.min() - 0.5
box_end = grouped.index.max() + 0.5

# forest vs urban boxes
start_y_point = 0.68
height = 1.7
box_color = 'olive' # edit ! 
set_alpha = 0.1 # edit ! 
hatch = '/'
points_rect_box(start = box_start, end = box_end, axis = ax,
                color = box_color)

start_y_point = 0
height = 0.21
box_color = 'green' # edit ! 
set_alpha = 0.1 # edit ! 
hatch = '/'
points_rect_box(start = box_start, end = box_end, axis = ax,
                color = box_color)

ax.set_xlim([grouped.index.min() - 0.5, grouped.index.max() + 0.5])
ax.set_ylim([0, grouped['75%'].max()])

ax.legend()

plt.show()
plt.close()  

# %% Combining HOD terpene data into one plot: option 1

df = pd.DataFrame()
df = full_cmpd_dataset[['limonene', 'a-pinene']].copy()
df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['hour'] = full_cmpd_dataset['hour'].copy()
df['lim_pine'] = df['limonene']/ df['a-pinene']
df['average_temp'] = meta_data['avg_temp_c']
df.set_index('date_time', inplace = True)

# grouping by hour for diel plot
grouped = df.groupby('hour')['lim_pine'].describe()

# splitting am and pm for OLS
df_copy = df.copy()
df_copy.dropna(inplace = True)
df_am = df_copy.between_time('05:00', '12:00')
df_pm = df_copy.between_time('12:01', '23:00')

# defining general variables 
loc = 'lower right'
legend_fontsize = 20
label_fontsize = 24
tick_fontsize = 20
fontweight = 'normal'
markersize = 14

cmap = plt.cm.gist_earth_r
# cutting off the top and bottom 1% of colors b/c they are too dark/light
cmap = mcolors.LinearSegmentedColormap.from_list(
    'earth_r_trim', cmap(np.linspace(0.1, 0.9, 256))
)
norm = mcolors.Normalize(vmin  = df['hour'].min(), vmax = df['hour'].max())
mc = 500

# setting the ind sizes of the two subplots + the white space inbetween
gridspec = {
    #'hspace' : 0.3, 
           # 'wspace' : 0.2,
            'height_ratios' : [2, 1, 2],
            'width_ratios' : [1, 1, 1]
            }

fig = plt.figure(figsize = (28, 18), dpi = 300)
gs = fig.add_gridspec(3, 3, **gridspec)

# top row (three subplots)
ax3 = fig.add_subplot(gs[0, 1]) # zero-based indexing
ax4 = fig.add_subplot(gs[0, 0])
ax5 = fig.add_subplot(gs[0, 2])
# bottom row (one subplot spanning both columns)
ax1 = fig.add_subplot(gs[1, :])
ax2 = fig.add_subplot(gs[2, :], sharex = ax1)

# now tweak positions manually
pos3 = ax3.get_position()
pos4 = ax4.get_position()
pos5 = ax5.get_position()
pos1 = ax1.get_position()
pos2 = ax2.get_position()
# increase spacing between top row and ax1
shift = 0.025  # how much space to add
ax3.set_position([pos3.x0, pos3.y0 + shift, pos3.width, pos3.height])
ax4.set_position([pos4.x0, pos4.y0 + shift, pos4.width, pos4.height])
ax5.set_position([pos5.x0, pos5.y0 + shift, pos5.width, pos5.height])

# decrease spacing between ax1 and ax2
ax1.set_position([pos1.x0, pos1.y0 - (shift*0.03), pos1.width, pos1.height])
ax2.set_position([pos2.x0, pos2.y0 + (shift), pos2.width, pos2.height])



# ax3 = diel plot
ax3.plot(grouped.index, grouped['50%'], label = 'Median',
         color = 'black',
         marker = 'o', 
         markersize = markersize)
ax3.fill_between(grouped.index, grouped['25%'], grouped['75%'],
                alpha = 0.2, color = 'gray', label = 'IQR')
ax3.set_xlabel('Hour of Day',
               fontsize = label_fontsize,
               labelpad = 10)
ax3.set_ylabel('limonene : α-pinene',
               fontsize = label_fontsize,
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
box_start = grouped.index.min() - 0.5
box_end = grouped.index.max() + 0.5

# forest vs urban boxes and labels
ax3.axhspan(
    ymin = 0.68,
    ymax = 1.7,
    color = 'olive',
    alpha = 0.1,
    hatch = '/'
    )
ax3.axhspan(
    ymin = 0,
    ymax = 0.21,
    color = 'green',
    alpha = 0.1,
    hatch = '/'
    )
ax3.text( 
    x = 12.5, 
    y = 1.45, 
    s = 'Urban Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'bold'
    )
ax3.text( 
    x = 12.5, 
    y = 0.05, 
    s = 'Forested Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'bold'
    )

ax3.set_xlim([grouped.index.min() - 0.5, grouped.index.max() + 0.5])
ax3.set_xticks(
    ticks = np.linspace(grouped.index.min(), grouped.index.max(), 5),
    labels = np.linspace(grouped.index.min(), grouped.index.max(), 5),
    fontsize = tick_fontsize
    )
ax3.set_ylim([0, grouped['75%'].max()])
ax3.set_yticks(
    ticks = np.linspace(0, grouped['75%'].max(), 5),
    labels = np.linspace(0, grouped['75%'].max(), 5),
    fontsize = tick_fontsize
    )
ax3.legend(fontsize = legend_fontsize,
           loc = 'upper left',
           bbox_to_anchor = (0.1, 1.01))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))



# ax4 = ols plot

x_variable_name = 'a-pinene' # edit !
y_variable_name = 'limonene' # edit !

pm_color = '#133378' # pm color!
am_color = '#CBA57F' # am color!

simple_corr_plot(ax4, 
                 x=x_variable_name, 
                 y=y_variable_name,
                 df=df_pm,
                 sci_note = False,
                 show_slope = False,
                 show_r = False,
                 show_both = True,
                 color=pm_color,
                 linecolor = pm_color,
                 lw = 4,
                 linestyle = 'dashed',
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
                 sci_note = False,
                 show_slope = False,
                 show_r = False,
                 show_both = True,
                 color=am_color,
                 linecolor = am_color,
                 lw = 4,
                 linestyle = 'dashed',
                 label_fontsize=label_fontsize, 
                 tick_fontsize=tick_fontsize,
                 fontweight=fontweight, 
                 markersize=markersize,
                 legend_fontsize=legend_fontsize, 
                 loc=loc,
                 x_unit = 'ppbv', 
                 y_unit='ppbv'
                 )
ax4.text(
    x = 0.55,
    y = 1.02,
    s = '13:00 - 18:00',
    fontsize = tick_fontsize,
    color = pm_color,
    transform = ax4.transAxes, # relative to axis, not data
    )
ax4.text(
    x = 0.05,
    y = 1.02,
    s = '06:00 - 12:00',
    fontsize = tick_fontsize,
    color = '#A47342',
    transform = ax4.transAxes, # relative to axis, not data,
    )
ax4.set_xlabel('α-pinene ($ppb_v$)',
               fontweight = fontweight,
               fontsize = label_fontsize,
               labelpad = 10)
ax4.set_ylabel('limonene ($ppb_v$)',
               fontweight = fontweight,
               fontsize = label_fontsize,
               labelpad = 20)
ax4.set_ylim([-0.005, df['limonene'].max() + 0.01])
ax4.set_yticks(
    ticks = np.linspace(0, df['limonene'].max() + 0.01, 5),
    labels = np.linspace(0, df['limonene'].max() + 0.01, 5),
    fontsize = tick_fontsize
    )
ax4.set_xlim([0, df['a-pinene'].max() + 0.01])
ax4.set_xticks(
    ticks = np.linspace(0, df['a-pinene'].max(), 5),
    labels = np.linspace(0, df['a-pinene'].max(), 5),
    fontsize = tick_fontsize
    )
ax4.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax4.legend(
    loc = 'upper center',
    fontsize = legend_fontsize,
    ncols = 2,
    frameon = False,
    reverse = True
    )


# ax5 = temp dependency


simple_corr_plot(
    axis = ax5,
    x = 'average_temp',
    y = 'lim_pine',
    df = df_am,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'white',
    linecolor = am_color,
    lw = 4,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 2,
    legend_fontsize = legend_fontsize,
    loc = 'upper left',
    x_unit = 'C',
    y_unit = ''
    )
simple_corr_plot(
    axis = ax5,
    x = 'average_temp',
    y = 'lim_pine',
    df = df_pm,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'white',
    linecolor = pm_color,
    lw = 4,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 2,
    legend_fontsize = 12,
    loc = 'upper left',
    x_unit = 'C',
    y_unit = ''
    )
ax5.scatter(
    x = df['average_temp'],
    y = df['lim_pine'],
    c = df['hour'], cmap = cmap, norm = norm,
    s = mc,
    edgecolor = 'black',
    zorder = 2 
    )
ax5.text(
    x = 0.45,
    y = 1.02,
    s = '12:00 - 18:00',
    fontsize = tick_fontsize,
    color = pm_color,
    transform = ax5.transAxes, # relative to axis, not data
    )
ax5.text(
    x = 0.05,
    y = 1.02,
    s = '06:00 - 12:00',
    fontsize = tick_fontsize,
    color = '#A47342',
    transform = ax5.transAxes, # relative to axis, not data,
    )

ax5.set_ylim([-0.4, 13])
ax5.set_xlim([df['average_temp'].min() - 0.5, df['average_temp'].max() + 0.5])
ax5.set_xticks(
    ticks = np.linspace(df['average_temp'].min(), df['average_temp'].max(), 5),
    labels = np.linspace(df['average_temp'].min(), df['average_temp'].max(), 5),
    fontsize = tick_fontsize
    )
ax5.set_yticks(
    ticks = np.linspace(0, df['lim_pine'].max() + 0.6, 5),
    labels = np.linspace(0, df['lim_pine'].max() + 0.6, 5),
    fontsize = tick_fontsize
    )
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax5.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax5.set_ylabel(
    'limonene : α-pinene',
    fontsize = label_fontsize,
    labelpad = 0
    )
ax5.set_xlabel(
    'Average Temperature (C)',
    fontsize = label_fontsize,
    labelpad = 10
    )
ax5.legend (loc = 'upper left',
           ncols = 2,
           fontsize = legend_fontsize,
     #      bbox_to_anchor = (0, 0.95),
           frameon = False
           )


# bottom plot

# adding lines to represent campaign average ratios 
ax2.hlines(y = 0.25,
       #    y = df_pm['lim_pine'].median(), 
           xmin = df.index[0] - timedelta(hours = 6), 
           xmax = df.index[-1] + timedelta(hours = 6),
           linewidth = 2.5,
           color = pm_color,
           linestyle = '--'
           )
ax2.hlines(y = 1.70,
    #       y = df_am['lim_pine'].median(), 
           xmin = df.index[0] - timedelta(hours = 6), 
           xmax = df.index[-1] + timedelta(hours = 6),
           linewidth = 2.5,
           color = am_color,
           linestyle = '--'
           )

ax1.plot(df.index, df['lim_pine'], 
         color = 'black', linewidth = 2.5, 
         mew = 10,
        # alpha = 0.6,  # marker edge width
         zorder = 1)
ax2.plot(df.index, df['lim_pine'], 
         color = 'black', linewidth = 2.5, 
         mew = 10, # marker edge width
      #   alpha = 0.6, 
         zorder = 1)
sc1 = ax1.scatter(df.index, df['lim_pine'],
            c = df['hour'], cmap = cmap, norm = norm,
            s = mc, zorder = 3,
            linewidth = 1.8, # marker edge width
            edgecolor = 'black'
            )
sc2 = ax2.scatter(df.index, df['lim_pine'],
            c = df['hour'], cmap = cmap, norm = norm,
            s = mc, zorder = 3,
            linewidth = 1.8, # marker edge width
            edgecolor = 'black'
            )

# y-axis, top subplot
ax1.set_ylim(2.55, 14)
ax1.set_yticks(
    ticks = (np.linspace(3, 14, 4)),
            labels = (np.linspace(3, 14, 4)),
            size = tick_fontsize,
            fontweight = 'normal'
    )
ax1.xaxis.set_visible(True)
ax1.tick_params(labelbottom = False)  # hide labels but keep ticks active
ax1.spines.bottom.set_visible(False)
# y-axis, bottom subplot
ax2.set_ylim(0, 2)
ax2.set_yticks(
    ticks = np.linspace(0, 2, 5),
            labels = np.linspace(0, 2, 5),
            size = tick_fontsize,
            fontweight = 'normal'
    )
ax2.spines.top.set_visible(False)

# setting those horizonantal lines
# source code for the split axis: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = {'marker' : [(-1, -d), (1, d)], 'markersize' : 12,
              'linestyle' : "none", 'color' : 'k', 'mec' : 'k', 'mew' : 1, 'clip_on' : False}
ax1.plot([0, 1], [0, 0], transform = ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform = ax2.transAxes, **kwargs)

for ax in [ax1, ax2]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# x-axis
ax2.set(xlim = (df.index[0] - timedelta(hours = 6), 
                df.index[-1] + timedelta(hours = 6)), # adjusting x-lim 
       )
ax2.tick_params(axis = 'x', labelsize = label_fontsize)
#fig.autofmt_xdate()
ax2.xaxis.set_major_formatter(md.DateFormatter('%D'))

# forest vs urban boxes
for ax in [ax1, ax2]:
    ax.axhspan(
        ymin = 0.68,
        ymax = 14,
        color = 'olive',
        alpha = 0.1,
        hatch = '/'
        )
for ax in [ax1, ax2]:
    ax.axhspan(
        ymin = 0,
        ymax = 0.21,
        color = 'green',
        alpha = 0.1,
        hatch = '/'
        )

# adding manual labels
x = md.date2num(datetime.datetime(2023, 7, 9, 7, 2, 0)) # have to have x as datetime
ax2.text( # y-axis label for broken axis
    x = x, 
    y = 0.65, 
    s = 'limonene : α-pinene',
    rotation = 'vertical',
    fontsize = label_fontsize,
    fontweight = 'normal'
    )

# # adding labels for ratio boxes
x = md.date2num(datetime.datetime(2023, 8, 9, 0, 0, 0)) # have to have x as datetime
ax1.text( 
    x = x, 
    y = df['lim_pine'].max() - 0.08, 
    s = 'Urban Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'bold'
    )
ax2.text( 
    x = x, 
    y = df['lim_pine'].min() - 0.01, 
    s = 'Forested Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'bold'
    )

# get vertical span for colorbar
pos_top = ax5.get_position()
pos_bottom = ax2.get_position()
cbar_bottom = pos_bottom.y0
cbar_top = pos_top.y1
cbar_height = cbar_top - cbar_bottom
# place colorbar axis to the right
cax = fig.add_axes([pos5.x1 + 0.015,   # small gap to the right of rightmost subplot
                    cbar_bottom,
                    0.015,             # colorbar width
                    cbar_height
                    ])
cbar = fig.colorbar(sc2, cax = cax, pad = 0.01)

cbar.set_label('Hour of Day')
cbar.set_ticks(np.arange(df['hour'].min(), df['hour'].max() + 1, 3))
cbar.ax.tick_params(labelsize = tick_fontsize)  
cbar.set_label(
    'Hour of Day',
    fontsize = label_fontsize,
    rotation = 270,
    labelpad = 25    
    )

plt.show()
plt.close()  

# %% Combining HOD terpene data into one plot: option 2

df = pd.DataFrame()
df = full_cmpd_dataset[['limonene', 'a-pinene']].copy()
df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['hour'] = full_cmpd_dataset['hour'].copy()
df['lim_pine'] = df['limonene']/ df['a-pinene']
df['average_temp'] = meta_data['avg_temp_c']
df.set_index('date_time', inplace = True)

# grouping by hour for diel plot
grouped = df.groupby('hour')['lim_pine'].describe()

# splitting am and pm for OLS
df_copy = df.copy()
df_copy.dropna(inplace = True)
df_am = df_copy.between_time('05:00', '12:00')
df_pm = df_copy.between_time('12:00', '23:00')

# defining general variables 
loc = 'lower right'
legend_fontsize = 20
label_fontsize = 18
tick_fontsize = 16
fontweight = 'normal'
markersize = 14

cmap = plt.cm.gist_earth_r
# cutting off the top and bottom 1% of colors b/c they are too dark/light
cmap = mcolors.LinearSegmentedColormap.from_list(
    'earth_r_trim', cmap(np.linspace(0.1, 0.9, 256))
)
norm = mcolors.Normalize(vmin  = df['hour'].min(), vmax = df['hour'].max())
mc = 250

# setting the ind sizes of the two subplots + the white space inbetween
gridspec = {
    #'hspace' : 0.3, 
           # 'wspace' : 0.2,
            'height_ratios' : [2, 1, 2],
            'width_ratios' : [1, 1]
            }

fig = plt.figure(figsize = (18, 12), dpi = 100)
gs = fig.add_gridspec(3, 2, **gridspec)

# top row (three subplots)
ax3 = fig.add_subplot(gs[0, 0]) # zero-based indexing
ax5 = fig.add_subplot(gs[0, 1])
# bottom row (one subplot spanning both columns)
ax1 = fig.add_subplot(gs[1, :])
ax2 = fig.add_subplot(gs[2, :], sharex = ax1)

# now tweak positions manually
pos3 = ax3.get_position()
pos5 = ax5.get_position()
pos1 = ax1.get_position()
pos2 = ax2.get_position()
# increase spacing between top row and ax1
shift = 0.025  # how much space to add
ax3.set_position([pos3.x0, pos3.y0 + shift, pos3.width, pos3.height])
ax5.set_position([pos5.x0, pos5.y0 + shift, pos5.width, pos5.height])

# decrease spacing between ax1 and ax2
ax1.set_position([pos1.x0, pos1.y0 - (shift*0.03), pos1.width, pos1.height])
ax2.set_position([pos2.x0, pos2.y0 + (shift), pos2.width, pos2.height])



# ax3 = diel plot
ax3.plot(grouped.index, grouped['50%'], label = 'Median',
         color = 'black',
         marker = 'o', 
         markersize = markersize)
ax3.fill_between(grouped.index, grouped['25%'], grouped['75%'],
                alpha = 0.2, color = 'gray', label = 'IQR')
ax3.set_xlabel('Hour of Day',
               fontsize = label_fontsize,
               labelpad = 10)
ax3.set_ylabel('limonene : α-pinene',
               fontsize = label_fontsize,
               labelpad = 20)

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
box_start = grouped.index.min() - 0.5
box_end = grouped.index.max() + 0.5

# forest vs urban boxes and labels
ax3.axhspan(
    ymin = 0.68,
    ymax = 1.7,
    color = 'olive',
    alpha = 0.1,
    hatch = '/'
    )
ax3.axhspan(
    ymin = 0,
    ymax = 0.21,
    color = 'green',
    alpha = 0.1,
    hatch = '/'
    )
ax3.text( 
    x = 13.5, 
    y = 1.45, 
    s = 'Urban Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'bold'
    )
ax3.text( 
    x = 13.5, 
    y = 0.05, 
    s = 'Forested Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'bold'
    )

ax3.set_xlim([grouped.index.min() - 0.5, grouped.index.max() + 0.5])
ax3.set_xticks(
    ticks = np.linspace(grouped.index.min(), grouped.index.max(), 5),
    labels = np.linspace(grouped.index.min(), grouped.index.max(), 5),
    fontsize = tick_fontsize
    )
ax3.set_ylim([0, grouped['75%'].max()])
ax3.set_yticks(
    ticks = np.linspace(0, grouped['75%'].max(), 5),
    labels = np.linspace(0, grouped['75%'].max(), 5),
    fontsize = tick_fontsize
    )
ax3.legend(fontsize = legend_fontsize,
           loc = 'upper left',
           bbox_to_anchor = (0.1, 1.01))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))




# ax5 = temp dependency


simple_corr_plot(
    axis = ax5,
    x = 'average_temp',
    y = 'lim_pine',
    df = df_am,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'white',
    linecolor = am_color,
    lw = 4,
    linestyle = 'solid',
    label_fontsize = label_fontsize,
    tick_fontsize = tick_fontsize,
    fontweight = 'normal',
    markersize = 2,
    legend_fontsize = legend_fontsize,
    loc = 'upper left',
    x_unit = 'C',
    y_unit = ''
    )
simple_corr_plot(
    axis = ax5,
    x = 'average_temp',
    y = 'lim_pine',
    df = df_pm,
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'white',
    linecolor = pm_color,
    lw = 4,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 2,
    legend_fontsize = 12,
    loc = 'upper left',
    x_unit = 'C',
    y_unit = ''
    )
ax5.scatter(
    x = df['average_temp'],
    y = df['lim_pine'],
    c = df['hour'], cmap = cmap, norm = norm,
    s = mc,
    edgecolor = 'black',
    zorder = 2 
    )
ax5.text(
    x = 0.45,
    y = 1.02,
    s = '12:00 - 18:00',
    fontsize = tick_fontsize,
    color = pm_color,
    transform = ax5.transAxes, # relative to axis, not data
    )
ax5.text(
    x = 0.05,
    y = 1.02,
    s = '06:00 - 12:00',
    fontsize = tick_fontsize,
    color = '#A47342',
    transform = ax5.transAxes, # relative to axis, not data,
    )

ax5.set_ylim([-0.4, 13])
ax5.set_xlim([df['average_temp'].min() - 0.5, df['average_temp'].max() + 0.5])
ax5.set_xticks(
    ticks = np.linspace(df['average_temp'].min(), df['average_temp'].max(), 5),
    labels = np.linspace(df['average_temp'].min(), df['average_temp'].max(), 5),
    fontsize = tick_fontsize
    )
ax5.set_yticks(
    ticks = np.linspace(0, df['lim_pine'].max() + 0.6, 5),
    labels = np.linspace(0, df['lim_pine'].max() + 0.6, 5),
    fontsize = tick_fontsize
    )
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax5.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax5.set_ylabel(
    'limonene : α-pinene',
    fontsize = label_fontsize,
    labelpad = 0
    )
ax5.set_xlabel(
    'Average Temperature (C)',
    fontsize = label_fontsize,
    labelpad = 10
    )
ax5.legend (loc = 'upper left',
           ncols = 2,
           fontsize = legend_fontsize,
     #      bbox_to_anchor = (0, 0.95),
           frameon = False
           )


# bottom plot

# adding lines to represent campaign average ratios 
ax2.hlines(y = 0.25,
       #    y = df_pm['lim_pine'].median(), 
           xmin = df.index[0] - timedelta(hours = 6), 
           xmax = df.index[-1] + timedelta(hours = 6),
           linewidth = 2.5,
           color = pm_color,
           linestyle = '--'
           )
ax2.hlines(y = 1.70,
    #       y = df_am['lim_pine'].median(), 
           xmin = df.index[0] - timedelta(hours = 6), 
           xmax = df.index[-1] + timedelta(hours = 6),
           linewidth = 2.5,
           color = am_color,
           linestyle = '--'
           )

ax1.plot(df.index, df['lim_pine'], 
         color = 'black', linewidth = 2.5, 
         mew = 10,
        # alpha = 0.6,  # marker edge width
         zorder = 1)
ax2.plot(df.index, df['lim_pine'], 
         color = 'black', linewidth = 2.5, 
         mew = 10, # marker edge width
      #   alpha = 0.6, 
         zorder = 1)
sc1 = ax1.scatter(df.index, df['lim_pine'],
            c = df['hour'], cmap = cmap, norm = norm,
            s = mc, zorder = 3,
            linewidth = 1.8, # marker edge width
            edgecolor = 'black'
            )
sc2 = ax2.scatter(df.index, df['lim_pine'],
            c = df['hour'], cmap = cmap, norm = norm,
            s = mc, zorder = 3,
            linewidth = 1.8, # marker edge width
            edgecolor = 'black'
            )

# y-axis, top subplot
ax1.set_ylim(2.55, 14)
ax1.set_yticks(
    ticks = (np.linspace(3, 14, 4)),
            labels = (np.linspace(3, 14, 4)),
            size = tick_fontsize,
            fontweight = 'normal'
    )
ax1.xaxis.set_visible(True)
ax1.tick_params(labelbottom = False)  # hide labels but keep ticks active
ax1.spines.bottom.set_visible(False)
# y-axis, bottom subplot
ax2.set_ylim(0, 2)
ax2.set_yticks(
    ticks = np.linspace(0, 2, 5),
            labels = np.linspace(0, 2, 5),
            size = tick_fontsize,
            fontweight = 'normal'
    )
ax2.spines.top.set_visible(False)

# setting those horizonantal lines
# source code for the split axis: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = {'marker' : [(-1, -d), (1, d)], 'markersize' : 12,
              'linestyle' : "none", 'color' : 'k', 'mec' : 'k', 'mew' : 1, 'clip_on' : False}
ax1.plot([0, 1], [0, 0], transform = ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform = ax2.transAxes, **kwargs)

for ax in [ax1, ax2]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# x-axis
ax2.set(xlim = (df.index[0] - timedelta(hours = 6), 
                df.index[-1] + timedelta(hours = 6)), # adjusting x-lim 
       )
ax2.tick_params(axis = 'x', labelsize = label_fontsize)
#fig.autofmt_xdate()
ax2.xaxis.set_major_formatter(md.DateFormatter('%D'))

# forest vs urban boxes
for ax in [ax1, ax2]:
    ax.axhspan(
        ymin = 0.68,
        ymax = 14,
        color = 'olive',
        alpha = 0.1,
        hatch = '/'
        )
for ax in [ax1, ax2]:
    ax.axhspan(
        ymin = 0,
        ymax = 0.21,
        color = 'green',
        alpha = 0.1,
        hatch = '/'
        )

# adding manual labels
x = md.date2num(datetime.datetime(2023, 7, 9, 7, 2, 0)) # have to have x as datetime
ax2.text( # y-axis label for broken axis
    x = x, 
    y = 0.65, 
    s = 'limonene : α-pinene',
    rotation = 'vertical',
    fontsize = label_fontsize,
    fontweight = 'normal'
    )

# # adding labels for ratio boxes
x = md.date2num(datetime.datetime(2023, 8, 9, 0, 0, 0)) # have to have x as datetime
ax1.text( 
    x = x, 
    y = df['lim_pine'].max() - 0.08, 
    s = 'Urban Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'bold'
    )
ax2.text( 
    x = x, 
    y = df['lim_pine'].min() - 0.01, 
    s = 'Forested Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'bold'
    )

# get vertical span for colorbar
pos_top = ax5.get_position()
pos_bottom = ax2.get_position()
cbar_bottom = pos_bottom.y0
cbar_top = pos_top.y1
cbar_height = cbar_top - cbar_bottom
# place colorbar axis to the right
cax = fig.add_axes([pos5.x1 + 0.015,   # small gap to the right of rightmost subplot
                    cbar_bottom,
                    0.015,             # colorbar width
                    cbar_height
                    ])
cbar = fig.colorbar(sc2, cax = cax, pad = 0.01)

cbar.set_label('Hour of Day')
cbar.set_ticks(np.arange(df['hour'].min(), df['hour'].max() + 1, 3))
cbar.ax.tick_params(labelsize = tick_fontsize)  
cbar.set_label(
    'Hour of Day',
    fontsize = label_fontsize,
    rotation = 270,
    labelpad = 25    
    )

plt.show()
plt.close()  

# %% Combining HOD terpene data into one plot: option 3

df = pd.DataFrame()
df = full_cmpd_dataset[['limonene', 'a-pinene']].copy()
df['date_time'] = full_cmpd_dataset['start_time'].copy()
df['hour'] = full_cmpd_dataset['hour'].copy()
df['lim_pine'] = df['limonene']/ df['a-pinene']
df['average_temp'] = meta_data['avg_temp_c']
df['toluene'] = full_cmpd_dataset['toluene']
df.set_index('date_time', inplace = True)

# grouping by hour for diel plot
grouped = df.groupby('hour')['lim_pine'].describe()

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
ax3.set_xlabel('Hour of Day',
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
box_start = grouped.index.min() - 0.5
box_end = grouped.index.max() + 0.5
# forest vs urban boxes and labels
ax3.axhspan(
    ymin = 0.68,
    ymax = 1.7,
    color = '#580C1F',
    alpha = 0.25,
    hatch = '/'
    )
ax3.axhspan(
    ymin = 0,
    ymax = 0.21,
    color = '#BBD1EA',
    alpha = 0.6,
    hatch = '/'
    )
ax3.text( 
    x = 13.2, 
    y = 1.42, 
    s = 'Urban Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'normal'
    )
ax3.text( 
    x = 13.2, 
    y = 0.05, 
    s = 'Forested Regime',
    rotation = 'horizontal',
    fontsize = tick_fontsize,
    fontweight = 'normal'
    )
ax3.set_xlim([grouped.index.min() - 0.5, grouped.index.max() + 0.5])
ax3.set_xticks(
    ticks = np.linspace(grouped.index.min(), grouped.index.max(), 5),
    labels = np.linspace(grouped.index.min(), grouped.index.max(), 5),
    fontsize = tick_fontsize
    )
ax3.set_ylim([0, grouped['75%'].max()])
ax3.set_yticks(
    ticks = np.linspace(0, grouped['75%'].max(), 5),
    labels = np.linspace(0, grouped['75%'].max(), 5),
    fontsize = tick_fontsize
    )
ax3.legend(fontsize = legend_fontsize,
           loc = 'upper left',
           bbox_to_anchor = (0.1, 1.01))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

#ax4 = OLS plot
x_variable_name = 'a-pinene' # edit !
y_variable_name = 'limonene' # edit !
simple_corr_plot(ax4, 
                 x=x_variable_name, 
                 y=y_variable_name,
                 df=df_am,
                 num_dec = 0.2, # formatting for decimals
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
ax4.set_xlabel('α-pinene ($ppb_v$)',
               fontweight = fontweight,
               fontsize = label_fontsize,
               )
ax4.set_xlim([0, df['a-pinene'].max() + 0.01])
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
ax2.set_ylabel('α-pinene ($ppb_v$)',
               fontweight = fontweight,
               fontsize = label_fontsize,
               )
ax2.set_ylim([0, df['a-pinene'].max() + 0.01])
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
    ax.set_xlim([0, df['toluene'].max() + 0.1])
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
        color = '#052241',
        transform = ax.transAxes, # relative to axis, not data
        )
    ax.text(
        x = 0.05,
        y = 1.02,
        s = '06:00 - 12:00',
        fontsize = tick_fontsize,
        color = '#580C1F',
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

# %% TIMEBLOCKS: Combining HOD terpene data into one plot: option 3

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
ax3.axhspan(
    ymin = 0.68,
    ymax = 1.7,
    color = '#580C1F',
    alpha = 0.25,
    hatch = '/'
    )
ax3.axhspan(
    ymin = 0,
    ymax = 0.21,
    color = '#BBD1EA',
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
ax3.set_ylim([0, grouped['75%'].max()])
ax3.set_yticks(
    ticks = np.linspace(0, grouped['75%'].max(), 5),
    labels = np.linspace(0, grouped['75%'].max(), 5),
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
                 df=df_am,
                 num_dec = 0.2, # formatting for decimals
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
ax4.set_xlabel('α-pinene ($ppb_v$)',
               fontweight = fontweight,
               fontsize = label_fontsize,
               )
ax4.set_xlim([0, df['a-pinene'].max() + 0.01])
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
ax2.set_ylabel('α-pinene ($ppb_v$)',
               fontweight = fontweight,
               fontsize = label_fontsize,
               )
ax2.set_ylim([0, df['a-pinene'].max() + 0.01])
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
    ax.set_xlim([0, df['toluene'].max() + 0.1])
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
        color = '#052241',
        transform = ax.transAxes, # relative to axis, not data
        )
    ax.text(
        x = 0.05,
        y = 1.02,
        s = '06:00 - 12:00',
        fontsize = tick_fontsize,
        color = '#580C1F',
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

# %% Camphor vs a-Pine vs camphor vs lim 

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

ax1.set_xlim([0, df[x_variable_name].max() + 0.005])
ax1.set_xticks(ticks = np.linspace(0, df[x_variable_name].max() + 0.005, 4),
               labels = np.linspace(0, df[x_variable_name].max() + 0.005, 4),
               fontsize = tick_fontsize
               )
ax1.set_ylim([0, df[y_variable_name].max() + 0.005])
ax1.set_yticks(ticks = np.linspace(0, df[y_variable_name].max() + 0.005, 4),
               labels = np.linspace(0, df[y_variable_name].max() + 0.005, 4),
               fontsize = tick_fontsize
               )
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
ax3.set_xlim([0, df[x_variable_name].max() + 0.005])
ax3.set_xticks(ticks = np.linspace(0, df[x_variable_name].max() + 0.005, 4),
               labels = np.linspace(0, df[x_variable_name].max() + 0.005, 4),
               fontsize = tick_fontsize
               )

x_variable_name = 'limonene' # edit !
y_variable_name = 'camphor' # edit !
color = 'slategrey'
num_dec = 0.1
ols_corr_plot_outliers(ax4, x_variable_name, y_variable_name, data_df, 
                       outlier_index, color, label_fontsize, tick_fontsize, 
                       fontweight, markersize, num_dec, sci_note, 
                       legend_fontsize, loc)
ax4.set_xlim([-0.005, df[x_variable_name].max() + 0.005])
ax4.set_xticks(ticks = np.linspace(0, df[x_variable_name].max() + 0.005, 4),
               labels = np.linspace(0, df[x_variable_name].max() + 0.005, 4),
               fontsize = tick_fontsize
               )

for ax in [ax1, ax2]:
    ax.set(xticklabels = [], xlabel = None)
for ax in [ax1, ax3]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
for ax in [ax3, ax4]:
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
for ax in [ax2, ax4]:
    ax.set_ylabel('')

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

# %%  xylenes vs ethylbenzene for MV

xylene = full_cmpd_dataset['m-xylene'] + full_cmpd_dataset['o-xylene']
d = {
     'xylene' : xylene,
     'ethylbenzene' : full_cmpd_dataset['ethylbenzene'],
     }
df = pd.DataFrame(data = d)
df = df.replace(-9999, np.nan)
df.dropna(inplace = True, ignore_index = True)

# df.set_index('date_time', inplace = True)
# df_am = df.between_time('05:00', '12:00')
# df_pm = df.between_time('12:01', '23:00')

fig, ax = plt.subplots(
    figsize = (10.5, 6), 
    dpi = 300)

simple_corr_plot(
    axis = ax,
    x = 'xylene',
    y = 'ethylbenzene',
    sci_note = False,
    df = df,
    num_dec = 0.3,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = 'brown',
    linecolor = 'brown',
    lw = 2,
    linestyle = 'solid',
    label_fontsize = 14,
    tick_fontsize = 12,
    fontweight = 'normal',
    markersize = 8,
    legend_fontsize = 12,
    loc = 'upper left',
    x_unit = '$ppb_v$',
    y_unit = '$ppb_v$'
    )

# %% MT % breakdown

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


# %% temp and PCP tracers

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

fig, ax = plt.subplots()
x = 'Average Temperature'
y = 'benzyl acetate'
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df,
    num_dec = 0.2, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'black',
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
plt.show()

fig, ax = plt.subplots()
x = 'Average Temperature'
y = 'benzyl acetate'
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_am,
    num_dec = 0.2, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'orange',
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
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_pm,
    num_dec = 0.2, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
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
plt.show()

fig, ax = plt.subplots()
y = 'tricyclodecenyl acetate'
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df,
    num_dec = 0.2, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'black',
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
plt.show()

fig, ax = plt.subplots()
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_am,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'orange',
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
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_pm,
    num_dec = 0.2, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
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
plt.show()


fig, ax = plt.subplots()
y = 'diphenyl ether'
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'black',
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
plt.show()
fig, ax = plt.subplots()
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_am,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'orange',
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
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_pm,
    num_dec = 0.2, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
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
plt.show()


fig, ax = plt.subplots()
y = '4-tert-butylcyclohexyl acetate 2'
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'black',
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
plt.show()
fig, ax = plt.subplots()
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_am,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'orange',
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
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_pm,
    num_dec = 0.2, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
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
plt.show()



fig, ax = plt.subplots()
y = '4-tert-butylcyclohexyl acetate 1'
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'black',
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
plt.show()
fig, ax = plt.subplots()
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_am,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'orange',
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
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_pm,
    num_dec = 0.2, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
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
plt.show()



fig, ax = plt.subplots()
y = 'toluene'
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'black',
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
plt.show()
fig, ax = plt.subplots()
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_am,
    num_dec = 0.3, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
    linecolor = 'orange',
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
simple_corr_plot(
    axis = ax,
    x = x,
    y = y,
    df = df_pm,
    num_dec = 0.2, 
    sci_note = False,
    show_slope = False,
    show_r = False,
    show_both = True,
    color = cmpd_colors_dict[f'{y}'],
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
plt.show()

# %% Temp + Fragrances (SI Figure)

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
            color = 'orange',
            transform = ax.transAxes, # relative to axis, not data
            )
    ax.text(
            x = 0.51,
            y = 1.02,
            s = '12:00 - 18:00',
            fontsize = 14,
            color = 'darkblue',
            transform = ax.transAxes, # relative to axis, not data
            )

for ax in [ax3, ax4]:
    ax.set_xlabel('Average Temperature (C)',
                  fontsize = 14)

plt.show()

# %%






# axes = [ax1, ax3]
# for i, col in enumerate(df.columns):
#     ax = axes[i]
#     ax.set_yticks(
#         ticks = (np.linspace(0, df[col].max(), 5)),
#         labels = (np.linspace(0, df[col].max(), 5)),
#         size = 10,
#         fontweight = 'bold'
#         )



#plt.plot(df[''])




# test = percent_factor_total_dfs[0].copy()
# test = test.transpose() 
# test.columns = test.iloc[0] # setting cmpd names as column headers
# test = test.iloc[1:] # dropping extra row of cmpd names


# variable_count = len(test.columns)

# #i like the colors stored, but just uncomment to generate a new batch
# #generating a random color for each compound
# hexadecimal_alphabets = '0123456789ABCDEF'
# color = set() # sets don't allow duplicates, so will have unique color choices
# for i in range(variable_count):
#     hex_code = "#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)]) # need a 6-string hex code
#     color.add(hex_code)


# fig, ax = plt.subplots(figsize = (6, 6), dpi = 200)
# bars = test.plot(ax = ax, stacked = True,
#                  kind = 'bar', subplots = False, sharex = True, use_index = True, 
#           title = '3 Factor Solution',
#           ylim = [0, 100],
#           ylabel = '% of Factor',
#           rot = 45,
#           fontsize = 10,
#           color = color,
#           edgecolor = 'white', 
#           linewidth = 0.75
#           )

# # labeling the top 5 compounds making up each factor:
# for bar_idx, (index, row) in enumerate(test.iterrows()):
#     top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(5).index.tolist()
    
#     cum_height = 0
#     for col in test.columns:
#         height = row[col]
#         if height > 0:
#             if col in top_cols:
#                 ax.text(
#                     x = bar_idx,
#                     y = cum_height + height / 2,
#                     s = col + ': ' + f'{height:.1f}',
#                     ha = 'center',
#                     va = 'center',
#                     fontsize = 5,
#                     color = 'black'  # or a color that contrasts the bar
#             )
#         cum_height += height

# plt.legend(ncols = 2, fontsize = 'small', loc = 'best',
#            bbox_to_anchor = (1, 1))
# plt.show()
# plt.close()

# %%



# variable_count = len(test.columns)
# colormap = plt.get_cmap('terrain')
# colors = [colormap(0.1 + (i / (variable_count - 1)) * 0.85) for i in range(variable_count)]

# #fig, ax = plt.subplots(figsize = (6, 6), dpi = 300)
# test.plot(ax = ax, kind = 'bar', subplots = False, sharex = True, use_index = True, 
#           title = '3 Factor Solution',
#           ylim = [0, 100],
#           ylabel = '% of Factor',
#           stacked = True,
#           rot = 45,
#           fontsize = 10,
#           color = colors,
#           edgecolor = 'black', 
#           linewidth = 0.75
#           )
# plt.legend(ncols = 2, fontsize = 'small', loc = 'best',
#            bbox_to_anchor = (1, 1))
# plt.show()


# for df in percent_factor_total_dfs:
#     df.set_index('compounds', inplace = True)
#     df.plot(
#         kind = 'bar', subplots = False, sharex = True, use_index = True, 
#             title = '3 Factor Solution',
#             ylim = [0, 100],
#             ylabel = '% of Factor',
#             stacked = True,
#             rot = 90,
#             fontsize = 10,
#             figsize = (6, 6)# dpi = 300
                  
#         )


# test = percent_factor_total_dfs[0].copy()
# test.set_index('compounds', inplace = True)
# test.plot(kind = 'bar', subplots = False, sharex = True, use_index = True, 
#           title = '3 Factor Solution',
#           ylim = [0, 100],
#           ylabel = '% of Factor',
#           stacked = True,
#           rot = 90,
#           fontsize = 10,
#           figsize = (6, 6)# dpi = 300
#           )
# plt.show()
# plt.close()


# color.add(["#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)]) for i in range(variable_count)])
#color_list = list(color)

# colormap = plt.get_cmap('terrain')
# colors = [colormap(0.1 + (i / (variable_count - 1)) * 0.85) for i in range(variable_count)]