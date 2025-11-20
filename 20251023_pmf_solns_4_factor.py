# -*- coding: utf-8 -*-
"""
Created on Thu May  8 08:18:21 2025
Based on v4 analyses
@author: rossellr
"""

# %% Importing libaries

# module imports

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
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.patches as mpatches
from matplotlib.cbook import boxplot_stats
import numpy as np
# import math as mat
import scipy.stats as scipy
import statsmodels.api as sm
import matplotlib.dates as md
# from matplotlib.dates import DateFormatter
import datetime as datetime
from datetime import timedelta
from windrose import WindroseAxes
# from matplotlib.ticker import FormatStrFormatter
# from matplotlib.ticker import ScalarFormatter
# import time as time
from collections import defaultdict
import molmass
from molmass import Formula
import warnings
# import random
import textwrap
import seaborn as sns

# %% Defining all fuctions

plt.rcParams["mathtext.default"] = 'regular' # getting rid of italics with subscripts
plt.rcParams["mathtext.fontset"] = 'dejavusans' # setting font for subscripted numbers

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
    axes[ax].add_patch(box)

# function to add totals to the top of each bar in the bar plot
def add_bar_totals(x, y, fontsize, fontweight): 
    for i in range(len(x)):
        plt.text(x = i, y = y.iloc[i] + (y.iloc[i] * 0.02) ,  # setting x and y positions of labels - centering with bar and then slightly above top of bar
                 s = f'{y.iloc[i]:0.3f}', # setting and formatting what appears in the label
                 ha = 'center',
                 fontsize = fontsize,
                 fontweight = fontweight
                 )

# function to add totals to the top of each bar in the percentage bar plot
def add_percent_bar_totals(axis, x, y, total, units, fontsize, fontweight): 
    for i in range(len(x)):
        axis.text(x = i, y = y + (y * 0.02) ,  # setting x and y positions of labels - centering with bar and then slightly above top of bar
                 s = f'{total[i]:0.3f} {units}', # setting and formatting what appears in the label
                 ha = 'center',
                 fontsize = fontsize,
                 fontweight = fontweight
                 )

# function to create correlation plots, ignoring outlier points
def ols_corr_plot (axis, x, y, df, index, 
                   color, label_fontsize, tick_fontsize, 
                   fontweight, markersize, legend_fontsize, loc):
    
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
            label = f'All Values\nY = {model_all.params.iloc[1]:0.4f}x\n r: {pearson_all[0]:0.4f}')
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
            label = f'No Outliers\nY = {model_no_out.params.iloc[1]:0.4f}x\n r: {pearson_no_out[0]:0.4f}')
    axis.legend(
        #title = 'OLS Models',
        fontsize = legend_fontsize,
        loc = loc
        )

    axis.set_xticks(
        ticks = np.linspace(df_all[x].min(), df_all[x].max(), 8),
        labels = np.linspace(df_all[x].min(), df_all[x].max(), 8),
        fontsize = tick_fontsize,
        weight = fontweight
        )
    axis.set_yticks(
        ticks = np.linspace(df_all[y].min(), df_all[y].max(), 8),
        labels = np.linspace(df_all[y].min(), df_all[y].max(), 8),
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
    
    # plt.show()
    # plt.close()

# function to calculate number densities from ppb space
def number_density(conc_ppb):
    avo_constant = 6.022E23
    R_constant = 0.08206
    temp_k = meta_data['avg_temp_c'].median() + 273.13 # conver to K
    pressure_atm = meta_data['avg_press_inHg'].median() * 0.03342 # convert from inHg to atm
    vol_conv = 1000 # L to mL to cm3
    
    number_density_value = ((conc_ppb * avo_constant * 1E-9) / ((R_constant * temp_k) / pressure_atm)) / vol_conv 
    return(number_density_value)
    
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
#drive_folder = r'C:\Users\rossellr\OneDrive - Colostate\Desktop'

# home
drive_folder = r'C:\\Users\\rosse\\OneDrive - Colostate\\Desktop'


# %% Reading in and organizing metadata

# turning off "A value is trying to be set on a copy of a slice " warning b/c its annoying
pd.options.mode.chained_assignment = None  # default='warn' 


dt_format = '%Y-%m-%d %H:%M:%S'
today = datetime.date.today().strftime('%Y%m%d')


# reading in meta data of weather and trace gas data (pre-averaged by GC time blocks)
path = drive_folder + '\\CSU\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\ground_data_workup\\meta_data\\'
file = '20250711_ground_meta_data.csv' # edit!
meta_data_file = path + file
meta_data = pd.read_csv(meta_data_file, parse_dates = ['start_time', 'end_time'])
meta_data.drop(columns = ['sample_name', 'end_time'], inplace = True )
meta_data['hour'] = meta_data['start_time'].round('H')
meta_data['hour'] = meta_data['hour'].dt.hour
full_meta_data = meta_data.copy()
# grouping meta data by hour of the day for future diel plots
grouped_metadata = meta_data.groupby(['hour']).median()
#grouped_metadata.drop(columns = ['start_time'], inplace = True)


# reading in full weather data (not averaged by GC blocks)
weather_file = drive_folder + '\\CSU\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\full_weather_data_20230705_20230814.csv'
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


# reading in cmpd refs data
cmpd_ref_path = drive_folder + '\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\meta_data\\'
cmpd_ref_filename = '20251023_cmpd_reference_sheet.csv'
cmpd_ref_file = cmpd_ref_path + cmpd_ref_filename
cmpd_refs = pd.read_csv(cmpd_ref_file)
# only want specific columns
cmpd_refs = cmpd_refs[['cmpd_name', 'CAS', 'Molecular Weight (g/mol)', 'Molecular Formula', 'lod', 'MIR_best',
                       'sat_vapor_pressure_mmHg_25C', 'k_o3_298', 'k_oh_298']]
# renaming columns
cmpd_refs.rename(columns = {'cmpd_name' : 'compound_name',
                            'Molecular Weight (g/mol)' : 'mw_g_mol',
                            'Molecular Formula' : 'formula'}, inplace = True) # getting rid of caps, etc
cmpd_refs.columns = cmpd_refs.columns.str.rstrip() # removing white sapce at the end of the column names
cmpd_refs['compound_name'] = [remove_commas(x) for x in cmpd_refs['compound_name']] # replacing ',' w/ '_'
cmpd_refs.drop([94, 95, 96], inplace = True) # getting rid of MIR source text
# making sure everything is numerical:
cmpd_refs['MIR_best'] = pd.to_numeric(cmpd_refs['MIR_best'], errors='coerce')
cmpd_refs['compound_name'] = [remove_text(x, [' peak']) for x in cmpd_refs['compound_name']]


# reading in ground cmpd data (>24 samples)
cmpd_file = drive_folder + '\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\python\\outputs\\20250710_targeted_untargeted_concat.csv'
ground_cmpds_ppb = pd.read_csv(cmpd_file, parse_dates = ['start_time_edt'])
ground_cmpds_ppb['hour'] = ground_cmpds_ppb['start_time_edt'].dt.hour
ground_cmpds_ppb.columns = [remove_text(x, [' peak']) for x in ground_cmpds_ppb.columns]
full_cmpd_ts = ground_cmpds_ppb.copy()
# grouping cmpd data by hour of the day for future diel plots
ground_cmpds_ppb['hour'] = ground_cmpds_ppb['start_time_edt'].round('H')
ground_cmpds_ppb['hour'] = ground_cmpds_ppb['hour'].dt.hour


#reading in full dataset
file = drive_folder + '\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\python\\outputs\\20250723_full_targeted_untargeted_concat.csv'
full_dataset = pd.read_csv(file, parse_dates = ['start_time_edt'])
# renaming columns
full_dataset.columns = full_dataset.columns.str.rstrip() # removing white sapce at the end of the column names
full_dataset.columns = [remove_commas(x) for x in full_dataset.columns] # replacing ',' w/ '_'
full_dataset.rename(columns = {'pentanoic acid_ 2-ethylhexyl_ester' : 'pentanoic acid_2-ethylhexyl_ester'}, inplace = True)
full_dataset.columns = [remove_text(x, [' peak']) for x in full_dataset.columns]


# Reading in the PMF input file
file = drive_folder + '\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\pmf\\input\\csv\\20250709_all_cmpds_ppb.csv'
pmf_input = pd.read_csv(file, parse_dates = [0])


# reading in ACSM OA Data
file = drive_folder + '\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\meta_data\\FROG_ACSM_NR-PM1.csv'
col_names = ['datetime_local', 'total_ug_m3', 'OA_ug_m3', 'no3_ug_m3', 'nh4_ug_m3', 'so4_ug_m3', 'chl_ug_m3']
oa_data = pd.read_csv(file, skiprows = [0], 
                      names = col_names, 
                      parse_dates = ['datetime_local'])

# %% Reading in and organizing HS data
'''
Reading in all tenax HS samples from Lydia's quant spreadsheet by placing each HS in its own DF. Each HS DF is stored in a list.
'''

path = drive_folder + '\\CSU\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\python\\upload\\'
file = '20251023_HotSpot_Quant.xlsx' # edit!
full_path = path + file

HS_names = pd.read_excel( # getting the names of the HS samples
    full_path, sheet_name = 'MetaData',
    usecols = [0]
    )
HS_names.set_index('Sample Code', inplace =  True)
skip_col = 'Unnamed: 0' # first column
hotspot_excel_sheet = pd.read_excel( # reading in the entire sheet of data first
    full_path,
    sheet_name = 'Summary Data after LOD check',
    skiprows = range(0, 6),
    usecols = lambda x: x not in skip_col # skipping first column in excel sheet
    )

hs_dfs = []
total_cols = hotspot_excel_sheet.shape[1] # 0 = index, 1 = columns
test_dict = []

for i, start in enumerate(range(0, total_cols, 9)): # getting the data from the 3 columns, and then skipping the next 9 cols.
    num_hs = HS_names.index[i]
    ind_hs = hotspot_excel_sheet.iloc[:, start:start+3]
    
    if ind_hs.shape[1] == 3: # if the HS has exactly 3 columns, store in a dict.
        test_dict.append({'HS' : num_hs, 'df' : ind_hs}) # creating a dict entry of each HS and its name

for dict in test_dict: # turning each dictionary into a DF
    result = dict_to_df(dict)
    hs_dfs.append(result)

for df in hs_dfs:
    df.dropna(inplace = True) # dropping all NaNs
    

# need to fix the compound names:
for df in hs_dfs:
    df.iloc[:, 0] = df.iloc[:, 0].str.rstrip() # removing white sapce at the end of the column names
    text_variable = [' peak', '(EIC)', '(RT Match VOC3/4)', '*', '(RT Match VOC4)',
                     ('Std-VOC3/4'), '(Std-VOC4)', '(Std-VOC3)', '(/)', '(Std-Terp)', '()', 
                     '(Std-VOC3/Std-Terp)', '(/Std-Terp)', ' ()', ' () ', '  ()']
    df.iloc[:, 0] = [remove_text(x, text_variable) for x in df.iloc[:, 0]]
    # naming all the df columns the same
    df.rename(columns = {df.columns[0] : 'compounds',
                         df.columns[1] : 'cmpd ppb',
                         df.columns[2] : 'abs ppb'}, inplace = True)
    df.iloc[:, 0] = [remove_commas(x) for x in df.iloc[:, 0]] # replacing ',' w/ '_'
    df.iloc[:, 0] = df.iloc[:, 0].str.rstrip() # removing white sapce at the end of the column names



# %% Reading factor solution data

'''
Pulling and organizing factor solutions from the PMF output files, into organized lists that contain dataframes for each factor solution.
'''

# file containing factor solution folders
names = ['compounds']
base_path = drive_folder + '\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\pmf\\output'
specific_date_file = '\\20250716' # edit!
path = base_path + specific_date_file
base_naming = '_factor_100_runs' # edit !
num_cmpds = 31 # edit!

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
                                           nrows = len(ground_cmpds_ppb) + 5,
                                           parse_dates = [1], 
                                           date_format = dt_format
                                           )

                
                residual_ts = pd.read_excel(full_file, sheet_name = 'Residuals', 
                                        skiprows = [0, 1], 
                                        nrows = num_cmpds + 5,
                                        parse_dates = [1], 
                                        date_format = dt_format
                                        ) 

                
                scaled_residual_ts = pd.read_excel(full_file, sheet_name = 'Residuals', 
                                           skiprows = range(0, num_cmpds + 9), 
                                           nrows = num_cmpds + 5,
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
factor_ts.replace(-999, np.nan, inplace = True) 
factor_ts.rename(columns = {'compounds' : 'datetime'}, inplace = True)
factor_ts['hour'] = factor_ts['datetime'].round('H')
factor_ts['hour'] = factor_ts['hour'].dt.hour
    
residuals = [residual_ts, scaled_residual_ts]
for df in residuals:
    df.drop(columns = ['Run'], inplace = True)
    df.replace(-999, np.nan, inplace = True)
    df.rename(columns = {'Date_Time' : 'datetime'}, inplace = True)
    df['hour'] = df['datetime'].dt.hour
    

   
# OKAY
# so now we have 3 lists, each containin 7 dataframes. Each df contains data for a different factor solution.

# code used in blocks below
ignore = ['datetime', 'hour']

# %% Dictionaries of colors and linestyles

# setting colors for 4 factor solution to always be consistent:
color_dict = {
    'Factor 1' : '#840032',
    'Factor 4' : '#789180', 
    'Factor 2' : '#E49273',
    'Factor 3' : '#3B6064',
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
    'benzyl alcohol' : '#E37383',
    'camphor' : '#c44f69', 
    'methyl salicylate' : '#d577af', 
    'benzothiazole' : '#777793', 
    '4-tert-butylcyclohexyl acetate peak 1' : '#ff859c', 
    '4-tert-butylcyclohexyl acetate 1' : '#ff859c',
    '4-tert-butylcyclohexyl acetate peak 2' : '#ffe3ea', 
    '4-tert-butylcyclohexyl acetate 2' : '#ffe3ea', 
    'hexadecane' : '#715412',
    'dodecane' : '#9c6f08',
    'benzene' : '#766856',
    '1_3_5-trioxane peak 1' : '#6c541e',
    '1_3_5-trioxane 1' : '#6c541e',
    'hexamethylcyclotrisiloxane' : '#e18e96',
    'tetrachloroethylene' : '#b6a999', 
    '4-chlorobenzotrifluoride' : '#708090', 
    'm-xylene' : '#402e32',
    '1_3_5_7-tetroxane peak 1' : '#ba7dd2', 
    '1_3_5_7-tetroxane 1' : '#ba7dd2',
    'propylbenzene' : '#702b18', 
    'phenol' : '#410000', 
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
    'methyl naphthalene 1' :  '#54424a', 
    '6-tert-butyl-o-cresol' : '#ffb8f8', 
    'methyl naphthalene peak 2' : '#86727b', 
    'methyl naphthalene 2' : '#86727b', 
    'dodecamethylcyclohexasiloxane' : '#e18e96',
    'tetradecane' : '#dda01e', 
    'tricyclodecenyl acetate' : '#F33A6A', 
    '2-ethyl-1-hexanol' : '#D3D3D3',
    'ethylhexyl benzoate' : '#ae6275',
    'heptadecane' : '#412e33',
    'txib' : '#4a3e20',
    'other' : '#87ceeb',
    'total_oh' : 'darkmagenta',
    'Total OH Reactivity' : 'darkmagenta'
    }

# %% Factors by total compund mass

df = profile_conc.copy()
df = df.sum(axis = 0, numeric_only = True)  # summing each factor total


fig, ax = plt.subplots(figsize = (5, 8), dpi = 300)
bar_plot = df.plot(ax = ax, kind = 'bar', sharex = True,
                   title = f'{len(df)}' + ' Factor Solution',
                   ylabel = 'Total Factor Mixing Ratio (ppb)',
                   rot = 45,
                   edgecolor = 'gray', 
                   color = 'cadetblue',
                   linewidth = 0.75
                   )
add_bar_totals(range(len(df)), df.iloc[range(len(df))], 12, 'bold')
plt.show() 

# %% T:B Ratios for Factor Solutions

df = profile_conc.copy()

df.set_index('compounds', inplace = True)
df.loc['b_t'] = df.loc['toluene'] / df.loc['benzene']


row = df.loc['b_t']
fig, ax = plt.subplots(figsize = (5, 8), dpi = 300)
bar_plot = row.plot(ax = ax, kind = 'bar', 
                   title = f'{len(df.columns)}' + ' Factor Solution',
                   ylabel = 'Toluene : Benzene',
                   rot = 45,
                   color = 'lightgray',
                   edgecolor = 'black', 
                   linewidth = 0.75,
                   )
add_bar_totals(range(len(row)), row.iloc[range(len(row))], 12, 'bold')
plt.show() 

# %% Full Factor TS w/ 2 subplots

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

df = factor_ts.copy() # no overwriting! 
df.set_index('datetime', inplace = True)
df.drop(columns = ['hour'], inplace = True)
    
fig, (ax1, ax2) = plt.subplots(
    nrows = 2,
    ncols = 1,
    sharex = True,
    figsize = (8, 6),
    dpi = 300)

# for ax in [ax1, ax2]:
#     cycler = plt.cycler(linestyle = ['solid', 'dashdot'])
#     ax.set_prop_cycle(cycler)


axis1 = ['Factor 1', 'Factor 4']
axis2 = ['Factor 2', 'Factor 3']
  
# plotting factor solns
for col in df.columns:
    if col in axis1:
        plt1 = ts_plot2(df, ax1, color = color_dict[col], linestyle = linestyle_dict[col])
    if col in axis2:
       plt2 = ts_plot2(df, ax2, color = color_dict[col], linestyle = linestyle_dict[col])

ax1.set(
       xlim = (df.index[0] - timedelta(hours = 6), df.index[-1] + timedelta(hours = 6)), # adjusting x-lim
       xlabel = 'Local Date/Time',  
       #title = f'{len(df.columns)} Factor Solution' 
       )

for ax in [ax1, ax2]:
    ax.legend(
        ncols = 1,
        loc = 'upper right',
        fontsize = 'small'
        )
    ax.set_ylabel(
        'Normalized Cmpd Conc.'
        )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.autofmt_xdate()

ax1.xaxis.set_major_formatter(md.DateFormatter('%D %H:%M'))


plt.show()
plt.close()



# %% Factor Full TS against Temp and Smoke Events

df = factor_ts.copy() # no overwriting! 
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
    ncols = 3
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
height = 20 
box_color = 'maroon' # edit ! 
set_alpha = 0.2 # edit ! 
rect_box(heatwave_start, heatwave_end, ax, '') # adding heatwave box

box_color = 'gray'
rect_box(smoke1_start, smoke1_end, ax, '')
rect_box(smoke2_start, smoke2_end, ax, '') # adding smoke boxes

fig.autofmt_xdate()
ax.xaxis.set_major_formatter(md.DateFormatter('%D %H:%M'))

plt.show()
plt.close()


# %% Factor Soln Diel Plots

df = factor_ts.copy() # no overwriting! 

fig, ax = plt.subplots()

for col in df.columns:
    if col not in ignore:
        grouped = df.groupby('hour')[col].median()
        ax.plot(grouped.index, grouped.values, label = col,
                color = color_dict[col],
                 linestyle = '--', marker = 'o')
   
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Median Normalized Cmpd Conc.')

ax.legend()

plt.show()
plt.close()   


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
            y = 0.8,
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
    bbox_to_anchor = [0.5, 0.98],
    framealpha = 0.2
    )
# fig.suptitle(f'{len(df_weekday.columns) - len(ignore)} Factor Solution',
#              x = 0.5,
#              y = 0.92,
#              ha = 'center',
#              )
ax.text( # making a y-axis label for all subplots
    x = 3.9,
    y = 10, 
    s = 'Median Cmpd Conc. (Normalized ppb)',
    rotation = 'vertical'
    )
plt.xlabel('Hour of Day')# setting x-axis label on bottom plot only
plt.show()
plt.close()
   
# %% Corr Matrix (R2) w/ MetaVariables Over Time

# getting the big ass meta data df ready

meta_data_copy = meta_data.copy() # no overwriting !
cmpds_ppb_copy = ground_cmpds_ppb.copy()

starttime = meta_data_copy['start_time']

meta_data_copy.drop(columns = ['start_time', 'avg_rain_in/s'], inplace = True)
cmpds_ppb_copy.drop(columns = ['start_time_edt'], inplace =  True)
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

# %% Corr Matrix (r) w/ All MetaVariables Over Time

# getting the big ass meta data df ready

meta_data_copy = meta_data.copy() # no overwriting !
cmpds_ppb_copy = ground_cmpds_ppb.copy()

starttime = meta_data_copy['start_time']

meta_data_copy.drop(columns = ['start_time', 'avg_rain_in/s'], inplace = True)
cmpds_ppb_copy.drop(columns = ['start_time_edt'], inplace =  True)
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

# %% Corr Matrix (r) w/ Specific Meta Tracers

meta_data_copy = meta_data.copy() # no overwriting !
#cmpds_ppb_copy = ground_cmpds_ppb.copy()

starttime = meta_data_copy['start_time']

meta_data_copy.drop(columns = ['start_time', 
                               'avg_rain_in/s',
                               'median_temp_c',
                               'max_temp_c',
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
    'avg_temp_c' : 'Average Temperature (C)',
    'median_SolarRad_W/m2' : 'Median Solar Radiation (W/$m^2$)',
    'avg_WindSpeed_m/s' : 'Average Wind Speed (m/s)',
    'avg_o3_ppb' : 'Average $O_3$ Conc. (ppb)',
    'avg_no_ppb' : 'Average NO Conc. (ppb)',
    'avg_no2_ppb' : 'Average $NO_2$ (ppb)',
    'avg_oa_ug_m3' : 'Average OA Conc. (μg/$m^3$)',
    }, inplace = True)

df = df.corr() # getting pearson coeficcients
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
      if float(f'{value:.2}') >= 0.4 : # only displaying R higher than 0.5
         ax.text(j, i, f'{value:.2}', ha = 'center', va = 'center', color = 'black', fontsize = 20) # adjusting the value:# will change the sigfigs
      if float(f'{value:.2}') <= - 0.4 : 
         ax.text(j, i, f'{value:.1}', ha = 'center', va = 'center', color = 'black', fontsize = 20)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

plt.show()
plt.close()


# %% Bar Charts of Compound Distributions in Factors

# getting the dfs organized correctly
df = percent_factor_total.copy()  # no overwriting! 
units = 'ppbv'
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

fontsize = 18
fontweight = 'bold'

fig, ax = plt.subplots(figsize = (16, 16), dpi = 300)
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

plt.legend(ncols = 1, fontsize = 'x-large', loc = 'best',
           bbox_to_anchor = (1, 1), reverse = True)

fontsize = 20
add_percent_bar_totals(ax, range(len(df)), 100, df_total, units, fontsize, fontweight)

plt.show()
plt.close()

# %% Bar Charts of Factors Total OH Reactivities

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
    
factor_oh_react = df_percent.copy()
oh_total = factor_oh_react.sum(axis = 0, numeric_only = True)
units = '($s^-1$)'
for col in df_percent:
    df_percent[col] = (df_percent[col] / df_percent[col].sum()) * 100 # getting factor reactivity percentages

df_percent = df_percent.transpose()  # need cmpds as column headers and factors as index
variable_count = len(df_percent.index) # of factors

fontsize = 18
fontweight = 'bold'

# OH reactivity percentages 
fig, ax = plt.subplots(figsize = (14, 14), dpi = 300)
bars = df_percent.plot(ax = ax, 
               stacked = True,
                 kind = 'bar', 
                 subplots = False, 
                 sharex = True, 
                 width = 0.9,
          color = [cmpd_colors_dict[col] for col in df.columns],
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
                    fontsize = 12,
                    color = 'black'  # or a color that contrasts the bar
                    )
        cum_height += height

plt.legend(ncols = 1, fontsize = 'x-large', loc = 'best',
           bbox_to_anchor = (1, 1))


add_percent_bar_totals(ax, range(len(df_percent)), 99, oh_total, units, fontsize, fontweight)

plt.show()
plt.close()

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
        
# %% Wind Roses w/ conc. and wind speed

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
                       figsize = (8.5, 9),
                       dpi = 300
                       )
axs = axs.flatten()
cols_to_plot = [col for col in df.columns if col not in dt_cols_ignore]
plt.subplots_adjust(hspace = 0.35, wspace = 0) # making the subplots closer together

for idx, col in enumerate(cols_to_plot[:4]):
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
                  alpha = 1)
    
    # Add colorbar for VOC concentration
    sm = plt.cm.ScalarMappable(cmap ='Reds', 
                               norm = plt.Normalize(vmin = np.nanmin(mean_voc), vmax = np.nanmax(mean_voc)))
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
    
    # ax.text(
    #     x = 0.515,
    #     y = 0.33,
    #     s = 'Wind Speed (m/s)',
    #     fontsize = 8,
    #     rotation = -45,
    #     transform = plt.gcf().transFigure
    #     )
        
plt.show()





# %% Temp-dependence of biogenic residuals and average temp

df = residual_ts.copy()

x_variable = meta_data['median_temp_c'] # edit!
x_name = 'Average Temperature' # edit!
units = '(° C)' # edit!
bios = ['limonene', 'a-pinene']

x_linear = x_variable.copy()
x_linear = sm.add_constant(x_linear)

color_list = ['#9B287B', '#9B287B', '#C0B141', '#C0B141'] # edit if needed.

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
                #label = f'{col}: $R^2$ = {model.rsquared:0.4f}',
                label = f'{col}: $r$ = {pearson[0]:0.4f}',
                marker = '',
                alpha = 0.75)
        ax.set_xlabel(f'{x_name} {units}')
        ax.set_ylabel('Factor Residual')
        ax.legend()
        # ax.set_title(
        #     f'{len(percent_factor_total.columns) - 1} Factor Solution'
        #              )
             
plt.show()
plt.close()

# %% Temp-dependence of biogenic residuals and max temp

df = residual_ts.copy()

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
        ax.set_ylabel('Factor Residual')
        ax.legend()
        ax.set_title(
            f'{len(percent_factor_total.columns) - 1} Factor Solution'
                     )
 
plt.show()
plt.close()

# %% Corr Matrix (r) w/ Residuals Over Time

# getting the big ass meta data df ready

# meta_data_copy = meta_data.copy() # no overwriting !

# starttime = meta_data_copy['start_time']

# meta_data_copy.drop(columns = ['start_time', 'avg_rain_in/s'], inplace = True)
# meta_data_copy.set_index('hour', inplace = True)

# df = residual_ts.copy()
# df.drop(columns = ['datetime'], inplace = True)
# df.set_index('hour', inplace = True)

# # adding metadata to each factor df to run the corr matrix
# df = pd.concat([df, meta_data_copy], axis = 1) 

# df = df.corr()  # getting pearson coeficcients
# # df = np.square(df)  # getting R2 values


# fig, ax = plt.subplots(figsize = (24, 20), dpi = 300)
# cax = ax.imshow(df, cmap = 'coolwarm',  # plot matrix w/ chosen colormap
#                 interpolation = 'nearest') # round to nearest decimal place for heatmap value
    
# cbar = plt.colorbar(cax, shrink = 0.95, label = 'Pearson Coefficient Correlations')
# cbar.ax.tick_params(labelsize = 18)
# ax.set_xticks(range(len(df.columns))) # set xticks based on the columns in the matrix
# ax.set_xticklabels(df.columns, 
#                        rotation = 90, fontsize = 14) # set x values based on columns and rotate for visibility
# ax.set_yticks(range(len(df.columns))) # repeat for yaxis
# ax.set_yticklabels(df.columns, fontsize = 14)
# for (i, j), value in np.ndenumerate(df): # (coordinates) & r-value in loop in values within the mask:
#     #     if i > j: # ensure annotated values are in the bottom half of the plot
#         if float(f'{value:.2}') >= 0.7 : # only displaying R higher than 0.5
#            ax.text(j, i, f'{value:.2}', ha = 'center', va = 'center', color = 'black', fontsize = 7) # adjusting the value:# will change the sigfigs
#         if float(f'{value:.1}') <= - 0.3 : 
#            ax.text(j, i, f'{value:.1}', ha = 'center', va = 'center', color = 'black', fontsize = 7)
# plt.tight_layout()
# plt.title('Residuals',
#           fontsize = 20)
# plt.show()
# plt.close()

# %% Cmpd vs Cmpd OLS Model (w/ and w/out outliers)

# days of high Factor 1 and Factor 4: 
    # Factor 4: July 17, ~6 and 10am (indicies 5, 6) and August 2 (index 20)
    # Factor 1: July 11 - July 16 (indicies 0 - 4)
    
both_days_index = [0, 1, 2, 3, 4, 5, 6, 20]
only_f4_index = [5, 6, 20]

# running function definied previously: requires the following variables to be definied:

# defining general variables 

data_df = pmf_input
loc = 'upper left'

legend_fontsize = 12
label_fontsize = 14
tick_fontsize = 12
fontweight = 'bold'
markersize = 14


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    ncols = 2,
    nrows = 2,
    figsize = (14, 12), 
    dpi = 300)


# defining plot-specific variables
x_variable_name = 'benzene' # edit !
y_variable_name = 'toluene' # edit !
color = 'dodgerblue'
outlier_index = only_f4_index
# applying function and creating the plot
plt1 = ols_corr_plot(ax1, x_variable_name, y_variable_name, data_df, outlier_index, color, label_fontsize,
              tick_fontsize,
              fontweight, markersize, legend_fontsize, loc) 
ax1.set_title(
    'High F4 days excluded'
    )


x_variable_name = 'a-pinene' # edit !
y_variable_name = 'limonene' # edit !
color = 'darkgreen'
outlier_index = only_f4_index
plt2 = ols_corr_plot(ax2, x_variable_name, y_variable_name, data_df, outlier_index, color, label_fontsize,
              tick_fontsize,
              fontweight, markersize, legend_fontsize, loc) 
ax2.set_title(
    'High F4 days excluded'
    )


x_variable_name = 'benzene' # edit !
y_variable_name = 'toluene' # edit !
color = 'dodgerblue' # edit ! 
outlier_index = both_days_index
plt3 = ols_corr_plot(ax3, x_variable_name, y_variable_name, data_df, outlier_index, color, label_fontsize,
              tick_fontsize,
              fontweight, markersize, legend_fontsize, loc) 
ax3.set_title(
    'High F1 and F4 days exclded'
    )

x_variable_name = 'a-pinene' # edit !
y_variable_name = 'limonene' # edit !
color = 'darkgreen' 
outlier_index = both_days_index
plt4 = ols_corr_plot(ax4, x_variable_name, y_variable_name, data_df, outlier_index, color, label_fontsize,
              tick_fontsize,
              fontweight, markersize, legend_fontsize, loc) 
ax4.set_title(
    'High F1 and F4 days exclded'
    )

plt.subplots_adjust(wspace = 0.22, hspace = 0.22) # adjusting space between subplots

plt.show()


# %% Orignal Run G-Space Plots # Uncomment to run - turns the other plots gray for some reason

# warnings.simplefilter(action = 'ignore', category = FutureWarning)
 
# df = factor_ts.copy()
# df.drop(columns = ['datetime', 'hour'], inplace = True)

# sns.set(font_scale = 1.5) # setting font scale for all subplots

# plot = sns.PairGrid(df,  # create the grid of plots for all numeric data columns in the df 
#                     corner = True, # only plots bottom half 
#                     layout_pad = 0.05 # shrinking space between subplots
#                     )
 
# plot.map(sns.scatterplot, # plots a scatter plot on each grid 
#          color = 'rebeccapurple',
#          #alpha = 0.75,
#          s = 60, # markersize
#          ) 

# for ax in plot.axes.flatten(): # for every axes in the subplot:
#     if ax is not None: # ignoring the ones that we are hiding b/c of Corner = True
#         ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, prune = None)) # setting equal gride lines despite scale
#         ax.yaxis.set_major_locator(MaxNLocator(nbins = 5, prune = None))
#         ax.grid(True)  # Optional: show grid lines
#         ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # formatting decimal points 
#         ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#         ax.set_facecolor('white')
#         ax.grid(color = 'gray', alpha = 0.6)
#         for spine in ax.spines.values():
#             spine.set_visible(True)
#             spine.set_linewidth(1)
#             spine.set_edgecolor('gray')   # Box color
            
# fig.tight_layout()
# plt.show()

# %% Negative Rotational FPEAK G-Space Analysis - # Uncomment to run - turns the other plots gray for some reason

# fpeak_min = 0.1 # edit ! # absolute value of fpeak
# fpeak_max = 0.5 # edit !

# fpeak_values = []
# for i in np.arange(fpeak_min, (fpeak_max + 1), 0.1):
#     fpeak_values.append(i)
# fpeak_values = [-x for x in fpeak_values]
# fpeak_values = [round(x, 1) for x in fpeak_values]

# fpeak_neg_pt1 = []
# fpeak_neg_pt2 = []

# names = ['empty_1', 'datetime', 'Factor 1', 'Factor 2', 'Factor 3', 'Factor 4']

# for i, folder in enumerate(file_paths):
#     factor = factors[i] # getting each factor name from the factor list
    
#     for x in os.listdir(folder): 
#         if x.endswith('_fpeak.xlsx'): # only pulling from the base file
#             full_file = os.path.join(folder, x)
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore') # ignoring all the parse formatting warnings
                
#                 fpeak_neg_pt1 = pd.read_excel(full_file, sheet_name = 'Contributions', 
#                                            skiprows = [0, 1, 2], 
#                                            nrows = len(ground_cmpds_ppb) - 1,
#                                            parse_dates = [1], 
#                                            date_format = dt_format,
#                                            header = None,
#                                            names = names
#                                            )
#                 fpeak_neg_pt2 = pd.read_excel(full_file, sheet_name = 'Contributions', 
#                                            skiprows = range(0, num_cmpds + 10), 
#                                            nrows = num_cmpds + 4,
#                                            parse_dates = [1], 
#                                            date_format = dt_format,
#                                            header = None,
#                                            names = names
#                                            )
# datetime = fpeak_neg_pt1['datetime']
    
# fpeaks = [fpeak_neg_pt1, fpeak_neg_pt2]
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
    


# %% Positive Rotational FPEAK G-Space Analysis - # Uncomment to run - turns the other plots gray for some reason

# fpeak_min = 0.1 # edit ! # absolute value of fpeak
# fpeak_max = 0.5 # edit !

# fpeak_values = []
# for i in np.arange(fpeak_min, (fpeak_max + 1), 0.1):
#     fpeak_values.append(i)
# fpeak_values = [round(x, 1) for x in fpeak_values]

# fpeak_pos_pt1 = []
# fpeak_pos_pt2 = []

# names = ['empty_1', 'datetime', 'Factor 1', 'Factor 2', 'Factor 3', 'Factor 4']

# for i, folder in enumerate(file_paths):
#     factor = factors[i] # getting each factor name from the factor list
    
#     for x in os.listdir(folder): 
#         if x.endswith('_fpeak_pos_0.1.xlsx'): # only pulling from the base file
#             full_file = os.path.join(folder, x)
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore') # ignoring all the parse formatting warnings
                
#                 fpeak_pos_pt1 = pd.read_excel(full_file, sheet_name = 'Contributions', 
#                                            skiprows = [0, 1, 2], 
#                                            nrows = len(ground_cmpds_ppb) - 1,
#                                            parse_dates = [1], 
#                                            date_format = dt_format,
#                                            header = None,
#                                            names = names
#                                            )
#                 fpeak_pos_pt2 = pd.read_excel(full_file, sheet_name = 'Contributions', 
#                                            skiprows = range(0, num_cmpds + 10), 
#                                            nrows = num_cmpds + 4,
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

# %% Bar charts of original ground data by TOD

df = ground_cmpds_ppb.copy()
df.rename(columns = {'start_time_edt' : 'datetime'}, inplace = True)

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

fig, ax = plt.subplots(figsize = (13, 15), dpi = 300)
bars = df.plot(ax = ax, 
               stacked = True,
                 kind = 'bar', 
                 subplots = False, 
                 sharex = True, 
                 use_index = True, 
                 width = 0.95,
          ylim = [0, 3.05],
          fontsize = 14,
          color = [cmpd_colors_dict[col] for col in df.columns],
          edgecolor = 'white', 
          linewidth = 0.75,
          rot = 360
          )

ax.set_ylabel(
    'Median Mixing Ratios ($ppb_v$)',
    fontsize = 16,
    fontweight = 'bold'
    )
ax.set_xticks(
    ticks = range(len(df)),
    labels = xlabels,
    fontsize = 16,
    fontweight = 'bold'
    )

ax.set_yticks(
    ticks = np.linspace(0, 3, 7),
    labels = np.linspace(0, 3, 7),
    fontweight = 'bold',
    fontsize = 16)

#labeling the top 5 compounds making up each time block:
for bar_idx, (index, row) in enumerate(df.iterrows()):
    top_cols = pd.to_numeric(row, errors = 'coerce').nlargest(5).index.tolist()
    
    cum_height = 0
    for col in df.columns:
        height = row[col]
        mid_point = cum_height + height / 2
        if col in top_cols and height > 0:
            
                    wrapped_label = '\n'.join(textwrap.wrap(col, width = 25))  # Adjust width as needed
                
                    ax.text(
                    x = bar_idx,
                    y = mid_point,
                    s = f'{wrapped_label}' + ': ' + f'{height:.2f}',
                    ha = 'center',
                    va = 'center',
                    fontsize = 12,
                    color = 'black'  # or a color that contrasts the bar
                    )
        cum_height += height

plt.legend(ncols = 1, fontsize = 'small', loc = 'best',
           bbox_to_anchor = (1, 1)
           )
ax.get_legend().remove()

add_percent_bar_totals(ax, range(len(df)), 3.04, df_total, units, 14, 'bold')

plt.show()
plt.close()

# %% Diel plots of tracer compounds

df = ground_cmpds_ppb.copy()
df.rename(columns = {'start_time_edt' : 'datetime'}, inplace = True)

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

# %% TS of tracer compounds

df = ground_cmpds_ppb.copy()
df.rename(columns = {'start_time_edt' : 'datetime'}, inplace = True)
df.set_index('datetime', inplace = True)
df.drop(columns = 'hour', inplace = True)

    
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
    nrows = 5,
    ncols = 1,
    sharex = True,
    figsize = (8, 8),
    dpi = 300)

tracer1 = ['toluene']
tracer2 = ['benzyl acetate']
tracer3 = ['acetophenone']
tracer4 = ['o-xylene']
tracer5 = ['3-ethyl toluene']
     

tracer_axes = {
    ax1: tracer1,
    ax2: tracer2,
    ax3: tracer3,
    ax4: tracer4,
    ax5: tracer5
}     

mc = 12

# plotting 
for col in df.columns:
    for ax, tracers in tracer_axes.items():
        if col in tracers:
            general_ts_plot(df, ax, color = cmpd_colors_dict[col], 
                                     linestyle = 'solid',  mc = mc) 
    
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

df = ground_cmpds_ppb.copy()
df.drop(columns = ['start_time_edt', 'hour'], inplace = True)

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
        return '{:.1f}%\n({v:0.2f} $ppb_v$)'.format(pct, v=val)
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
        textprops = {'fontsize' : 16, 'fontweight' : 'bold'},
        pctdistance = 0.83
            )
ax.get_legend().remove()
ax.axis('off')
ax.set_title(
    f'Total OFP: {df_total:0.2f} $ppb_v$', # based on all median values
    x = 0.48,
    y = 0.93,
    fontsize = 20,
    fontweight = 'bold'
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
                 s = f'{value[i] / total * 100:0.2f}%', # setting and formatting what appears in the label
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
    axis.set_title(f'Factor {i + 1}',
                   x = 0.9, 
                   y = 0.85,
                   fontweight = 'bold',
                   )
    axis.set_ylim(0, 1.5)
    add_percent_bar_totals(axis, df.index, values, values, total)
    

fig, axes = plt.subplots(
     nrows = len(factor_dfs),
     ncols = 1,
     sharey = True,
     dpi = 200,
     figsize = (8, 12)
     )

i = 0 # starting the counter at 0
      
for j, df in enumerate(factor_dfs):
    ax = axes[i]
    bar_graph_subplot(df = df, axis = ax)
    ax.text(
        x = 4.65,
        y = 1,
        s = f'Total:\n{ofp_factor_totals.iloc[i, 1]:0.3f} $ppb_v$'
        )
    i += 1  # move to next axis
    
plt.text( # making a y-axis label for all subplots
    x = -1.5,
    y = 3, 
    s = 'OFP ($ppb_v$)',
    rotation = 'vertical',
    fontsize = 14
    )

plt.show()
    
# %% Van Kreeveln plots vs VP (all compounds)

df = full_dataset.copy()
df.drop(columns = ['start_time_edt'], inplace = True)
cmpds = df.columns

ref_copy = cmpd_refs[['compound_name', 'formula', 'k_oh_298', 'sat_vapor_pressure_mmHg_25C', 'mw_g_mol']].copy()
ref_copy.set_index('compound_name', inplace = True)
ref_copy = ref_copy.loc[cmpds] # filtering for cmpds  # need exact match for loc / list
ref_copy.set_index('formula', inplace = True)
formulas = ref_copy.index

vankrev_df = calc_ratios(formulas) # apply the van krev function
vankrev_df.set_index('compound', inplace = True)

df = pd.concat([vankrev_df, ref_copy], axis = 1)
df['mw_g_mol'] = pd.to_numeric(df['mw_g_mol'], errors='coerce')

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
axes[0].set_xlim([df['sat_vapor_pressure_mmHg_25C'].min() - 0.000001, df['sat_vapor_pressure_mmHg_25C'].max() + 75])
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



cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), ax = axes.ravel().tolist(), pad = 0.1)
cbar.set_label(label = 'OH Reactivity ($sec{^-1}$', 
                   fontsize = 8) # set label of the colorbar on the top plot
# cbar = plt.colorbar(cm.ScalarMappable(norm = color_norm, cmap = cmap), ax = ax2, pad = 0.1)
# cbar.set_label(label = 'OH Reactivity ($sec{^-1}$', 
#                    fontsize = 8) # set label of the colorbar on the top plot

axes[0].set_title('All Measured sVOCs (sized by MW)')



plt.show()

# %% Van Krevveln plots vs C* (all compounds)

#cstar equation from SI, Donahue et al (2006); https://doi.org/10.1021/es052297c


df = full_dataset.copy()
df.drop(columns = ['start_time_edt'], inplace = True)
cmpds = df.columns

ref_copy = cmpd_refs[['compound_name', 'formula', 'k_oh_298', 'sat_vapor_pressure_mmHg_25C', 'mw_g_mol']].copy()
ref_copy.set_index('compound_name', inplace = True)
ref_copy = ref_copy.loc[cmpds] # filtering for cmpds  # need exact match for loc / list
ref_copy.set_index('formula', inplace = True)
formulas = ref_copy.index

vankrev_df = calc_ratios(formulas) # apply the van krev function
vankrev_df.set_index('compound', inplace = True)

df = pd.concat([vankrev_df, ref_copy], axis = 1)
df['mw_g_mol'] = pd.to_numeric(df['mw_g_mol'], errors='coerce')

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
    [0.3, 300],
    [300, 3E6],
    [3E6, 3E15]
    )

box_colors = ['#7A8586' , #svoc
              '#9EBABD' ,  #ivoc
              '#DBE6E7', #voc
              ]

label_list = [
    'SVOC', 
    'IVOC',
    'VOC'
    ]

fontweight = 'bold'

# adding  boxes to both axes.
for axis in [0, 1]:
    for i, color in zip(voc_ranges, box_colors):
        min = i[0]
        max = i[1]
        points_rect_box(min, max, axis, color = color)

# adding manual legends
patches = []
for i, color in zip(label_list, box_colors):
    leg = mpatches.Patch(color = color, label = i)
    patches.append(leg)
axes[0].legend(handles = patches, loc = 'upper left', 
               ncols = 3, 
               #columnspacing = 15
               )

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
cbar.set_label(label = 'OH Reactivity ($sec{^-1}$)', 
                   fontsize = 10) # set label of the colorbar on the top plot

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

# df = ground_cmpds_ppb.copy()
# df.rename(columns = {'start_time_edt' : 'datetime'}, inplace = True)

# df['timeblock'] = df['hour'].apply(lambda x: 'am1' if x  <= 9 and x > 6
#                                    else 
#                                    'am2' if x <= 12 and x > 10
#                                    else
#                                    'pm1' if x <= 16 and x > 13
#                                    else 
#                                    'pm2'
#                                    )

# start_time = df['datetime']
# df.drop(columns = ['datetime', 'hour'], inplace = True)
     
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

# %% Timeseries of OH reactivities

df = ground_cmpds_ppb.copy()
starttime = df['start_time_edt'] # storing for later
df.drop(columns = ['start_time_edt', 'hour'], inplace = True)

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

# Total OH timeseries plot
fig, ax1 = plt.subplots(
    figsize = (8, 8),
    dpi = 300)
tracer = df_OH_timeseries.columns[0]
ts_plot(chosen_df = df_OH_timeseries, # applying previousy definied timeseries wave 
               tracer = tracer,
               axis = ax1)
ax1.get_legend().remove() # don't need the legend since we're only plotting total OH
# setting y-ticks + y-label
ax1.set_ylabel(
    "Total OH Reactivity ($sec^{-1}$)",
    size = 14,
    fontweight = 'bold'
    )
ax1.set_yticks(
    ticks = (np.linspace(0, df_OH_timeseries[tracer].max(), 4)),
    labels = (np.linspace(0, df_OH_timeseries[tracer].max(), 4)),
    size = 12,
    fontweight = 'bold'
    )
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# setting x-ticks and labels
plt.xticks(fontsize = 12, fontweight = 'bold')
fig.autofmt_xdate()
ax1.xaxis.set_major_formatter(md.DateFormatter('%D'))
plt.show()


# %% Total OH timeseries plot against factors TS 

# Calculating total OH for all PMF compounds:
df = ground_cmpds_ppb.copy()
starttime = df['start_time_edt'] # storing for later
df.drop(columns = ['start_time_edt', 'hour'], inplace = True)

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

fig, (ax1, ax2, ax3) = plt.subplots(
    nrows = 3,
    ncols = 1,
    sharex = True,
    figsize = (12, 9),
    dpi = 300)

axis1 = ['Factor 1', 'Factor 4']
axis2 = ['Factor 2', 'Factor 3']
  
# plotting factor solns
for col in df.columns:
    if col in axis1:
        plt1 = general_ts_plot(df, ax1, color = color_dict[col], 
                               linestyle = linestyle_dict[col],  mc = mc)
    if col in axis2:
       plt2 = general_ts_plot(df, ax2, color = color_dict[col], 
                              linestyle = linestyle_dict[col], mc = mc)

for col in df_OH_timeseries:
    plt3 = general_ts_plot(df_OH_timeseries, ax3, color = color_dict[col], 
                           linestyle = linestyle_dict[col], mc = mc)

    
ax1.set(
       xlim = (df.index[0] - timedelta(hours = 6), df.index[-1] + timedelta(hours = 6)), # adjusting x-lim
       xlabel = 'Local Date/Time',  
       )

for ax in [ax1, ax2, ax3]:
    ax.legend(
        ncols = 1,
        loc = 'upper right',
        fontsize = 'small'
        )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

for ax in [ax1, ax2]:
    ax.set_ylabel(
        'Normalized Cmpd Conc.'
        )
ax3.set_ylabel(
    'Total OH Reactivity ($sec^{-1}$)'
    )

fig.autofmt_xdate()

ax1.xaxis.set_major_formatter(md.DateFormatter('%D %H:%M'))


plt.show()
plt.close()



# %% Tracer ratios over time

df = pd.DataFrame()

df['date_time'] = ground_cmpds_ppb['start_time_edt'].copy()
df['t_b'] = ground_cmpds_ppb['toluene'].copy() / ground_cmpds_ppb['benzene'].copy()
df['lim_pine'] = ground_cmpds_ppb['limonene'].copy()/ ground_cmpds_ppb['a-pinene'].copy()
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

# %% OFP for Hot Spots

hs_dfs_ofp = [df.copy() for df in hs_dfs] # no overwriting!
for df in hs_dfs_ofp:
    df.drop(df.columns[2], axis = 1, inplace = True) # don't need uncert column
    df.set_index('compounds', inplace = True) # need compound names as index


ref_data = cmpd_refs.copy()
ref_data.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
ref_data.set_index('compounds', inplace = True)

for df in hs_dfs_ofp:
    ref_copy = ref_data.copy()
    ref_copy = ref_copy.loc[df.index] # sorting reference copy by each HS df index
    for value in df:
        df[value] = df[value] * ref_copy['MIR_best'] #  calculating OFP
    df.rename(columns = {df.columns[0] : 'cmpd ofp (ppb)'}, inplace = True)

# getting an OFP total for each HS sample
hs_ofp_totals = [df.copy() for df in hs_dfs_ofp]
hs_ofp_totals = [df.iloc[:, 0].sum() for df in hs_dfs_ofp] # adding the 1st col in each DF

for df in hs_dfs_ofp:
    df.sort_values(by = df.columns[0], ascending = False, inplace = True)


# top_number = 5
# others = pd.DataFrame(data = {
#     'other' : df[top_number:].sum()
#     })
# others = others.transpose()
# df = df[:top_number].copy()
# df = pd.concat([df, others], axis = 0)

# %% VCP Tracer Ratios

df = pd.DataFrame()

df['date_time'] = ground_cmpds_ppb['start_time_edt'].copy()
df['D5_benz'] = ground_cmpds_ppb['decamethylcyclopentasiloxane'].copy() / ground_cmpds_ppb['benzene'].copy()
df['benzyl_acetate_benzene'] = ground_cmpds_ppb['benzyl acetate'].copy()/ ground_cmpds_ppb['benzene'].copy()
df['benzyl_alcohol_benzene'] = ground_cmpds_ppb['benzyl alcohol'].copy()/ ground_cmpds_ppb['benzene'].copy()
df['acetophenone_benzene'] = ground_cmpds_ppb['acetophenone'].copy()/ ground_cmpds_ppb['benzene'].copy()
df.set_index('date_time', inplace = True)
    
axis1 = ['D5_benz']
axis2 = ['benzyl_acetate_benzene']
axis3 = ['benzyl_alcohol_benzene']
axis4 = ['acetophenone_benzene']

#gridspec = {'hspace' : 0.12, 'height_ratios' : [2, 0.08, 0.5, 2]}

fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    nrows = 4,
    ncols = 1,
    sharex = True,
    figsize = (12, 8),
    dpi = 200,
    #gridspec_kw = gridspec
    )


mc = 10

for col in df.columns:
    if col in axis1: 
        plt1_top = general_ts_plot(chosen_df = df, axis = ax1, 
                                   color = 'dodgerblue', linestyle  =  '--', mc = mc)

    if col in axis2:
        plt2 = general_ts_plot(chosen_df = df, axis = ax2, 
                               color = 'darkgreen', linestyle  =  '--', mc = mc)
    if col in axis3:
        plt3 = general_ts_plot(chosen_df = df, axis = ax3, 
                               color = 'darkgreen', linestyle  =  '--', mc = 10)
        
    if col in axis4: 
        plt4 = general_ts_plot(chosen_df = df, axis = ax4, 
                               color = 'darkgreen', linestyle  =  '--', mc = 10)

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
fig, axs = plt.subplots(2, 2, subplot_kw = {'projection':'polar'}, 
                        figsize = (16, 16), dpi = 300)
axs = axs.flatten()

for idx, col in enumerate(cols_to_plot[:4]):
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
fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05, pad=0.1, label='Factor Mixing Ratio ($ppb_v$)')

plt.show()



# %% Ind. Tracer TS Plot

df = ground_cmpds_ppb.copy()
df.rename(columns = {'start_time_edt' : 'datetime'}, inplace = True)
df.set_index('datetime', inplace = True)
df.drop(columns = 'hour', inplace = True)

    
fig, (ax1) = plt.subplots(
    nrows = 1,
    ncols = 1,
    sharex = True,
    figsize = (8, 3),
    dpi = 300)


tracer1 = ['benzyl acetate']
tracer_axes = {
    ax1: tracer1,
   }     

mc = 12

# plotting 
for col in df.columns:
    for ax, tracers in tracer_axes.items():
        if col in tracers:
            general_ts_plot(df, ax, color = cmpd_colors_dict[col], 
                                     linestyle = 'solid',  mc = mc) 
ax1.set_ylabel(
    'Mixing Ratio ($ppb_v$)',
    fontsize = 12,
    fontweight = 'bold'
    )

   
ax1.set_yticks(np.linspace(df['benzyl acetate'].min() - (df['benzyl acetate'].min() * 0.15), 
                            (df['benzyl acetate'].max() + (df['benzyl acetate'].max() * 0.15)), 5),
                labels = np.linspace(df['benzyl acetate'].min(), df['benzyl acetate'].max(), 5),
                fontsize = 13,
                ) 
for axis in [ax1]:
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))



ax1.set_xticks(pd.date_range(start = df.index[0], end = df.index[-1], periods = 10),
               labels = pd.date_range(start = df.index[0], end = df.index[-1], periods = 10),
               fontsize = 12
                ) 


ax1.yaxis.set_minor_locator(AutoMinorLocator())


fig.autofmt_xdate()
ax1.xaxis.set_major_formatter(md.DateFormatter('%D'))




plt.show()
plt.close()

# %% Box plots of tracer ratios

df = pd.DataFrame()

df['date_time'] = ground_cmpds_ppb['start_time_edt'].copy()
df['D5_benz'] = ground_cmpds_ppb['decamethylcyclopentasiloxane'].copy() / ground_cmpds_ppb['benzene'].copy()
df['benzyl_acetate_benzene'] = ground_cmpds_ppb['benzyl acetate'].copy()/ ground_cmpds_ppb['benzene'].copy()
df['benzyl_alcohol_benzene'] = ground_cmpds_ppb['benzyl alcohol'].copy()/ ground_cmpds_ppb['benzene'].copy()
df['acetophenone_benzene'] = ground_cmpds_ppb['acetophenone'].copy()/ ground_cmpds_ppb['benzene'].copy()
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