# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:59:23 2025
Preparing ground VOC data for PMF analysis
@author: rossellr
"""
# %% Module imports

import os
import glob
import re
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
import seaborn as sns 
import numpy as np
import math as mat
import scipy.stats as scipy
import statsmodels.api as sm
import matplotlib.dates as md
from matplotlib.dates import DateFormatter
import datetime as datetime
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import time as time
from collections import defaultdict
import molmass
from molmass import Formula

# %% Defining functions

# removing commas and replace with underscores
def remove_commas(text):
    chars_to_replace = [','] # text to remove
    for char in chars_to_replace:
        text = str(text)
        text = text.replace(char, '_') 
    return text

# %% Prepping datafiles

dt_format = '%Y-%m-%d %H:%M:%S'
today = datetime.date.today().strftime('%Y%m%d')
save_path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\python\outputs\\'

# all files are stored in the same folder for easy upload
file_path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\python\upload\quantified_cmpds'

for files in os.listdir(file_path):
    if files.endswith('targeted_cmpds_ppb.csv'): # reading in the targeted and nontargeted into seperate dfs
        full_path = os.path.join(file_path, files)
        targeted = pd.read_csv(full_path, parse_dates = ['start_time_edt', 'end_time_edt'])
    elif files.endswith('unknown_cmpds_ppb.csv'):
        full_path = os.path.join(file_path, files)
        non_targeted = pd.read_csv(full_path, parse_dates = ['start_time_edt', 'end_time_edt'])

# sensitivity csv files
path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\python\upload\\'
file = '20250827_all_stds_slopes_uncerts_lods.csv'
calibration_data = pd.read_csv(path + file)


# metadata files
meta_file = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\meta_data\ground_meta_data.csv'
vg_meta_data = pd.read_csv(meta_file, parse_dates = ['start_time', 'end_time'])
vg_meta_data.drop([11], axis = 0, inplace = True)
vg_meta_data.reset_index(inplace = True) # this added another index column which is stupid
vg_meta_data.drop(columns = {'index'}, inplace = True)

cmpd_ref_path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\meta_data\\'
cmpd_ref_filename = '20250827_cmpd_reference_sheet.csv'
cmpd_ref_file = cmpd_ref_path + cmpd_ref_filename
cmpd_refs = pd.read_csv(cmpd_ref_file)
# only want specific columns
cmpd_refs = cmpd_refs[['cmpd_name', 'CAS', 'Molecular Weight (g/mol)', 'Molecular Formula', 'surrogate_code', 'lod', 
                       'sat_vapor_pressure_mmHg_25C', 'k_o3_298', 'k_oh_298']]
# renaming columns
cmpd_refs.rename(columns = {'cmpd_name' : 'compound_name',
                            'Molecular Weight (g/mol)' : 'mw_g_mol',
                            'Molecular Formula' : 'formula'}, inplace = True) # getting rid of caps, etc
cmpd_refs.columns = cmpd_refs.columns.str.rstrip() # removing white sapce at the end of the column names
comma_copy = cmpd_refs.copy()
comma_copy.drop(index = 17, inplace= True) # dropping repeat 1,3,5-trioxepane 
comma_copy.reset_index(drop = True, inplace = True)
cmpd_refs['compound_name'] = [remove_commas(x) for x in cmpd_refs['compound_name']] # replacing ',' w/ '_'

pmf_path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\Data_Workup\pmf\python'
csv_name = '_cmpd_refs' # edit!
path = pmf_path + '\\' + today + csv_name + '.csv'
#cmpd_refs.to_csv(path, index = False)

# uncertainity files
file_path = r'C:\\Users\\rossellr\\OneDrive - Colostate\\Desktop\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\ground_data_workup\\python\\upload\\cmpds_uncert'

for files in os.listdir(file_path):
    if files.endswith('targeted_cmpds_uncerts.csv'): # reading in the targeted and nontargeted into seperate dfs
        full_path = os.path.join(file_path, files)
        targeted_uncert = pd.read_csv(full_path, parse_dates = ['start_time_edt', 'end_time_edt'])
    elif files.endswith('unknown_cmpds_uncerts.csv'):
        full_path = os.path.join(file_path, files)
        nontargeted_uncert = pd.read_csv(full_path, parse_dates = ['start_time_edt', 'end_time_edt'])

nontargeted_uncert.drop(columns = ['sample_code'], inplace = True) # dont need this column

full_target = targeted.copy()
full_non_target = non_targeted.copy()

# handling the data dfs
df_list = [targeted, non_targeted, targeted_uncert, nontargeted_uncert]

for df in df_list: 
    df.columns = [remove_commas(x) for x in df.columns] # replacing all commas with underscores
    df.columns = df.columns.str.rstrip()
    df.columns = df.columns.str.lstrip()
    df.replace([-9999, -10000, 0], np.nan, inplace = True) # gotta replace all the -9999 and 0 with nans
    df.dropna(thresh = len(df) - 10, axis = 1, inplace = True) # dropping cmpds that appear in less than 24 samples

non_targeted_dfs = [non_targeted, nontargeted_uncert]
for df in non_targeted_dfs:
    df.drop(columns = ['1_3_5-trioxepane peak 2.1'], inplace = True)
    df.rename(columns = {'2-tert-butylcyclohexyl acetate peak  1' : '2-tert-butylcyclohexyl acetate peak 1'}, inplace = True)

# %% Exporting concated targeted and nontargeted df - skip this cell 

non_target_copy = full_non_target.copy()
target_copy = full_target.copy()

dfs = [non_target_copy, target_copy]
start_time = target_copy['start_time_edt']

for df in dfs:
    df.drop(columns = ['end_time_edt'], inplace= True)
    df.set_index('start_time_edt', inplace = True)
    
concat = pd.concat([target_copy, non_target_copy], axis = 1)
concat.reset_index(inplace = True)

csv_name = '_full_targeted_untargeted_concat' # edit!
path = save_path + '\\' + today + csv_name + '.csv'  

# uncomment to save ! change df as needed
concat.to_csv(path, index = False)

# %% Prepping Data to Run PMF Analysis w/ EPA Software:

targeted_pmf = targeted.copy() # no overwriting !
non_targeted_pmf = non_targeted.copy()
start_time = targeted['start_time_edt'] # storing variable for later so it doesn't interfere with the loops
pmf_dfs_list = [targeted_pmf, non_targeted_pmf]
new_list = []

# getting median of all compounds to fill NaNs
for df in pmf_dfs_list:
    df = pd.DataFrame(df.median(numeric_only = True))
    new_list.append(df)
    
cmpd_medians = pd.concat(new_list)
cmpd_medians.rename(columns = {0 : 'cmpd_medians'}, inplace = True)

# replacing NaNs with median sample value
for df in pmf_dfs_list:
    df.drop(columns = ['end_time_edt', 'start_time_edt'], inplace = True)
    for col in df.columns:
        fill_value = cmpd_medians.loc[col, 'cmpd_medians']
        df[col] = df[col].fillna(fill_value)
    df.insert(0, column = 'start_time_edt', value = start_time, allow_duplicates = False) # putting the datetime column back in 
    df.set_index('start_time_edt', inplace = True)

# concatting targeted and untargeted compounds
all_cmpds_ppb = pd.concat([targeted_pmf, non_targeted_pmf], axis = 1, verify_integrity = True) # checking for overlapping columns
all_cmpds_ppb.reset_index(inplace = True, drop = True)

lods = cmpd_refs.copy()
lods = lods[['compound_name', 'lod']]
lods.set_index('compound_name', inplace = True)
lod_values = lods.loc[all_cmpds_ppb.columns]

# concentrations < DL were set to half the DL, as per Yuan et al (2012), Debevec et al (2017), Hopeke (2000)
for col in all_cmpds_ppb.columns:
    fill_value = lod_values.loc[col, 'lod']
    all_cmpds_ppb[col] = all_cmpds_ppb[col].where(all_cmpds_ppb[col] >= fill_value, other = fill_value * 0.5)
    
all_cmpds_ppb.insert(0, column = 'start_time_edt', value = start_time, allow_duplicates = False) # sticking the time column back up front

# %% saving the concentration pmf matrices # skip this cell if not saving

pmf_save_path = r'C:\\Users\\rossellr\\OneDrive - Colostate\\Desktop\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\pmf\\input\\csv'
csv_name = '_all_cmpds_ppb' # edit!
path = pmf_save_path + '\\' + today + csv_name + '.csv'  

# uncomment to save ! change df as needed
#all_cmpds_ppb.to_csv(path, index = False)

# %% reading in compound errors and concatting to match the main_vocs matrix
''' 
Uncertainites were handled as per Yuan et al (2012) and Debevec et al (2017):
Nans = 4x median concentration 
If concentrations were < LOD, replaced uncertainities w/ 5/6 lod value
'''

start_time = targeted_uncert['start_time_edt'] # storing variable for later so it doesn't interfere with the loops

targeted_uncert_pmf = targeted_uncert.copy() # no over-writing
non_targeted_uncert_pmf = nontargeted_uncert.copy()
pmf_uncert_list = [targeted_uncert_pmf, non_targeted_uncert_pmf]


# missing values = 4 * median concentration
for df in pmf_uncert_list:
    df.drop(columns = ['start_time_edt', 'end_time_edt'], inplace = True)
    for col in df:
        fill_value = cmpd_medians.loc[col, 'cmpd_medians'] # filling all NaN values with the median cmpd uncert.
        df[col] = (df[col].fillna(fill_value) * 4)
    df.insert(0, column = 'start_time_edt', value = start_time, allow_duplicates = False) 

for x in pmf_uncert_list:
    x.set_index('start_time_edt', inplace = True)
all_cmpds_uncert = pd.concat([targeted_uncert_pmf, non_targeted_uncert_pmf], axis = 1, verify_integrity = True)
all_cmpds_uncert.reset_index(inplace = True, drop = True)


# uncerts for concentrations < DL were set to 5/6 * DL
for col in all_cmpds_uncert:
    check_value = lod_values.loc[col, 'lod']
    fill_value = check_value * 5/6
    mask = all_cmpds_ppb[col] < check_value
    all_cmpds_uncert.loc[mask, col] = fill_value
    
# putting start_time col back in df
all_cmpds_uncert.insert(0, column = 'start_time_edt', value = start_time, allow_duplicates = False)

# %% saving the uncertainity pmf matrices # skip this cell if not saving

# check first!
# do the uncerts and main VOC dfs have the same cmpds?
test1 = all_cmpds_ppb.columns
test2 = all_cmpds_uncert.columns
check = test1 == test2
print(check)

csv_name = '_all_cmpds_uncert'
path = pmf_save_path + '\\' + today + csv_name + '.csv' 

# uncomment to save !
#all_cmpds_uncert.to_csv(path, index = False)

# %% Preparing stats table of quantified compounds

target_copy = full_target.copy()
nontarget_copy = full_non_target.copy()
df_list_copies = [target_copy, nontarget_copy]

for df in df_list_copies:
    df.replace([-9999, -10000, 0], np.nan, inplace = True) # gotta replace all the -9999 and 0 with nans
    df.columns = df.columns.str.rstrip()
    df.columns = df.columns.str.lstrip()

target_cmpds = pd.DataFrame(data = target_copy.columns, columns = ['target_cmpds'])
nontarget_cmpds = pd.DataFrame(data = nontarget_copy.columns, columns = ['nontarget_cmpds'])

df_list = [target_cmpds, nontarget_cmpds]
for df in df_list:
    
    df.drop(index = [0, 1], inplace = True) # dropping datetime out of target cmpds
    df.reset_index(drop = True, inplace = True)

all_cmpds = pd.concat([target_cmpds, nontarget_cmpds], axis = 1) # putting them together


for df in df_list_copies: # setting index for concat
    df.set_index(['start_time_edt', 'end_time_edt'], inplace = True)

cmpds_concat = pd.concat([target_copy, nontarget_copy], axis = 1)
cmpds_concat.reset_index(inplace = True, drop = True) # don't need datetime cols for stats table
stats_table = cmpds_concat.describe()
stats_table.drop(columns = ['1,3,5-trioxepane peak 2.1'], inplace = True)
stats_table.rename(columns = {'2-tert-butylcyclohexyl acetate peak  1' : '2-tert-butylcyclohexyl acetate peak 1'}, inplace = True)

# for col in stats_table.columns: # formatting all values but the sample counts (row 1)  
#     col_index = stats_table.columns.get_loc(col)    
#     col_slice = stats_table.iloc[1:, col_index].astype('object')
#     stats_table.iloc[1:, col_index] = col_slice.apply(
#         lambda x:'{:.3e}'.format(x) if isinstance(x, (int, float)) and not pd.isna(x) else x)


stats_table = stats_table.transpose()
stats_table.reset_index(inplace = True)
stats_table.rename(columns = {'index' : 'compounds', '50%' : 'median'}, inplace = True)

#  add column about calibration method
stats_table['cal_method'] = stats_table['compounds'].isin(target_cmpds['target_cmpds'])
stats_table.replace({True : 'direct', False : 'indirect'}, inplace = True)
# moving df columns to the front if needed
#stats_table = stats_table[['cal_method'] + [col for col in stats_table.columns if col != 'cal_method']]


stats_table.sort_values('compounds', ascending = True, inplace= True)
#stats_table.reset_index(drop = True, inplace = True)

stats_table.set_index('compounds', inplace = True)

cal_info = comma_copy.copy()
cal_info = cal_info[['compound_name', 'CAS', 'formula', 'surrogate_code']]
cal_info.sort_values('compound_name', ascending = True, inplace = True) # need indexes to match for concat
cal_info.rename(columns = {'compound_name' : 'compounds'}, inplace = True)
cal_info.dropna(inplace = True)
#cal_info.reset_index(drop = True, inplace = True)

cal_info.set_index('compounds', inplace = True)


stats_table = pd.concat([stats_table, cal_info], axis = 1)
# putting  cal method column as last column in df 
stats_table.insert(len(stats_table.columns)-1, 'cal_method', stats_table.pop('cal_method'))

# %% Exporting Stats Table - Skip this cell

df = stats_table.copy()

csv_name = 'full_cmpd_stats_table'
path = save_path + '\\' + today + '_' + csv_name + '.csv' 

# uncomment to save !
df.to_csv(path, index = True)



