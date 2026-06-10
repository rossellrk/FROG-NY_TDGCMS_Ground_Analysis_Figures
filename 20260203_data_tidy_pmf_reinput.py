# -*- coding: utf-8 -*-
"""
Created on 20260203
Re-preparing ground VOC data for PMF analysisafter re-quantification w/ TDGCMS macro 
@author: rossellr
"""
# %% Module imports

import pandas as pd
import numpy as np
import datetime as datetime
from pathlib import Path

dt_format = '%Y-%m-%d %H:%M:%S'
today = datetime.date.today().strftime('%Y%m%d')
save_path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\python\outputs\\'

# %% Defining functions

# removing commas and replace with underscores
def remove_commas(text):
    chars_to_replace = [','] # text to remove
    for char in chars_to_replace:
        text = str(text)
        text = text.replace(char, '_') 
    return text

# removing 'peak' labels and other text from cmpd names
def remove_text(text, to_replace):
    text = str(text)
    if isinstance(to_replace, str):
        to_replace = [to_replace]
    for char in to_replace:
        text = text.replace(char, '') 
    return text

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

# %% Prepping meta datafiles 

# metadata files
file_path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\meta_data\\'
file = '_ground_meta_data.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(file_path, file)
vg_meta_data = pd.read_csv(updated_file,
                           parse_dates = ['start_time', 'end_time'])

vg_meta_data['start_time'] = pd.to_datetime(vg_meta_data['start_time'])
vg_meta_data['end_time'] = pd.to_datetime(vg_meta_data['end_time'])

vg_meta_data.reset_index(inplace = True) # this added another index column which is stupid
vg_meta_data.drop(columns = {'index'}, inplace = True)

# compound reference data
file_path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\meta_data\\'
file = '_cmpd_reference_sheet.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(file_path, file)
cmpd_ref_data = pd.read_csv(updated_file,
                            usecols = [
                                'cmpd_name',
                                'CAS',
                                'Molecular Weight (g/mol)',
                                'Molecular Formula'
                                ])
cmpd_ref_data.set_index('cmpd_name', inplace = True) # need cmpds as index so they become col names
cmpd_ref_data = cmpd_ref_data.transpose()
# updating the column names
replace_text_list = [
    'EIC', 'RT Match', 'Std-', 'VOC3/4/', 'VOC4/', 'VOC3', 'VOC4', '/', 'Terp', 
    '( 4)', '(4)', '*', 
    '()', '( )'
    ]
cmpd_ref_data.columns = [remove_commas(x) for x in cmpd_ref_data.columns] # replacing all commas with underscores
cmpd_ref_data.columns = [remove_text(x, replace_text_list) for x in cmpd_ref_data.columns]
cmpd_ref_data.columns = cmpd_ref_data.columns.str.rstrip() # weird spaces after the text removal
cmpd_ref_data.columns = cmpd_ref_data.columns.str.lstrip()

cmpd_ref_data.rename(columns = { # just manually  changing the problem children
    'Dibutoxymethane' : 'dibutoxymethane',
    'pentanoic acid_ 2-ethylhexyl ester' : 'pentanoic acid_2-ethylhexyl ester',
    'homosalate 2' : 'homosalate',
    '2-tert-butylcyclohexyl acetate peak 2' : '2-tert-butylcyclohexyl acetate 2',
    '2-tert-butylcyclohexyl acetate peak 1' : '2-tert-butylcyclohexyl acetate 1',
    '4-tert-butylcyclohexyl acetate peak 1' : '4-tert-butylcyclohexyl acetate 1',
    '4-tert-butylcyclohexyl acetate peak 2' : '4-tert-butylcyclohexyl acetate 2',
    '4-chlorobenzotrifluoride' : '4-chlorobenzotrifluoride 1',
    'hexyl acetate' : 'hexyl acetate 1',
    'methyl naphthalene peak 1' : 'methylnaphthalene 1',
    'methyl naphthalene peak 2' : 'methylnaphthalene 2',
    }, inplace = True)
cmpd_ref_data.columns = cmpd_ref_data.columns.str.rstrip() #cleaning up the final names
cmpd_ref_data.columns = cmpd_ref_data.columns.str.lstrip()

cmpd_ref_data = cmpd_ref_data.transpose()
# reseting
cmpd_ref_data.reset_index(inplace = True)
cmpd_ref_data.rename(columns = {'index' : 'compound'}, inplace = True)
cmpd_ref_data.dropna(inplace = True)

# list of surrogate standards
file_path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\meta_data\\'
file = '_compound_surrogates.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(file_path, file)
surrogates = pd.read_csv(updated_file,
                         usecols = [
                             'Compound',
                             'Standard Code'
                             ])
surrogates.set_index('Compound', inplace = True)
surrogates = surrogates.transpose() # need cmpds as index so they become col names
# updating the column names
surrogates.columns = [remove_commas(x) for x in surrogates.columns] # replacing all commas with underscores
surrogates.columns = [remove_text(x, replace_text_list) for x in surrogates.columns]
surrogates.columns = surrogates.columns.str.rstrip() # weird spaces after the text removal
surrogates.columns = surrogates.columns.str.lstrip()

surrogates.drop(columns = [
        # artificat of something - its in the blanks
        'decanal', 
        'nonanal', 
        # weird weird weird weird weird
        '1_3_5-trioxane', 
        '1_3_5_7-tetroxane',
        '1_3_5-trioxepane peak 1',
        '2nd peak of 1_3_5_7-tetroxane',
        '1_3_5-trioxepane 2',
        # don't trust this ID
        '4-cyanocyclohexene',
        'diacetonyl'
        ], inplace = True)
surrogates.rename(columns = { # just manually  changing the problem children
    'Dibutoxymethane' : 'dibutoxymethane',
    'pentanoic acid_ 2-ethylhexyl ester' : 'pentanoic acid_2-ethylhexyl ester',
    'homosalate 2' : 'homosalate',
    '2-tert-butylcyclohexyl acetate peak 2' : '2-tert-butylcyclohexyl acetate 2',
    '2-tert-butylcyclohexyl acetate peak 1' : '2-tert-butylcyclohexyl acetate 1',
    '4-chlorobenzotrifluoride  1' : '4-chlorobenzotrifluoride 1',
    'methylnapthalene 2' : 'methylnaphthalene 2'
    }, inplace = True)
surrogates.columns = surrogates.columns.str.rstrip() #cleaning up the final names
surrogates.columns = surrogates.columns.str.lstrip()

surrogates = surrogates.transpose()
# reseting
surrogates.reset_index(inplace = True)
surrogates.rename(columns = {'index' : 'compound'}, inplace = True)

# %% Tidying quant datafile

"""
Quant df consists of rows of samples. Each compound has 3 columns: ppb, abs_error, and relative_error.
"""

file_path = r'C:\Users\rossellr\OneDrive - Colostate\Desktop\CSU\Research\Research Projects\Field Campaigns\Frog 2023\2025\ground_data_workup\quantification\\' 
file = '_groundVOCs_all_cmpds_requant.csv'
# function will read in most recently generated file
updated_file = find_updated_csv(file_path, file)
quant_df = pd.read_csv(updated_file,
                       skiprows = [0])

# storing variables for future use
meta = pd.DataFrame({
    'start_time' : vg_meta_data['start_time'],
     'end_time' : vg_meta_data['end_time'],
     'sample_id' : quant_df['Sample ID']
     })
meta.set_index('start_time', inplace = True)

quant_df.insert(loc = 0, column = 'start_time', value = meta.index)
quant_df.drop(columns = ['Sample ID'], inplace = True)
quant_df.set_index('start_time', inplace = True)

# bc the relative error is the same for each compound, we're going to drop it.
quant_df.drop(quant_df.filter(regex = '_relerr'), axis = 1, inplace = True)
quant_df.drop(columns = ['Unit'], inplace = True)

# now, we are going to split the uncert_df from the conc_df
conc_df = quant_df.copy()
conc_df.drop(conc_df.filter(regex = '_err'), axis = 1, inplace = True)
uncert_df = quant_df.copy()
uncert_df = uncert_df.loc[:,uncert_df.columns.str.contains('_err', case = False)] # sorting for columns that contain that string value


# need uncert + conc column names to match
uncert_df.columns = uncert_df.columns.str.removesuffix('_err')

df_list = [conc_df, uncert_df]


# %% Handling the targeted and untargeted compounds seperately

"""
We need to split the targeted compounds from the untargeted compounds,
b/c we increased the PMF uncertainity on the untargeted.
"""

df_copies = [df.copy() for df in df_list] # no overwriting !

# targeted first
# fitering for cmpd names that include "Std"
df_copies = [df.filter(regex = 'Std') for df in df_copies]

target_cmpds_conc = df_copies[0] # target cmpds is the first filtered df in the df_copies list
list_target_cmpds = target_cmpds_conc.columns # list of standards
target_cmpds_uncert = df_copies[1] # target cmpds is the 2nd filtered df in the df_copies list


# untargeted next
# applying a boolean mask to sort for cmpds NOTIN the target cmpd list
untarget_cmpds_conc = conc_df.loc[:, ~conc_df.columns.isin(list_target_cmpds)].copy() # creating a new df, not just a slice
untarget_cmpds_uncert = uncert_df.loc[:, ~uncert_df.columns.isin(list_target_cmpds)].copy()

# need to move phenoyxisopropanol from untargeted to targeted
target_cmpds_conc['phenoxyisopropanol'] = untarget_cmpds_conc['phenoxyisopropanol']
target_cmpds_uncert['phenoxyisopropanol'] = untarget_cmpds_uncert['phenoxyisopropanol']

for df in [untarget_cmpds_conc, untarget_cmpds_uncert]:
    df.drop(columns = ['phenoxyisopropanol'], inplace = True)
    
# update the list in case I re-use it:
list_target_cmpds = target_cmpds_conc.columns 

# updating the column names
replace_text_list = [
    'EIC', 'RT Match', 'Std-', 'VOC3/4/', 'VOC4/', 'VOC3', 'VOC4', '/', 'Terp', 
    '( 4)', '(4)', '*', 
    '()', '( )'
    ]

full_df_list = [target_cmpds_conc, target_cmpds_uncert, untarget_cmpds_conc, untarget_cmpds_uncert]
for df in full_df_list: 
    df.columns = [remove_commas(x) for x in df.columns] # replacing all commas with underscores
    df.columns = [remove_text(x, replace_text_list) for x in df.columns]
    df.columns = df.columns.str.rstrip() # weird spaces after the text removal
    df.columns = df.columns.str.lstrip()
    
target = [target_cmpds_conc, target_cmpds_uncert]
untarget = [untarget_cmpds_conc, untarget_cmpds_uncert]

for df in untarget:
    df.drop(columns = [
        # artificat of something - its in the blanks
        'decanal', 
        'nonanal', 
        # weird weird weird weird weird
        '1_3_5-trioxane', 
        '1_3_5_7-tetroxane',
        '1_3_5-trioxepane peak 1',
        '2nd peak of 1_3_5_7-tetroxane',
        '1_3_5-trioxepane 2',
        # don't trust this ID
        '4-cyanocyclohexene',
        'diacetonyl'
        ], inplace = True)
    df.rename(columns = { # just manually  changing the problem children
        'Dibutoxymethane' : 'dibutoxymethane',
        'pentanoic acid_ 2-ethylhexyl ester' : 'pentanoic acid_2-ethylhexyl ester',
        'homosalate 2' : 'homosalate',
        '2-tert-butylcyclohexyl acetate peak 2' : '2-tert-butylcyclohexyl acetate 2',
        '2-tert-butylcyclohexyl acetate peak 1' : '2-tert-butylcyclohexyl acetate 1',
        '4-chlorobenzotrifluoride  1' : '4-chlorobenzotrifluoride 1',
        'methylnapthalene 2' : 'methylnaphthalene 2'
        }, inplace = True)
    df.columns = df.columns.str.rstrip() #cleaning up the final names
    df.columns = df.columns.str.lstrip()


# only running PMF on compounds that appear in > 25 samples

# conc dfs first
conc = [target_cmpds_conc, untarget_cmpds_conc]
conc = [df.copy() for df in conc]

filter_list = []
for i, df in enumerate(conc):
    # ignoring first 4 columns
    first_four = df.iloc[:, :4]
    # counting how many rows are only numerical
    numeric_mask = df.iloc[:, 4:].apply(pd.to_numeric, errors = 'coerce') # turns ND/<Dl into NaN
    numeric_counts = numeric_mask.count()
    # Keep columns with at least 25 numeric values
    cols_to_keep = numeric_counts[numeric_counts >= 25].index
    # Combine the first 4 columns with the filtered columns
    final = pd.concat([first_four, df[cols_to_keep]], axis = 1)
    conc[i] = final
    
    filter_list.append(final.columns)
    
common_target_cmpds_conc = conc[0]
common_untarget_cmpds_conc = conc[1]


# uncert dfs 2nd

common_target_cmpds_uncert = target_cmpds_uncert.loc[:, filter_list[0]]
common_untarget_cmpds_uncert = untarget_cmpds_uncert.loc[:, filter_list[1]]


    
# %% Exporting concentration dataframes - Skip this cell

save_list = [target_cmpds_conc, untarget_cmpds_conc]
save_list = [df.copy() for df in save_list]
df = pd.concat(save_list, axis = 1)
df = pd.concat([meta, df], axis = 1)
df.reset_index(inplace = True)

csv_name = '_groundVOCs_all_compounds_requant' # edit!
path = save_path + '\\' + today + csv_name + '.csv'  
df.to_csv(path, index = False)

save_list = [common_target_cmpds_conc, common_untarget_cmpds_conc]
save_list = [df.copy() for df in save_list]
df = pd.concat(save_list, axis = 1)
df = pd.concat([meta, df], axis = 1)
df.reset_index(inplace = True)

csv_name = '_groundVOCs_common_compounds_requant' # edit!
path = save_path + '\\' + today + csv_name + '.csv'  
df.to_csv(path, index = False)


# %% Prepping PMF input files

"""
For the concentration file, all missing data (NaN) is replaced with the median compound value, 
as per Norris et al (2012)
"""

pmf_conc_dfs = [common_target_cmpds_conc, common_untarget_cmpds_conc]

new_list = []
# getting median of all compounds to fill NaNs
for df in pmf_conc_dfs:
    for col in df.columns:
        df[col] = df[col].apply(pd.to_numeric, errors = "coerce") # all the values need to be numeric
    df = pd.DataFrame(df.median(numeric_only = True))
    new_list.append(df)

cmpd_medians_ppb = pd.concat(new_list)
cmpd_medians_ppb.rename(columns = {0 : 'cmpd_medians_ppb'}, inplace = True)


# replacing NaNs with median sample value
for df in pmf_conc_dfs:
    for col in df.columns:
        fill_value = cmpd_medians_ppb.loc[col, 'cmpd_medians_ppb']
        df[col] = df[col].fillna(fill_value)


# we now do not have any values lower than any DL, thank the literal lord. 
# Plan was to set concentrations < DL were set to half the DL, as per Yuan et al (2012), Debevec et al (2017), Hopeke (2000)
# code was previously run as follows: 
    # lods = cmpd_refs.copy()
    # lods = lods[['compound_name', 'lod']]
    # lods.set_index('compound_name', inplace = True)
    # lod_values = lods.loc[all_cmpds_ppb.columns]

    # # concentrations < DL were set to half the DL, as per Yuan et al (2012), Debevec et al (2017), Hopeke (2000)
    # for col in all_cmpds_ppb.columns:
    #     fill_value = lod_values.loc[col, 'lod']
    #     all_cmpds_ppb[col] = all_cmpds_ppb[col].where(all_cmpds_ppb[col] >= fill_value, other = fill_value * 0.5)
        
    # all_cmpds_ppb.insert(0, column = 'start_time_edt', value = start_time, allow_duplicates = False) # sticking the time column back up front

# however, that code assumes our LODs were in ppb, WHICH THEY AREN'T. So if needed, will need to assume similar sampling concentrations.

pmf_input_conc = pd.concat(pmf_conc_dfs, axis = 1, verify_integrity = True)
pmf_input_conc.reset_index(inplace = True, drop = True)


"""
For the uncertainity file:
    Uncertainites were handled as per Yuan et al (2012) and Debevec et al (2017):
    Nans = 4x median compound concentration 
    If concentrations were < LOD, replaced uncertainities w/ 5/6 lod value
"""

pmf_uncert_dfs = [common_target_cmpds_uncert, common_untarget_cmpds_uncert]

for df in pmf_uncert_dfs:
    for col in df.columns:
        df[col] = df[col].apply(pd.to_numeric, errors = "coerce") # all the values need to be numeric
        fill_value = cmpd_medians_ppb.loc[col, 'cmpd_medians_ppb'] # filling all NaN values with the median cmpd uncert.
        #fill_value = fill_value * 4
        # missing values = 4 * median concentration
        df[col] = (df[col].fillna(fill_value) * 4)


# Again, we don't have any values <DL in ng/tube, so don't need to replace any values. 
# code was previously run as follows: 
# for col in all_cmpds_uncert:
#     check_value = lod_values.loc[col, 'lod']
#     fill_value = check_value * 5/6
#     mask = all_cmpds_ppb[col] < check_value
#     all_cmpds_uncert.loc[mask, col] = fill_value
# however, that code assumes our LODs were in ppb, WHICH THEY AREN'T. So if needed, will need to assume similar sampling concentrations.

pmf_input_uncert = pd.concat(pmf_uncert_dfs, axis = 1, verify_integrity = True)
pmf_input_uncert.reset_index(inplace = True, drop = True)

# gotta put back in the start_times:
for df in [pmf_input_conc, pmf_input_uncert]:
    df.insert(loc = 0,
              column = 'start_time',
              value = meta.index,
              allow_duplicates = False
              )

# check uncertainity and conc columns match:
test1 = pmf_input_conc.columns
test2 = pmf_input_uncert.columns
check = test1 == test2
print(check)



# %% saving the pmf matrices # skip this cell if not saving


pmf_save_path = r'C:\\Users\\rossellr\\OneDrive - Colostate\\Desktop\\CSU\\Research\\Research Projects\\Field Campaigns\\Frog 2023\\2025\\ground_data_workup\\pmf\\input\\'
csv_name = '_pmf_conc_reanalysis' # edit!
path = pmf_save_path + '\\' + today + csv_name + '.csv'  

# uncomment to save ! change df as needed
pmf_input_conc.to_csv(path, index = False)

csv_name = '_pmf_uncert_reanalysis' # edit!
path = pmf_save_path + '\\' + today + csv_name + '.csv'  

# uncomment to save ! change df as needed
pmf_input_uncert.to_csv(path, index = False)


# %% Preparing stats table of quantified compounds 
# need CAS, formula, surrogate code, cal method 

conc = [target_cmpds_conc, untarget_cmpds_conc]
conc = [df.copy() for df in conc]

list_target_cmpds = target_cmpds_conc.columns

for df in conc:
    for col in df.columns:
        df[col] = df[col].apply(pd.to_numeric, errors = "coerce")
    df.replace('ND', np.nan, inplace = True) # gotta replace all the ND w/ NaN
    df.columns = df.columns.str.rstrip()
    df.columns = df.columns.str.lstrip()

all_cmpds_ppb = pd.concat(conc, axis = 1)
# need to drop datetime index for the stats table
all_cmpds_ppb.reset_index(inplace = True, drop = True) 

stats_table = all_cmpds_ppb.describe()
stats_table = stats_table.transpose()
stats_table.reset_index(inplace = True)
stats_table.rename(columns = {'index' : 'compounds', '50%' : 'median'}, inplace = True)

#  add column about calibration method
stats_table['cal_method'] = stats_table['compounds'].isin(target_cmpds_conc[list_target_cmpds])
stats_table.replace({True : 'direct', False : 'indirect'}, inplace = True)

# moving df columns to the front if needed
stats_table = stats_table[['cal_method'] + [col for col in stats_table.columns if col != 'cal_method']]

# arranging table by alphabetical order 
stats_table.sort_values('compounds', ascending = True, inplace= True)

stats_table.reset_index(drop = True, inplace = True)

# adding in surrogate codes
surrogate = surrogates.copy() # this keeps getting overwritten during debuging
surrogate.set_index('compound', inplace = True)

stats_table['standard code'] = stats_table['compounds'].map(surrogate['Standard Code'])

# adding in the other compound information (CAS, formula, etc)

compound_references = cmpd_ref_data.copy() # no overwriting!
compound_references.set_index('compound', inplace = True)

stats_table['CAS'] = stats_table['compounds'].map(compound_references['CAS'])
stats_table['molecular formula'] = stats_table['compounds'].map(compound_references['Molecular Formula'])

# organizing the columns
stats_table = stats_table[['molecular formula'] + [col for col in stats_table.columns if col != 'molecular formula']]
stats_table = stats_table[['CAS'] + [col for col in stats_table.columns if col != 'CAS']]
stats_table = stats_table[['compounds'] + [col for col in stats_table.columns if col != 'compounds']]
stats_table = stats_table[[col for col in stats_table.columns if col != 'cal_method'] + ['cal_method']]


# %% Exporting Stats Table - Skip this cell

df = stats_table.copy()

csv_name = 'full_cmpd_stats_table'
path = save_path + '\\' + today + '_' + csv_name + '.csv' 

# uncomment to save !
df.to_csv(path, float_format="%.4f", index = False)



