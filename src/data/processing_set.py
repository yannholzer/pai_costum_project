import data
from sklearn.model_selection import train_test_split
import os
import numpy as np
import sys
from utils import get_processing_args
import pandas as pd

PLANETS_COLUMNS = np.array(["Total Mass (Mearth)", "sma (AU)", "GCR (gas to core ratio)", "fraction water", "sma ini", "radius", "System number"])
DISK_COLUMNS = np.array(["metallicity", "gas disk (Msun)", "solid disk (Mearth)", "life time (yr)", "luminosity", "System number"])
EPSILON = 1e-5 


# get the arguments
argv = get_processing_args()

data_path = argv.data_path
seed = argv.seed
processing_path = argv.processed_path
labels_type = argv.labels_type

val_ratio = argv.validation_ratio
train_ratio = argv.train_ratio


os.makedirs(processing_path, exist_ok=True)

# get the raw dataset
df = data.get_dataframe_single_file(data_path)

# get the features
df_disk = data.get_disk_dataframe(df, drop_system_number=False)





if labels_type == "SET_all":
    df_planets = data.get_planets_dataframe(df)
    
    planet_column = PLANETS_COLUMNS

    
    # df_label = df_planets[["Total Mass (Mearth)", "System number"]]
    
    # df_label["Total Mass (Mearth)"] = np.log10(EPSILON+df_label["Total Mass (Mearth)"])
    
    # df_label["Total Mass (Mearth)"] = 2* (df_label["Total Mass (Mearth)"] - df_label["Total Mass (Mearth)"].min()) / (df_label["Total Mass (Mearth)"].max() - df_label["Total Mass (Mearth)"].min()) - 1
    # label_names = np.array(["Total_Mass_(Mearth)"])

    np_planets = np.zeros((df_planets.shape[0], len(PLANETS_COLUMNS)))    
    
    for i, col in enumerate(PLANETS_COLUMNS):
        np_planets[:, i] = df_planets[col].values
    
    # make sure the disk features are ordered like the planet columns 
    np_disk = np.zeros((df_disk.shape[0], len(DISK_COLUMNS)))
    for i, col in enumerate(DISK_COLUMNS):
        np_disk[:, i] = df_disk[col].values
    
    
    np_planet_grouped = np.zeros((np_disk.shape[0], 20, len(PLANETS_COLUMNS)))
    
    for i_sys, sysN in enumerate(np_disk[:, -1]):
        p = np_planets[np_planets[:, -1] == sysN]
        np_planet_grouped[i_sys, :len(p)] = p
        
    np_disk = np_disk[:, :-1]
    
    np_planet_grouped = np_planet_grouped[:, :-1]
    
elif labels_type == "SET_masses":
    df_planets = data.get_planets_dataframe(df)
    
    # df_label = df_planets[["Total Mass (Mearth)", "System number"]]
    
    # df_label["Total Mass (Mearth)"] = np.log10(EPSILON+df_label["Total Mass (Mearth)"])
    
    # df_label["Total Mass (Mearth)"] = 2* (df_label["Total Mass (Mearth)"] - df_label["Total Mass (Mearth)"].min()) / (df_label["Total Mass (Mearth)"].max() - df_label["Total Mass (Mearth)"].min()) - 1
    # label_names = np.array(["Total_Mass_(Mearth)"])
    planet_column = ["Total Mass (Mearth)", "System number"]


    np_planets = np.zeros((df_planets.shape[0], len(planet_column)))    
    for i, col in enumerate(planet_column):
        np_planets[:, i] = df_planets[col].values
        
    
    np_disk = np.zeros((df_disk.shape[0], len(DISK_COLUMNS)))
    for i, col in enumerate(DISK_COLUMNS):
        np_disk[:, i] = df_disk[col].values
    
    
    np_planet_grouped = np.zeros((np_disk.shape[0], 20, len(planet_column)))
    np_planet_grouped.fill(np.nan)
    
    for i_sys, sysN in enumerate(np_disk[:, -1]):
        p = np_planets[np_planets[:, -1] == sysN]
        np_planet_grouped[i_sys, :len(p)] = p
            
        
    np_disk = np_disk[:, :-1] # drop system number
    
    np_planet_grouped = np_planet_grouped[:, :, :-1] # drop system number
    
    
    
elif labels_type == "SET_FLAG_masses":
    df_planets = data.get_planets_dataframe(df)
    
    # df_label = df_planets[["Total Mass (Mearth)", "System number"]]
    
    # df_label["Total Mass (Mearth)"] = np.log10(EPSILON+df_label["Total Mass (Mearth)"])
    
    # df_label["Total Mass (Mearth)"] = 2* (df_label["Total Mass (Mearth)"] - df_label["Total Mass (Mearth)"].min()) / (df_label["Total Mass (Mearth)"].max() - df_label["Total Mass (Mearth)"].min()) - 1
    # label_names = np.array(["Total_Mass_(Mearth)"])
    planet_column = ["Total Mass (Mearth)", "System number"]


    np_planets = np.zeros((df_planets.shape[0], len(planet_column)))    
    for i, col in enumerate(planet_column):    
        np_planets[:, i] = df_planets[col].values
    
    
    planet_column = ["exist_flag", "Total Mass (Mearth)", "System number"]
    
    np_disk = np.zeros((df_disk.shape[0], len(DISK_COLUMNS)))
    for i, col in enumerate(DISK_COLUMNS):
        np_disk[:, i] = df_disk[col].values
    
    
    np_planet_grouped = np.zeros((np_disk.shape[0], 20, len(planet_column)))
    np_planet_grouped.fill(np.nan)
    np_planet_grouped[:,:, 0] = -1
    for i_sys, sysN in enumerate(np_disk[:, -1]):
        p = np_planets[np_planets[:, -1] == sysN]
        np_planet_grouped[i_sys,:len(p), 0] = 1
        np_planet_grouped[i_sys, :len(p), 1:] = p
        
            
        
    np_disk = np_disk[:, :-1] # drop system number
    
    np_planet_grouped = np_planet_grouped[:, :, :-1] # drop system number

    

else:
    print(labels_type)
    print("Labels type not recognized")
    sys.exit(1)


# for i_sys, sysN in enumerate(df_disk["System number"]):
#     print(20*"=")
#     print(i_sys, sysN)
#     print(df_disk[df_disk["System number"] == sysN])
#     print(df_planets_subgroup[i_sys])
#     #print(df_label[df_label["System number"] == sysN])




# log scale the features with epsilon value to offset the log(0) issue
np_disk = np.log10(EPSILON+np_disk)

min_maxs = np.zeros((np_planet_grouped.shape[2], 2))
for i, n in enumerate(planet_column[:-1]):
    if n == "exist_flag":
        continue
    np_planet_grouped[:,:,i] = np.log10(EPSILON+np_planet_grouped[:,:,i])
    
    nanmin = np.nanmin(np_planet_grouped[:,:,i])
    nanmax = np.nanmax(np_planet_grouped[:,:,i])
        
    min_maxs[i] = nanmin, nanmax
    
    
    np_planet_grouped[:,:,i] = 2* (np_planet_grouped[:,:,i] - nanmin) / (nanmax - nanmin) - 1
    



#print(np.nanmean(np_planet_grouped, axis=2), np.nanstd(np_planet_grouped, axis=2))
# set the missing value to -2. Not sure it works, will see
np_planet_grouped[np.isnan(np_planet_grouped)] = -2
#print(np.nanmean(np_planet_grouped, axis=2), np.nanstd(np_planet_grouped, axis=2))


# Split the data into training, validation, and test sets
train_data, val_test_data, train_label, val_test_label = train_test_split(np_disk, np_planet_grouped, train_size=train_ratio, random_state=seed)
val_data, test_data, val_label, test_label = train_test_split(val_test_data, val_test_label, train_size=val_ratio, random_state=seed)

train_mean = train_data.mean(axis=0)
train_std = train_data.std(axis=0)

train_data = (train_data - train_mean) / train_std
val_data = (val_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std






train = np.array([train_data, train_label, DISK_COLUMNS[:-1], planet_column[:-1], min_maxs], dtype=object)
val = np.array([val_data, val_label, DISK_COLUMNS[:-1], planet_column[:-1], min_maxs], dtype=object)
test = np.array([test_data, test_label, DISK_COLUMNS[:-1], planet_column[:-1], min_maxs], dtype=object)

train_path = os.path.join(processing_path, "train.npy")
val_path = os.path.join(processing_path, "val.npy")
test_path = os.path.join(processing_path, "test.npy")

np.save(train_path, train)
np.save(val_path, val)
np.save(test_path, test)