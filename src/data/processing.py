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

# log scale the features with epsilon value to offset the log(0) issue
np_disk = np.zeros((df_disk.shape[0], len(DISK_COLUMNS)))
for i, col in enumerate(DISK_COLUMNS):
    np_disk[:, i] = df_disk[col].values

np_disk = np_disk[:, :-1]

np_disk = np.log10(EPSILON+np_disk)


if labels_type == "mass_and_count":
# get the labels: total planet mass and planet count

    count, count_label = data.get_planet_counts(df)
    masses, masses_label = data.get_total_planets_mass(df)
    df_label = data.join_dataframe(count, masses)
    
    
    labels_names = np.array([count_label, masses_label], dtype=str)
    np_label = np.zeros((df_label.shape[0], len(labels_names)))
    
    for i, col in enumerate(labels_names):
        np_label[:, i] = df_label[col].values
        
    
    # had fix for memory issue when changing the lenght of an array str element
    masses_label = f"log_{masses_label}"
    labels_names = np.array([count_label, masses_label], dtype=str)
            
    # log scale the label total masses
    
    np_label[:, 1] = np.log10(EPSILON+np_label[:, 1])
    

else:
    print(labels_type)
    print("Labels type not recognized")
    sys.exit(1)


# rescale the labels between -1 and 1

min_maxs = np.zeros((np_label.shape[1], 2))


mins, maxs = np.min(np_label, axis=0), np.max(np_label, axis=0)
min_maxs[:] = mins, maxs
np_label = 2* (np_label - mins) / (maxs - mins) - 1






# Split the data into training, validation, and test sets
train_data, val_test_data, train_label, val_test_label = train_test_split(np_disk, np_label, train_size=train_ratio, random_state=seed)
val_data, test_data, val_label, test_label = train_test_split(val_test_data, val_test_label, train_size=val_ratio, random_state=seed)

train_mean = np.mean(train_data, axis=0)
train_std = np.std(train_data, axis=0)

train_data = (train_data - train_mean) / train_std
val_data = (val_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std



train_path = os.path.join(processing_path, "train.npy")
val_path = os.path.join(processing_path, "val.npy")
test_path = os.path.join(processing_path, "test.npy")


train = np.array([train_data, train_label, DISK_COLUMNS[:-1], labels_names, min_maxs], dtype=object)
val = np.array([val_data, val_label, DISK_COLUMNS[:-1], labels_names, min_maxs], dtype=object)
test = np.array([test_data, test_label, DISK_COLUMNS[:-1], labels_names, min_maxs], dtype=object)

np.save(train_path, train)
np.save(val_path, val)
np.save(test_path, test)

