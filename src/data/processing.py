import data
from sklearn.model_selection import train_test_split
import os
import numpy as np
import sys
from utils import get_processing_args
import pandas as pd


# get the arguments
argv = get_processing_args()

data_path = argv.data_path
seed = argv.seed
processing_path = argv.processed_path
labels_type = argv.labels_type



os.makedirs(processing_path, exist_ok=True)

# get the raw dataset
df = data.get_dataframe_single_file(data_path)

# get the features
df_disk = data.get_disk_dataframe(df)

# log scale the features with epsilon value to offset the log(0) issue
epsilon = 1e-5 
df_disk = np.log10(epsilon+df_disk)


if labels_type == "mass_and_count":
# get the labels: total planet mass and planet count
    count = data.get_planet_counts(df)
    masses = data.get_total_planets_mass(df)
    df_label = data.join_dataframe(count, masses)
    label_names = np.array(["Number of planets", "Total planets mass (Mearth)"])
    
    # log scale the label total masses
    df_label["Total planets mass (Mearth)"] = np.log10(epsilon+df_label["Total planets mass (Mearth)"])
    

else:
    print(labels_type)
    print("Labels type not recognized")
    sys.exit(1)


# rescale the labels between -1 and 1
df_label = 2* (df_label - df_label.min()) / (df_label.max() - df_label.min()) - 1


# Split the data into training, validation, and test sets
train_data, val_test_data, train_label, val_test_label = train_test_split(df_disk, df_label, test_size=0.33, random_state=seed)
val_data, test_data, val_label, test_label = train_test_split(val_test_data, val_test_label, test_size=0.33, random_state=seed)

train_mean = train_data.mean(axis=0)
train_std = train_data.std(axis=0)

train_data = (train_data - train_mean) / train_std
val_data = (val_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std





np.save(os.path.join(processing_path, "train_data.npy"), np.array([train_data.to_numpy(), train_label.to_numpy(), label_names],dtype=object))
np.save(os.path.join(processing_path, "val_data.npy"), np.array([val_data.to_numpy(), val_label.to_numpy(), label_names],dtype=object))
np.save(os.path.join(processing_path, "test_data.npy"), np.array([test_data.to_numpy(), test_label.to_numpy(), label_names],dtype=object))
