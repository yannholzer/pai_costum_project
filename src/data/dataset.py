from torch.utils.data import Dataset
import pandas as pd
import numpy as np

DISK_COLUMNS = ["metallicity", "gas disk (Msun)", "solid disk (Mearth)", "life time (yr)", "luminosity"]



class DiskPlanetDataset(Dataset):
    def __init__(self, path):
        
        #data = np.load(path, allow_pickle=True)
        self.disk_data, self.labels, self.feature_name, self.labels_names, self.mins_maxs = np.load(path, allow_pickle=True)
        
        self.disk_data = self.disk_data.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
        
          
                

    def __len__(self):
        return len(self.disk_data)
    
    def input_len(self):
        return self.disk_data.shape[1]
    
    def output_len(self):
        return self.labels.shape[1]
    
    def labels_shape(self):
        return self.labels.shape
    
    def __getitem__(self, idx):
        
        return self.disk_data[idx], self.labels[idx]
    
    
    def permute_feature(self, feature_to_permute):
        
        if feature_to_permute < 0 or feature_to_permute > len(DISK_COLUMNS):
            raise ValueError(f"Feature to permute must be between 0 and {len(DISK_COLUMNS)}")
        
        feature_name = DISK_COLUMNS[feature_to_permute]
        
        self.disk_data[feature_name] = self.disk_data[feature_name].sample(frac=1).values
        
        return feature_name

        
    
    
    
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        

