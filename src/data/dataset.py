from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np


class DiskPlanetDataset(Dataset):
    def __init__(self, path):
        
        data = np.load(path, allow_pickle=True)
            
        self.disk_data = data[0].astype(np.float32)
        
        self.labels = data[1].astype(np.float32)
        
        self.label_names = data[2]
                

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
        
    
    
    
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        

