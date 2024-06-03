from src.data import data
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np



class DiskPlanet_singleStar_massesAndCount(Dataset):
    def __init__(self, disk_data:pd.DataFrame, labels:pd.DataFrame):
            
        self.disk_data = disk_data
        
        self.labels = labels
                

    def __len__(self):
        return len(self.disk_data)
    
    
    def __getitem__(self, idx):
        
        return self.disk_data[idx], self.labels[idx]
        
        
    # def add_label_total_planets_mass(self):
    #     total_planets_mass = data.get_total_planets_mass(self.df)
        
    #     self.labels = data.join_dataframe(self.labels, total_planets_mass)
        
    #     return self
        
    # def add_label_planets_count(self):
    #     planet_counts = data.get_planet_counts(self.df)
        
    #     self.labels = data.join_dataframe(self.labels, planet_counts)
    
    #     return self
        
        
    def process(self):
        
        # process the disk of the data, set up in log scale
        
        self.disk_data = np.log10(1e-5+self.disk_data)
        
        # process the labels, set up in log scale for the masses
        self.labels["Total planets mass (Mearth)"] = np.log10(1e-5+self.labels["Total planets mass (Mearth)"])
        
        
        
        return self
    
    
    def normalize(self, mean, std):

        self.disk_data = (self.disk_data - mean) / std
        
        return self
    
    def to_numpy(self):
        self.disk_data = self.disk_data.to_numpy().astype(np.float32)
        self.labels = self.labels.to_numpy().astype(np.float32)
        
        return self
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        

