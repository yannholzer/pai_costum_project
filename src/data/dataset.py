from src.data import data
from torch.utils.data import Dataset


class CostumDataset(Dataset):
    def __init__(self, data_path:str):
        
        self.df  = data.get_dataframe(data_path)
        
        self.df_disk = data.get_disk_dataframe(self.df)
        
        self.labels = None
                
        

        
    def __len__(self):
        return len(self.df_disk)
    
    
    def __getitem__(self, idx):
        if self.labels is None:
            raise ValueError('No labels have been added to the dataset')
        
        return self.df_disk.iloc[idx], self.labels.iloc[idx]
        
        
    def add_label_total_planets_mass(self):
        total_planets_mass = data.get_total_planets_mass(self.df)
        
        self.labels = data.join_dataframe(self.labels, total_planets_mass)
        
    def add_label_planets_count(self):
        planet_counts = data.get_planet_counts(self.df)
        
        self.labels = data.join_dataframe(self.labels, planet_counts)
        
        
        
        
        
        
        
        

