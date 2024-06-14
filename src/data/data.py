import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Union
import os


PLANETS_COLUMNS = ["Total Mass (Mearth)", "sma (AU)", "GCR (gas to core ratio)", "fraction water", "sma ini", "radius"]
#DISK_COLUMNS = ["metallicity", "gas disk (Msun)", "solid disk (Mearth)", "life time (yr)", "luminosity", "star mass (solar mass)"]
DISK_COLUMNS = ["metallicity", "gas disk (Msun)", "solid disk (Mearth)", "life time (yr)", "luminosity"]

N_BINS = 50

# panda dataframe utiliy functions 

def get_dataframe_single_file(data_path:str) -> pd.DataFrame:
    """Take the path of a csv file and return a pandas dataframe.

    Parameters
    ----------
    data_path : str
        path to the csv file

    Returns
    -------
    pd.DataFrame
        return the scv table as a pandas dataframe
    """
    
    dataframe = pd.read_csv(data_path)
    dataframe = dataframe.replace(-1000, np.nan).dropna()
        
    return dataframe


def get_dataframe_from_folder(data_path:str) -> pd.DataFrame:
    """Take the path of a folder containing all cvs files and return a pandas dataframe.

    Parameters
    ----------
    data_path : str
        path to the folder

    Returns
    -------
    pd.DataFrame
        return the scv tables as a pandas dataframe
    """
    
    dataframe = None
    for files in os.listdir(data_path):
        print("opening file: ", files)
        path = os.path.join(data_path, files)
    
        if dataframe is None:
            dataframe = pd.read_csv(path)
        else:
            df_temp = pd.read_csv(path)
            df_temp["System number"] += dataframe["System number"].max() + 1
            dataframe = pd.concat([dataframe, df_temp])
            

    dataframe = dataframe.replace(-1000, np.nan).dropna()
        
    return dataframe


def get_disk_dataframe(df:pd.DataFrame, drop_duplicate=True, drop_system_number=True) -> pd.DataFrame:
    """Take the dataframe and return a dataframe containing the disk properties.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the disk and planets properties

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Return one pandas dataframe, the disk properties.
    """
    
        
    # hardcoded columns names of the planets and disk properties
    if drop_duplicate:
        dataframe_disk_propriety = df.drop_duplicates(subset='System number').drop(columns=PLANETS_COLUMNS).reset_index(drop=True)
    else:
        dataframe_disk_propriety = df.drop(columns=PLANETS_COLUMNS).reset_index(drop=True)
        
    if drop_system_number:
        dataframe_disk_propriety = dataframe_disk_propriety.drop(columns="System number")
    
    
    
    return dataframe_disk_propriety


def get_planets_dataframe(df:pd.DataFrame, drop_duplicate=True) -> pd.DataFrame:
    """Take the dataframe and return the planets properties.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the disk and planets properties

    Returns
    -------
    pd.DataFrame
        Return the planets properties.
    """
    
        
    # hardcoded columns names of the planets and disk properties
    
   
    dataframe_planets_propriety = df.drop(columns=DISK_COLUMNS).reset_index(drop=True)
    
    
    return dataframe_planets_propriety


def get_disk_planets_dataframe(df:pd.DataFrame, drop_duplicate=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Take the dataframe and return two pandas dataframe, one for the disk properties and one for the planets properties.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the disk and planets properties

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Return two pandas dataframe, one for the disk properties and one for the planets properties.
    """
    
        
    # hardcoded columns names of the planets and disk properties
    
    if drop_duplicate:
        dataframe_disk_propriety = df.drop_duplicates(subset='System number').drop(columns=PLANETS_COLUMNS).reset_index(drop=True)
    else:
        dataframe_disk_propriety = df.drop(columns=PLANETS_COLUMNS).reset_index(drop=True)
    
    dataframe_planets_propriety = df.drop(columns=DISK_COLUMNS).reset_index(drop=True)
    
    
    return dataframe_disk_propriety, dataframe_planets_propriety


                
def get_planet_counts(df:pd.DataFrame) -> pd.DataFrame:
    """Return the number of planets in each system.

    Parameters
    ----------
    df : pd.DataFrame
        the planets dataframe

    Returns
    -------
    pd.DataFrame
        return the number of planets in each system.
    """
    planet_counts_name = "Number_of_planets"
    planet_counts = df.groupby("System number").size().reset_index(drop=True).to_frame().rename(columns={0: planet_counts_name})
    
    return planet_counts, planet_counts_name


def get_total_planets_mass(df:pd.DataFrame) -> pd.DataFrame:
    """Return the total mass of the planets in each system.

    Parameters
    ----------
    df : pd.DataFrame
        the planets dataframe

    Returns
    -------
    pd.DataFrame
        return the total mass of the planets in each system.
    """
    total_mass_name = "Total_planets_mass_(Mearth)"
    total_mass = df.groupby("System number")["Total Mass (Mearth)"].sum().reset_index(drop=True).to_frame().rename(columns={"Total Mass (Mearth)": total_mass_name})
    
    return total_mass, total_mass_name


def join_dataframe(df1:Union[None, pd.DataFrame], df2:pd.DataFrame) -> pd.DataFrame:
    """Join two dataframe into one, or return the dataframe if only one is given.
    
    Parameters
    ----------
    df1 : pd.DataFrame
        the first dataframe to join, if None return df2
    
    df2 : Union[None, pd.DataFrame]
        the second dataframe to join
        
    Returns
    -------
    pd.DataFrame
        the joined dataframe, if df1 is None return df2
    """
    
    if df1 is None:
        return df2
    
    return df1.join(df2)
    
    
    
def plot_disk_features_histogram(df_disk:pd.DataFrame, name_disk:Tuple[str,...]=DISK_COLUMNS, log:bool=True):
    
    
    for c in name_disk:
        
        fig, ax = plt.subplots(1,2, figsize=(8, 3))
        
        
        ax[0].hist(df_disk[c], bins=N_BINS)
        ax[0].set(
            title = f"disk property: {c}",
            xlabel=c,
            ylabel="count"
            )
        # lower_quantile = df_disk[c].quantile(0.1)
        # upper_quantile = df_disk[c].quantile(0.9)

        if log:
            # data_quantile = df_disk[c][(df_disk[c] >= lower_quantile) & (df_disk[c] <= upper_quantile)]
            ax[1].hist(np.log(1e-5+df_disk[c]), bins=N_BINS)
            ax[1].set(
                title = f"log planet property: {c}",
                xlabel=f"log {c}",
                ylabel="count",
                #xscale="log"
                )
            
        plt.tight_layout()
        plt.show()
        



def plot_planets_features_histogram(df_planets:pd.DataFrame, name_planets:Tuple[str,...]=PLANETS_COLUMNS, log:bool=True):
    
    for c in name_planets:
        fig, ax = plt.subplots(1,2, figsize=(8, 3))
        
        ax[0].hist(df_planets[c], bins=N_BINS)
        ax[0].set(
            title = f"planet property: {c}",
            xlabel=c,
            ylabel="count"
            )

        if log:
            ax[1].hist(np.log(1e-5+df_planets[c]), bins=N_BINS)
            ax[1].set(
                title = f"log disk property: {c}",
                xlabel=f"log {c}",
                ylabel="count",
                #xscale="log"
                )

        plt.tight_layout()
        plt.show()
    

def corner_plot_disk_features(df_disk:pd.DataFrame, name_disk:Tuple[str,...]=DISK_COLUMNS, log:bool=False):
    corner_plot(df_disk, name_disk, log)


def corner_plot_planets_features(df_planets:pd.DataFrame, name_planets:Tuple[str,...]=PLANETS_COLUMNS, log:bool=False):
    corner_plot(df_planets, name_planets, log)
    
    
    
def corner_plot_all_features(df, log:bool):
  
    df_disk, df_planets = get_disk_planets_dataframe(df, drop_duplicate=False)
    
    q_width = 10
    if log:
        df_planets+=1e-100
        df_disk+=1e-100
    fig, ax = plt.subplots(len(PLANETS_COLUMNS), len(DISK_COLUMNS), figsize=(18, 18))
    for i_c1, c1 in enumerate(PLANETS_COLUMNS):
        for i_c2, c2 in enumerate(DISK_COLUMNS):
            if log:
                ax[i_c1, 0].set(ylabel=f"planets: log {c1}")
                ax[-1, i_c2].set(xlabel=f"disk: log {c2}")
            else:
                ax[i_c1, 0].set(ylabel=f"planet: {c1}")
                ax[-1, i_c2].set(xlabel=f"disk: {c2}")
            
            
            m_c1 = df_planets[c1].quantile(0.5)
            lq_c1 = df_planets[c1].quantile(0.1)
            uq_c1 = df_planets[c1].quantile(0.9) 
    
            m_c2 = df_disk[c2].quantile(0.5)
            lq_c2 = df_disk[c2].quantile(0.1)
            uq_c2 = df_disk[c2].quantile(0.9)
            
            if log:
                
                ax[i_c1, i_c2].scatter(df_disk[c2], df_planets[c1], s=0.1, alpha=0.1, color="black")
                ax[i_c1, i_c2].set(xlim=(lq_c2/q_width, uq_c2*q_width), ylim=(lq_c1/q_width, uq_c1*q_width))
                
            else:
                ax[i_c1, i_c2].scatter(df_disk[c2], df_planets[c1], s=0.1, alpha=0.1, color="black")
                ax[i_c1, i_c2].set(xlim=(m_c2 - 3*(m_c2 -lq_c2), m_c2 + 3*(m_c2 +uq_c2)), 
                                    ylim=(m_c1 - 3*(m_c1 -lq_c1), m_c1 + 3*(m_c1 +uq_c1))
                                    )
            if log:
                ax[i_c1, i_c2].set(xscale='log', yscale='log')


    plt.tight_layout()
    plt.show()
    
    if log:
        df_planets-=1e-100
        df_disk-=1e-100

    
    
def corner_plot(df:pd.DataFrame, names:Tuple[str,...], log:bool):    
    
    q_width = 10
    if log:
        df+=1e-100
    fig, ax = plt.subplots(len(names), len(names), figsize=(18, 18))
    for i_c1, c1 in enumerate(names):
        for i_c2, c2 in enumerate(names):
            if log:
                ax[i_c1, 0].set(ylabel=f"log {c1}")
                ax[-1, i_c2].set(xlabel=f"log {c2}")
            else:
                ax[i_c1, 0].set(ylabel=f"{c1}")
                ax[-1, i_c2].set(xlabel=f"{c2}")
            if i_c1 == i_c2:
                lower_quantile = df[c1].quantile(0.1)
                upper_quantile = df[c1].quantile(0.9)
                if log:
                    bins = 10**np.linspace(np.log10(df[c1].min()), np.log10(df[c1].max()), N_BINS)
                    ax[i_c1, i_c2].hist(df[c1], bins=bins, color="black", fill=False)
                    ax[i_c1, i_c2].set(xscale='log')
                    ax[i_c1, i_c2].axvline(lower_quantile, c="red")
                    ax[i_c1, i_c2].axvline(upper_quantile, c="red")
                    
                else:
                    ax[i_c1, i_c2].hist(df[c1], bins=N_BINS, color="black", fill=False)
                    ax[i_c1, i_c2].axvline(lower_quantile, c="red")
                    ax[i_c1, i_c2].axvline(upper_quantile, c="red")
                
                
            elif i_c1 > i_c2:
                
                
                m_c1 = df[c1].quantile(0.5)
                lq_c1 = df[c1].quantile(0.1)
                uq_c1 = df[c1].quantile(0.9) 
        
                m_c2 = df[c2].quantile(0.5)
                lq_c2 = df[c2].quantile(0.1)
                uq_c2 = df[c2].quantile(0.9)
                
                if log:
                    
                    ax[i_c1, i_c2].scatter(df[c2], df[c1], s=0.1, alpha=0.1, color="black")
                    ax[i_c1, i_c2].set(xlim=(lq_c2/q_width, uq_c2*q_width), ylim=(lq_c1/q_width, uq_c1*q_width))
                    
                else:
                    ax[i_c1, i_c2].scatter(df[c2], df[c1], s=0.1, alpha=0.1, color="black")
                    ax[i_c1, i_c2].set(xlim=(m_c2 - 3*(m_c2 -lq_c2), m_c2 + 3*(m_c2 +uq_c2)), 
                                       ylim=(m_c1 - 3*(m_c1 -lq_c1), m_c1 + 3*(m_c1 +uq_c1))
                                       )
                if log:
                    ax[i_c1, i_c2].set(xscale='log', yscale='log')
            else:
                fig.delaxes(ax[i_c1, i_c2])
    
    plt.tight_layout()
    plt.show()
    
    if log:
        df-=1e-100
                