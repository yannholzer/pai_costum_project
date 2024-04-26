import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Union


PLANET_COLUMNS = ["Total Mass (Mearth)", "sma (AU)", "GCR (gas to core ratio)", "fraction water", "sma ini", "radius"]
DISK_COLUMNS = ["metallicity", "gas disk (Msun)", "solid disk (Mearth)", "life time (yr)", "luminosity"]


def get_dataframe(data_path:str) -> pd.DataFrame:
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


def get_disk_dataframe(df:pd.DataFrame) -> pd.DataFrame:
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
    
    dataframe_disk_propriety = df.drop_duplicates(subset='System number').drop(columns=PLANET_COLUMNS).reset_index(drop=True)
    
    
    
    return dataframe_disk_propriety



def get_disk_planets_dataframe(df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    
    dataframe_disk_propriety = df.drop_duplicates(subset='System number').drop(columns=PLANET_COLUMNS).reset_index(drop=True)
    
    dataframe_planets_propriety = df.drop(columns=DISK_COLUMNS).reset_index(drop=True)
    
    
    return dataframe_disk_propriety, dataframe_planets_propriety



def plot_features(columns:Tuple[str,...], df:pd.DataFrame, name:str, scale:str="linear", return_fig:bool=False) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """Plot or return the histogram of the columns of the disk or the planets dataframes.

    Parameters
    ----------
    columns : Tuple[str,...]
        The tuple containing the columns names to plot
    df : pd.DataFrame
        the dataframe of the disks or planets to plots
    name : str
        the name of the dataframe to plot (disk or planets)
    scale : str, optional
        the scale to plot the histogram, by default "linear"
    return_fig : bool, optional
        choose to return the fig ax instead of ploting, by default False

    Returns
    -------
    Union[None, Tuple[plt.Figure, plt.Axes]]
        either plots the histograms or return the fig ax variables
    """
    for c in columns:
        fig, ax = plt.subplots(1,2, figsize=(8, 3), width_ratios=[2.5, 1])
        
        ax[0].hist(df[c], bins=100)
        ax[0].set(
            title = f"{name} property: {c}",
            xlabel=c,
            ylabel="count"
            )
        lower_quantile = df[c].quantile(0.1)
        upper_quantile = df[c].quantile(0.9)


        data_quantile = df[c][(df[c] >= lower_quantile) & (df[c] <= upper_quantile)]

                
        ax[1].hist(data_quantile, bins=100)
        ax[1].set(
            title = f"zoomed {name} property: {c}",
            xlabel=c,
            ylabel="count"
            )
        
        if return_fig:
            return fig, ax
        
        
        plt.tight_layout()
        plt.show()
        
        
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
    
    planet_counts = df.groupby("System number").size().reset_index(name="planets count", drop=True)
    
    return planet_counts


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
    
    total_mass = df.groupby("System number")["Total Mass (Mearth)"].sum().reset_index(name="Total planets mass (Mearth)", drop=True)
    
    return total_mass


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
    
    return df1.merge(df2, on="System number")
    


    
