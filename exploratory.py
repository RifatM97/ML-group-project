import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def variable_info(data):
    """ Print the data types of each variable to the console"""
    print(list(data.columns))
    for column in data.columns.values:
        print (column, "data type: ", type(data[column][1]))

def plot_heatmap(data):
    """Plot and save a heatmap of the correlation of each variable used in the data"""
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(), annot=True)
    plt.savefig('plots\Correlation_heatmap.png')

def plot_overlay_hist(data):
    """Plot an overlay histogram of the survival of passengers for the Age and Fare 
        variable separately"""
    survived = data[data.Survived==1]
    notsurvived = data[data.Survived==0]

    bins = np.linspace(0, 80, 16)
    plt.figure(figsize=(10,8))
    plt.hist(survived.Age, bins=bins, alpha=0.5, label='Survived = 1')
    plt.hist(notsurvived.Age, bins=bins, alpha=0.5, label='Survived = 0')
    plt.legend(loc='upper right')
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.savefig('plots\Histogram_age_by_survival.png')

    bins = np.linspace(0, 250, 25)
    plt.figure(figsize=(10,8))
    plt.hist(survived.Fare, bins=bins, alpha=0.5, label='Survived = 1')
    plt.hist(notsurvived.Fare, bins=bins, alpha=0.5, label='Survived = 0')
    plt.legend(loc='upper right')
    plt.xlabel("Fare")
    plt.ylabel("Count")
    plt.savefig('plots\Histogram_Fare_by_survival.png')