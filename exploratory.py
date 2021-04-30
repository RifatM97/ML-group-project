import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def variable_info(data):
    print(list(data.columns))
    for column in data.columns.values:
        print (column, "data type: ", type(data[column][1]))

def plot_heatmap(data):
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(), annot=True)

def plot_overlay_hist(data):
    survived = data[data.Survived==1]
    notsurvived = data[data.Survived==0]

    bins = np.linspace(0, 80, 16)
    plt.figure(figsize=(10,8))
    plt.hist(survived.Age, bins=bins, alpha=0.5, label='Survived = 1')
    plt.hist(notsurvived.Age, bins=bins, alpha=0.5, label='Survived = 0')
    plt.legend(loc='upper right')
    plt.xlabel("Age")
    plt.ylabel("Count")

    bins = np.linspace(0, 250, 25)
    plt.figure(figsize=(10,8))
    plt.hist(survived.Fare, bins=bins, alpha=0.5, label='Survived = 1')
    plt.hist(notsurvived.Fare, bins=bins, alpha=0.5, label='Survived = 0')
    plt.legend(loc='upper right')
    plt.xlabel("Fare")
    plt.ylabel("Count")
    plt.show()