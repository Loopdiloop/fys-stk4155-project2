import numpy as np 
from imageio import imread
import sys
import os
from data_generation import data_generate
from functions import reduce4

"""Generates a data set for the terrain that can be loaded in other functions"""

# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')

#reduce the terrain file by 16
terrain1 = reduce4(reduce4(terrain1))

#If lasso, then we should have an even smaller map.
terrain1 = reduce4(reduce4(terrain1))

#Load the dataset
dataset = data_generate()
dataset.load_terrain_data(terrain1)
dataset.save_data()