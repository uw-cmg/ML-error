import numpy as np
from package import CVData as cvd
from package import CorrectionFactors as cf
from package import MakePlot as mp
from package import TestData as td
from sklearn.model_selection import train_test_split
import sys
import os

# Check for command line input
if len(sys.argv) < 4:
    print("Need at least 3 command line arguments. (run/plot, model, and dataset)")
    quit()

action = sys.argv[1]
model = sys.argv[2]
dataset = sys.argv[3]
if len(sys.argv) >= 5:
    path = sys.argv[4]
    if path[-1] != "/":
        path = path + "/"
else:
    path = ""

# make file to save plots
directory = "{}_{}_{}".format(model, dataset, action)
path = os.path.join(path, directory)
os.mkdir(path)

