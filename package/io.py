import pandas as pd
import pickle
from datetime import datetime as dt
import pytz as tz


# Fetch data with filename
def importdata(filename):
    data = pd.read_csv(filename, header=None, sep=',')
    data.columns = data.iloc[0]
    data = data.drop([0])
    return data


# Sanitize data. Temporarily dropping columns E_regression, Material Composition
def sanitizedata(data, user_list=None):
    col_list = ['Material compositions 1', 'Material compositions 2', 'E_regression', 'predict_Pt',
                'Hop activation barrier']
    if user_list is not None:
        col_list = col_list + user_list
    # Remove columns if present
    for c in col_list:
        if c in data.columns:
            data = data.drop(columns=[c])
    return data


def getModelSaveFileName(filename_from_user):
    return filename_from_user.replace("/", "_").replace(".", "_") + "_" \
           + dt.now(tz=tz.timezone('America/Chicago')).strftime("%m-%d-%y_%H-%M-%S")


def savemodelobj(obj, filename_from_user=''):
    with open(getModelSaveFileName(filename_from_user), 'wb') as out:
        pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)


def loadmodelobj(filename_from_user):
    with open(filename_from_user, 'rb') as inp:
        obj = pickle.load(inp)
    return obj
