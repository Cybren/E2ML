import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
#from e2ml import utils

initial_molluscs_data = pd.read_csv('./data/initial_molluscs_data.csv')
initial_molluscs_data

adults = initial_molluscs_data.loc[initial_molluscs_data["Stage of Life"] == "Adult"]
#print(adults)
adoles = initial_molluscs_data.loc[initial_molluscs_data["Stage of Life"] == "Adole"]
#print(adoles)
children = initial_molluscs_data.loc[initial_molluscs_data["Stage of Life"] == "Child"]
#print(children)
#will 282 haben (5 Tage. 21 Stunden)
full_length = initial_molluscs_data["Length"]
full_width = initial_molluscs_data["Width"]
full_height = initial_molluscs_data["Height"]
full_weight = initial_molluscs_data["Weight"]
full_non_shell_weight = initial_molluscs_data["Non_Shell Weight"]
full_intestine_weight = initial_molluscs_data["Intestine Weight"]
full_shell_weight = initial_molluscs_data["Shell Weight"]
full_sexes = initial_molluscs_data["Sex"]
full_stages = initial_molluscs_data["Stage of Life"]


def getOneHotEncoding(data):
    values = np.sort(np.unique(data))
    enc = np.zeros((len(data), len(values)))
    for i, x in enumerate(data):
        enc[i, np.where(values == x)[0][0]] = 1
    return enc

x = np.concatenate((getOneHotEncoding(full_sexes), initial_molluscs_data.values[:,1:-1]), axis=1)
y = getOneHotEncoding(full_stages)

def getNewSamples(old_data:pd.Series, size:int):
    return np.random.normal(old_data.mean(), scale=old_data.std(), size=size)

def getDict(size:int):
    d = {}
    d["Sex"] = np.random.choice(full_sexes, size)
    d["Length"] = getNewSamples(full_length, size)
    d["Width"] = getNewSamples(full_width, size)
    d["Height"] =  getNewSamples(full_height, size)
    d["Weight"] = getNewSamples(full_weight, size)

def score_cross_entropy_loss(mdl, x, y):
    print(f"{y=}")
    y_pred = softmax(mdl.predict_proba(x))
    return -log_loss(y, y_pred)

def cross_entropy_loss(y_true, y_pred):
    if(len(y_true.shape) == 1):
        return np.array([np.log(y_pred[i]) for i in y_true])
    elif(len(y_true.shape) == 2):
        return np.array([-(y_true[i] * np.log(y_pred[i])).sum() for i in range(len(y_pred))])
    
def softmax(x):
    x = np.array(x)
    print(f"{x=}")
    return np.exp(x) / np.exp(x).sum(axis=1).reshape(x.shape[0],-1)

print("mlp")
mlp = MLPClassifier(max_iter=1000)
mlp.fit(x[:10], y[:10])
y_pred = mlp.predict_proba(x[11:])
y_pred = softmax(y_pred)
mlp = MLPClassifier(max_iter=1000)
cvs_mlp = cross_val_score(mlp, x, y, cv=3, scoring=score_cross_entropy_loss)
#print(cvs_mlp)

y_rfc = full_stages.replace("Adult",0).replace("Adole",1).replace("Child",2)

print("svc")
svc = SVC(kernel="rbf", probability=True)
svc.fit(x[:10], full_stages[:10])
y_pred = softmax(svc.predict_proba(x[11:]))
svc = SVC(kernel="rbf", probability=True)
cvs_svc = cross_val_score(svc, x, y_rfc, cv=3, scoring=score_cross_entropy_loss)
print(cvs_svc)

print("rfc")

rfc = RandomForestClassifier()
rfc.fit(x[:10], y_rfc[:10])
y_pred = softmax(rfc.predict_proba(x[11:]))
rfc = RandomForestClassifier()
cvs_rfc = cross_val_score(rfc, x, y_rfc, cv=3, scoring=score_cross_entropy_loss)
print(f"{cvs_rfc=}")



    
