import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from e2ml.experimentation import perform_bayesian_optimization
from e2ml.preprocessing import PrincipalComponentAnalysis
from sklearn.decomposition import PCA
from e2ml.experimentation import acquisition_ei
from e2ml.models import GaussianProcessRegression
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

volume = full_length * full_width * full_height
#print(volume)
weight_volume_quotients = full_weight / volume 
non_shell_quotient = full_non_shell_weight / full_weight
intestine_quotient = full_intestine_weight / full_weight
shell_quotient = full_shell_weight / full_weight

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


def score_cross_entropy_loss(mdl, x, y):
    if(len(y.shape) == 1):
        y = getOneHotEncoding(y)
    y_pred = softmax(mdl.predict_proba(x))
    #return cross_entropy_loss(y, y_pred)
    return [log_loss(y[i], y_pred[i])*-1 for i in range(len(y_pred))]

def cross_entropy_loss(y_true, y_pred):
    if(len(y_true.shape) == 1):
        return np.array([np.log(y_pred[i]) for i in y_true])
    elif(len(y_true.shape) == 2):
        return np.array([-(y_true[i] * np.log(y_pred[i])).sum() for i in range(len(y_pred))])
    
def softmax(x):
    x = np.array(x)
    return np.exp(x) / np.exp(x).sum(axis=1).reshape(x.shape[0],-1)

print("mlp")
mlp = MLPClassifier(max_iter=1000)
mlp.fit(x[:10], y[:10])
y_pred = mlp.predict_proba(x[11:])
y_pred = softmax(y_pred)
mlp = MLPClassifier(max_iter=1000)
#cvs_mlp = cross_val_score(mlp, x, y, cv=3, scoring=score_cross_entropy_loss)
#print(cvs_mlp)

y_rfc = full_stages.replace("Adult",0).replace("Adole",1).replace("Child",2)

print("svc")
svc = SVC(kernel="rbf", probability=True)
svc.fit(x[:10], y_rfc[:10])
y_pred = softmax(svc.predict_proba(x[11:]))
svc = SVC(kernel="rbf", probability=True)
#cvs_svc = cross_val_score(svc, x, y_rfc, cv=3, scoring=score_cross_entropy_loss)
#print(cvs_svc)

print("rfc")

rfc = RandomForestClassifier()
rfc.fit(x[:10], y_rfc[:10])
y_pred = softmax(rfc.predict_proba(x[11:]))
rfc = RandomForestClassifier()
#cvs_rfc = cross_val_score(rfc, x, y_rfc, cv=3, scoring=score_cross_entropy_loss)
#print(f"{cvs_rfc=}")

def objectiveFunction(x, y):
    rfc = RandomForestClassifier()
    cvs_rfc = cross_val_score(rfc, x, y_rfc, cv=3, scoring=score_cross_entropy_loss)
    svc = SVC(kernel="rbf", probability=True)
    cvs_svc = cross_val_score(svc, x, y_rfc, cv=3, scoring=score_cross_entropy_loss)
    mlp = MLPClassifier(max_iter=1000)
    cvs_mlp = cross_val_score(mlp, x, y, cv=3, scoring=score_cross_entropy_loss)
    return (cvs_rfc.mean() + cvs_svc.mean() + cvs_mlp.mean()) / 3


def getDict(size:int):
    d = {}
    d["Sex"] = np.random.choice(full_sexes, size)
    d["Length"] = getNewSamples(full_length, size)
    d["Width"] = getNewSamples(full_width, size)
    d["Height"] =  getNewSamples(full_height, size)
    d["Weight"] = d["Height"] * d["Width"] * d["Length"] * np.random.normal(weight_volume_quotients.mean(), weight_volume_quotients.std())
    d["Non_Shell Weight"] = d["Weight"] * np.random.normal(non_shell_quotient.mean(), non_shell_quotient.std())
    d["Intestine Weight"] = d["Weight"] * np.random.normal(intestine_quotient.mean(), intestine_quotient.std())
    d["Shell Weight"] = d["Weight"] * np.random.normal(shell_quotient.mean(), shell_quotient.std())
    return d
d = getDict(282)
print(f"{d['Weight'].mean()=} {d['Weight'].std()=}")
print(f"{full_weight.mean()=} {full_weight.std()=}")
print(f"{d['Non_Shell Weight'].mean()=} {d['Non_Shell Weight'].std()=}")
print(f"{full_non_shell_weight.mean()=} {full_non_shell_weight.std()=}")
print(f"{d['Intestine Weight'].mean()=} {d['Intestine Weight'].std()=}")
print(f"{full_intestine_weight.mean()=} {full_intestine_weight.std()=}")
print(f"{d['Shell Weight'].mean()=} {d['Shell Weight'].std()=}")
print(f"{full_shell_weight.mean()=} {full_shell_weight.std()=}")
orignial_diffs = full_weight - full_intestine_weight - full_non_shell_weight - full_shell_weight
new_diffs = d["Weight"] - d["Intestine Weight"] - d["Non_Shell Weight"] - d["Shell Weight"]
print(orignial_diffs)
print(new_diffs)
print(f"{orignial_diffs.mean()=} {orignial_diffs.std()=}")
print(f"{new_diffs.mean()=} {new_diffs.std()=}")
print(pd.DataFrame(d))

full_replaced = initial_molluscs_data.replace("F",0).replace("I",1).replace("M",2)
x_full_replaced = full_replaced.values[:,:-1]
pca = PCA(2)
pca = pca.fit(x_full_replaced)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

x_pca = pca.transform(x_full_replaced)
print(x_pca)

#new valuerange is -1.5, 1.5

print("mlp")
mlp = MLPClassifier(max_iter=1000)
mlp.fit(x_pca[:10], y[:10])
y_pred = mlp.predict_proba(x_pca[11:])
y_pred = softmax(y_pred)
mlp = MLPClassifier(max_iter=1000)
#cvs_mlp = cross_val_score(mlp, x_pca, y, cv=3, scoring=score_cross_entropy_loss)
#print(cvs_mlp)

y_rfc = full_stages.replace("Adult",0).replace("Adole",1).replace("Child",2)

print("svc")
svc = SVC(kernel="rbf", probability=True)
svc.fit(x_pca[:10], y_rfc[:10])
y_pred = softmax(svc.predict_proba(x_pca[11:]))
svc = SVC(kernel="rbf", probability=True)
#cvs_svc = cross_val_score(svc, x_pca, y_rfc, cv=3, scoring=score_cross_entropy_loss)
#print(cvs_svc)

print("rfc")

rfc = RandomForestClassifier()
rfc.fit(x_pca[:10], y_rfc[:10])
y_pred = softmax(rfc.predict_proba(x_pca[11:]))
rfc = RandomForestClassifier()
#cvs_rfc = cross_val_score(rfc, x_pca, y_rfc, cv=3, scoring=score_cross_entropy_loss)
#print(f"{cvs_rfc=}")

def objectiveFunctionRFC(x, y_rfc):
    rfc = RandomForestClassifier()
    rfc.fit(x,y_rfc)
    return score_cross_entropy_loss(rfc, x, y_rfc)

def objectiveFunctionSVC(x,y_rfc):
    svc = SVC(kernel="rbf", probability=True)
    svc.fit(x, y_rfc)
    return score_cross_entropy_loss(svc, x, y_rfc)

def objectiveFunctionMLP(x, y):
    mlp = MLPClassifier(max_iter=1000)
    mlp.fit(x,y)
    return score_cross_entropy_loss(mlp, x, y)


x_acquired = x_pca
y_acquired_rfc = objectiveFunctionRFC(x_acquired, y_rfc)
print(y_acquired_rfc)
y_acquired_svc = objectiveFunctionSVC(x_pca, y_rfc)
print(y_acquired_svc)
y_acquired_mlp = objectiveFunctionMLP(x_pca, y)
print(y_acquired_mlp)

metrics_dict = {'gamma': 50, 'metric': 'rbf'}
gpr = GaussianProcessRegression(metrics_dict=metrics_dict)
gpr.fit(x_acquired, y_acquired_rfc)

x1_new = np.linspace(-1.5, 1.5, 1000)
x_mesh,y_mesh = np.meshgrid(x1_new, x1_new)

x_cand = np.stack((x_mesh, y_mesh), axis=2).reshape(-1,2)
print(x_cand.shape)
means, stds = gpr.predict(x_cand, True)
scores = acquisition_ei(means, stds, max(y_acquired_rfc))
print(scores)