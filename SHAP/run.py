from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import os

train=pd.read_csv('Table.S1_train.csv')
test=pd.read_csv('Table.S1_test.csv')

features = ['nA','nB','nX',
            'rA','rB','rX',
            'vA','vB','vX',
            'xA','xB','xX',
            'iA','iB','iX']

sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

X_train = np.array(train[features])
X_test = np.array(test[features])
Y_train = np.array(train[['Label']]).ravel()
Y_test = np.array(test[['Label']]).ravel()

param_grid = {
    'random_state' : [42],
    'max_features' : [1.0, 'sqrt'],
    'n_estimators':[50,100],
    'max_depth': [100],
    'criterion': ['gini','entropy', 'log_loss'],
    'bootstrap': [True],
    'min_samples_leaf': [1,10,20],
    'min_samples_split': [5,10,20],
}
rfc=RandomForestClassifier()

best_rf = RandomForestClassifier(bootstrap=True, 
                                 criterion='gini', 
                                 max_depth=100, 
                                 max_features=1.0, 
                                 min_samples_leaf=1, 
                                 min_samples_split=5, 
                                 n_estimators=100, 
                                 random_state=42)

best_rf.fit(X_train,Y_train)
best_rf.predict(X_test)

explainer = shap.TreeExplainer(best_rf)
rf_shap = explainer.shap_values(X_train)
rf_shap_interaction = shap.TreeExplainer(best_rf).shap_interaction_values(X_train)

shap_sum = abs(rf_shap[1]).sum(0)
shap_fi = shap_sum/shap_sum.sum()
shap_fi = [round(x,3)*100 for x in shap_fi]

shap_importance_df = pd.DataFrame({'name':features, 'shap_importance':shap_fi})
shap_importance_df.sort_values(by='shap_importance', ascending=False, inplace=True)
shap_importance_df.to_csv('result.csv', index=False)