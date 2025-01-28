from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import argparse
import shap
import yaml
import os

def shap_analysis(config):
    train=pd.read_csv(config['data']['train'])
    test=pd.read_csv(config['data']['test'])

    features=config['shap']['features']
    X_train = np.array(train[features])
    X_test = np.array(test[features])
    Y_train = np.array(train[config['shap']['target']]).ravel()

    best_rf = RandomForestClassifier(
                                    bootstrap=config['randomforest']['bootstrap'],
                                    criterion=config['randomforest']['criterion'],
                                    max_depth=config['randomforest']['max_depth'],
                                    max_features=config['randomforest']['max_features'],
                                    min_samples_leaf=config['randomforest']['min_samples_leaf'],
                                    min_samples_split=config['randomforest']['min_samples_split'],
                                    n_estimators=config['randomforest']['n_estimators'],
                                    random_state=config['random_state']
                                )

    best_rf.fit(X_train,Y_train)
    best_rf.predict(X_test)

    explainer = shap.TreeExplainer(best_rf)
    rf_shap = explainer.shap_values(X_train)
    #rf_shap_interaction = shap.TreeExplainer(best_rf).shap_interaction_values(X_train)

    shap_sum = abs(rf_shap[1]).sum(0)
    shap_fi = shap_sum/shap_sum.sum()
    shap_fi = [round(x,3)*100 for x in shap_fi]

    shap_importance_df = pd.DataFrame({'name':features, 'shap_importance':shap_fi})
    shap_importance_df.sort_values(by='shap_importance', ascending=False, inplace=True)

    save_path = os.path.join(config['save_path'], 'shap_result.csv')
    shap_importance_df.to_csv(save_path, index=False)

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config={}
    return config

def main():
    parser = argparse.ArgumentParser(description="Extract important features through SHAP analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)
    shap_analysis(config)