import json
import yaml
import argparse
import numpy as np
import pandas as pd

def get_radius(shannon, site, charge, coord_number):
    """ Get the Shannon radius for a given site, charge, and coordination number. """
    try:
        cn_list_1 = list(shannon[site][str(int(charge))].keys())
        cn_list_2 = [abs(int(coord) - coord_number) for coord in cn_list_1]
        min_index = [idx for idx in range(len(cn_list_2)) if cn_list_2[idx] == np.min(cn_list_2)][0]
        coord = cn_list_1[min_index]
        return shannon[site][str(int(charge))][str(coord)]['only_spin']
    except KeyError:
        print(f"No radius available in Shannon dictionary for {site} with charge {charge} and CN {coord_number}")
        return None

def get_bias(config):
    level = config['level']
    chalcogen = config['chalcogen']
    if config['bias']:
        bias = config['bias']
    else:
        if chalcogen:
            bias_dict = {1: 1.83, 2: 1.82, 3: 1.79}
            bias = bias_dict[level]
        else:
            bias = 1.72
    return bias

def set_prediction(config):
    param = config['set_parameter']
    bias = get_bias(config)
    df = pd.read_csv(param['data_path'])

    df['p_Label'] = ''
    df = df[df['n']==config['level']].reset_index(drop=True)
    df['rp'] = (df['rB']/df['rA'])**2+np.sqrt(df['rX']/df['rB'])
    #df['rp'] = (df['rA']/df['rX'])+(np.log(df['rB'])/np.exp(df['nA']))

    chalcogen = ['O','S','Se','Te']
    halogen = ['F','Cl','Br','I']
    if config['chalcogen']:
        df = df[(df['X1'].isin(chalcogen)) | (df['X2'].isin(chalcogen))].reset_index(drop=True)
    else:
        df = df[df['X1'].isin(halogen)].reset_index(drop=True)
        df = df[df['X2'].isin(halogen)].reset_index(drop=True)
    
    if param['hybrid']:
        pass
    else:
        df = df[df['Source']!='Hybrid'].reset_index(drop=True)

    pred_rp_o = df[(df['rp']<=bias) & (df['Label']==1)]
    pred_rp_x = df[(df['rp']>bias) & (df['Label']==1)]
    pred_nrp_x = df[(df['rp']<=bias) & (df['Label']!=1)]
    pred_nrp_o = df[(df['rp']>bias) & (df['Label']!=1)]

    #pred_rp_o = df[(df['rp']>=bias) & (df['Label']==1)]
    #pred_rp_x = df[(df['rp']<bias) & (df['Label']==1)]
    #pred_nrp_x = df[(df['rp']>=bias) & (df['Label']!=1)]
    #pred_nrp_o = df[(df['rp']<bias) & (df['Label']!=1)]

    TP = len(pred_rp_o)
    TN = len(pred_nrp_o)
    FP = len(pred_nrp_x)
    FN = len(pred_rp_x)
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    accuracy_rp = TP/(TP+FN)
    accuracy_non_rp = TN/(TN+FP)
    
    precision = TP/(TP+FP) if TP+FP != 0 else 0
    recall = TP/(TP+FN) if TP+FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    
    accuracy = [round(accuracy,2)]
    accuracy_rp = [round(accuracy_rp,2)]
    accuracy_non_rp = [round(accuracy_non_rp,2)]
    f1_score = [round(f1_score,2)]

    result = pd.DataFrame({'accuracy':accuracy, 'accuracy_rp':accuracy_rp, 'accuracy_nrp':accuracy_non_rp, 'f1':f1_score})
    result.to_csv(param['save_path'])
    return result

def prediction(config):
    param = config['single_parameter']

    compound = param['compound']
    a1_site = param['A1']
    a2_site = param['A2']
    b1_site = param['B1']
    b2_site = param['B2']
    x1_site = param['X1']
    x2_site = param['X2']
    
    bias = get_bias(config)
    
    a1_ratio, a2_ratio = param['A1_ratio'], param['A2_ratio']
    b1_ratio, b2_ratio = param['B1_ratio'], param['B2_ratio']
    x1_ratio, x2_ratio = param['X1_ratio'], param['X2_ratio']
    
    cn_a1, cn_a2 = param['A1_CN'], param['A2_CN']
    cn_b1, cn_b2 = param['B1_CN'], param['B2_CN']
    cn_x1, cn_x2 = param['X1_CN'], param['X2_CN']

    q_a1, q_a2 = param['A1_Q'], param['A2_Q']
    q_b1, q_b2 = param['B1_Q'], param['B2_Q']
    q_x1, q_x2 = param['X1_Q'], param['X2_Q']

    sum_charge = q_a1 * a1_ratio + q_a2 * a2_ratio + q_b1 * b1_ratio + q_b2 * b2_ratio + q_x1 * x1_ratio + q_x2 * x2_ratio
    if sum_charge != 0:
        raise ValueError("Total charge is not neutral!")

    if param['shannon_radius']:
        with open(param['shannon_radius'], 'r') as f:
            shannon = json.load(f)

        a1_radius = get_radius(shannon, a1_site, q_a1, cn_a1)
        a2_radius = get_radius(shannon, a2_site, q_a2, cn_a2)
        b1_radius = get_radius(shannon, b1_site, q_b1, cn_b1)
        b2_radius = get_radius(shannon, b2_site, q_b2, cn_b2)
        x1_radius = get_radius(shannon, x1_site, q_x1, cn_x1)
        x2_radius = get_radius(shannon, x2_site, q_x2, cn_x2)

    else:
        a1_radius = param['A1_radius']
        a2_radius = param['A2_radius']
        b1_radius = param['B1_radius']
        b2_radius = param['B2_radius']
        x1_radius = param['X1_radius']
        x2_radius = param['X2_radius']

    a_radi = (a1_radius * a1_ratio + a2_radius * a2_ratio) / (a1_ratio + a2_ratio)
    b_radi = (b1_radius * b1_ratio + b2_radius * b2_ratio) / (b1_ratio + b2_ratio)
    x_radi = (x1_radius * x1_ratio + x2_radius * x2_ratio) / (x1_ratio + x2_ratio)

    rp = (b_radi / a_radi) ** 2 + np.sqrt(x_radi / b_radi)

    if rp <= bias:
        print(f"Stable RP Phase: {rp}, {compound}")
    else:
        print(f"Unstable RP Phase: {rp}, {compound}")

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config={}
    return config

def main():
    parser = argparse.ArgumentParser(description="Prediction Stable RP Phase")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    if config['set']:
        set_prediction(config)
    else:
        prediction(config)
