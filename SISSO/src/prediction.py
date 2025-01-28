import json
import yaml
import argparse
import numpy as np

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

def prediction(config):
    compound = config['compound']
    level = config['level']

    a1_site = config['A1']
    a2_site = config['A2']
    b1_site = config['B1']
    b2_site = config['B2']
    x1_site = config['X1']
    x2_site = config['X2']
    
    chalcogen_list = ['O', 'S', 'Se', 'Te']
    halogen_list = ['F', 'Cl', 'Br', 'I']

    chalcogen = False
    halogen = False
    if x1_site in chalcogen_list or x2_site in chalcogen_list:
        chalcogen = True
    else:
        halogen = True

    if config['bias']:
        bias = config['bias']
    else:
        if chalcogen:
            bias_dict = {1: 1.83, 2: 1.82, 3: 1.79}
            bias = bias_dict[level]
        else:
            bias = 1.72
    
    a1_ratio, a2_ratio = config['A1_ratio'], config['A2_ratio']
    b1_ratio, b2_ratio = config['B1_ratio'], config['B2_ratio']
    x1_ratio, x2_ratio = config['X1_ratio'], config['X2_ratio']
    
    cn_a1, cn_a2 = config['A1_CN'], config['A2_CN']
    cn_b1, cn_b2 = config['B1_CN'], config['B2_CN']
    cn_x1, cn_x2 = config['X1_CN'], config['X2_CN']

    q_a1, q_a2 = config['A1_Q'], config['A2_Q']
    q_b1, q_b2 = config['B1_Q'], config['B2_Q']
    q_x1, q_x2 = config['X1_Q'], config['X2_Q']

    sum_charge = q_a1 * a1_ratio + q_a2 * a2_ratio + q_b1 * b1_ratio + q_b2 * b2_ratio + q_x1 * x1_ratio + q_x2 * x2_ratio
    if sum_charge != 0:
        raise ValueError("Total charge is not neutral!")

    if config['shannon_radius']:
        with open(config['shannon_radius'], 'r') as f:
            shannon = json.load(f)

        a1_radius = get_radius(shannon, a1_site, q_a1, cn_a1)
        a2_radius = get_radius(shannon, a2_site, q_a2, cn_a2)
        b1_radius = get_radius(shannon, b1_site, q_b1, cn_b1)
        b2_radius = get_radius(shannon, b2_site, q_b2, cn_b2)
        x1_radius = get_radius(shannon, x1_site, q_x1, cn_x1)
        x2_radius = get_radius(shannon, x2_site, q_x2, cn_x2)

    else:
        a1_radius = config['A1_radius']
        a2_radius = config['A2_radius']
        b1_radius = config['B1_radius']
        b2_radius = config['B2_radius']
        x1_radius = config['X1_radius']
        x2_radius = config['X2_radius']

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
    prediction(config)
