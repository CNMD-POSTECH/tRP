import os
import yaml
import argparse
#from SISSO.src.code.descriptors_reg import Out
from SISSO.src.code.descriptors import Out
from SISSO.src.code.criterion import Out_bias

def sisso_output(config):
    try:
        extraction_config = config['extraction']
        out = Out(label=config['columns']['label'],
                  run_path=config['sisso_path'],
                  out_path=config['save_path'],
                  train_set=config['data']['train'],
                  test_set=config['data']['test'],
                  feature_list=extraction_config['feature_list'],
                  max_complexity=extraction_config['max_complexity'],
                  min_complexity=extraction_config['min_complexity'])
    except:
        criterion_config = config['criterion']
        chalcogen = criterion_config['chalcogen']
        if chalcogen:
            halogen = False
        else:
            halogen = criterion_config['halogen']
        out = Out_bias( label=config['columns']['label'],
                        levels=criterion_config['n_value'],
                        data_path=criterion_config['data_path'],
                        except_source=criterion_config['except_source'],
                        out_path=config['save_path'],
                        chalcogen=chalcogen,
                        halogen=halogen,
                        feature_list=criterion_config['feature_list'],
                        descriptor=criterion_config['descriptor'],
                        )

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config={}
    return config

def main():
    parser = argparse.ArgumentParser(description="Extraction the results of SISSO")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    sisso_output(config)
