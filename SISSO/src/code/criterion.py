from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import shutil
import os

from SISSO.src.code.etc import math_functions, bias_functions, get_bias_functions

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def process_descriptor(desc, data, label, path, level, seed, large, 
                       chalcogen=True, halogen=False, between_large=False, between_small=False, between_value=0):
    rf = RandomForestClassifier(random_state=seed,
                                bootstrap=True,
                                criterion='entropy',
                                max_depth=1)
    results = []
    # test accuracy 계산
    if between_large:
        data = data[data[desc]>between_value].reset_index(drop=True)
    if between_small:
        data = data[data[desc]<between_value].reset_index(drop=True)
    x, y = np.array(data[[desc]]), np.array(data[[label]]).ravel()
    rf.fit(x,y)
    biases = bias_functions(rf.estimators_, path, 0)
    biases = [round(bias, 2) for bias in biases]
    data_results = get_bias_functions(data, desc, label, biases, large)
    results.append(data_results)

    df_results = pd.DataFrame(results, 
                              index=['total'])
    
    if chalcogen:
        df_results.to_csv(os.path.join(path, f'criterion_chalcogen_{level}.csv'), index=True)
    if halogen:
        df_results.to_csv(os.path.join(path, f'criterion_halogen_{level}.csv'), index=True)
    print(df_results)
    return df_results

class Out_bias:
    def __init__(self,
                 seed=42,
                 large=None,
                 splits=5,
                 levels=0,
                 halogen=False,
                 chalcogen=False,
                 label='Label',
                 out_path='./',
                 data_path='./',
                 except_source=False,
                 descriptor='',
                 feature_list=['rA','rB','rX','xX'],
                 operator_list=['+','-','*','/',
                                '^-1','^2',
                                'sqrt','log',
                                'ln','exp'],
                 between_large=False,
                 between_small=False,
                 between_value=0):
        
        self.seed = seed
        self.label = label
        self.large = large
        self.splits = splits
        self.levels = levels
        self.halogen = halogen
        self.chalcogen = chalcogen
        self.out_path = out_path
        self.data_df = pd.read_csv(data_path)
        self.descriptor = descriptor
        self.feature_list = feature_list
        self.operator_list = operator_list
        
        if except_source:
            self.data_df = self.data_df[~self.data_df['Source'].str.contains(except_source)].reset_index(drop=True)
                            
        if self.levels !=0:
            self.data_df = self.data_df[self.data_df['n']==levels].reset_index(drop=True)
        # anion 기준으로 데이터셋 생성
        self.data_df = self.get_anions(self.data_df)
        
        print('data', len(self.data_df))
             
        self.df = self.data_df.copy()
                        
        # descriptor들의 값 계산
        self.get_calculation()
        
        # 특정 range의 데이터셋 생성
        self.between_large = between_large
        self.between_small = between_small
        self.between_value = between_value
            
        # descriptor들의 정확도 계산
        self.get_accuracy()
                
    def get_anions(self, df):
        chalcogen = ['O','S','Se','Te']
        halogen = ['F','Cl','Br','I']
        if self.halogen:
            df = df[df['X1'].isin(halogen)].reset_index(drop=True)
            df = df[df['X2'].isin(halogen)].reset_index(drop=True)
        elif self.chalcogen:
            df = df[(df['X1'].isin(chalcogen)) | (df['X2'].isin(chalcogen))].reset_index(drop=True)
        return df

    def get_calculation(self):
        calculate_desc = {}
        try:
            for feature in self.feature_list:
                self.descriptor = self.descriptor.replace(feature, f'self.df["{feature}"]')
            self.descriptor = self.descriptor.split(' ')[0]
            self.descriptor = math_functions(self.descriptor)
            #descriptor = bracket_functions(descriptor)
            calculate_desc['cand_'+str(self.descriptor)] = eval(self.descriptor)
        except Exception as e:
            print('error_descriptor: ',e, self.descriptor)
                
        self.df = pd.concat([self.df, pd.DataFrame(calculate_desc)], axis=1)
        
    def get_accuracy(self):        
        def process_parallel(descriptors, data, label, path, seed, large, level):
            with Pool() as pool:
                for _, desc in enumerate(descriptors):
                    pool.apply_async(process_descriptor, 
                                    (
                                    desc, 
                                    data, 
                                    label, 
                                    path, 
                                    level, 
                                    seed,
                                    large,
                                    self.chalcogen,
                                    self.halogen,
                                    self.between_large,
                                    self.between_small,
                                    self.between_value,
                                    ))
                pool.close()
                pool.join()
        descriptors = [cols for cols in self.df.columns if 'cand_' in cols]
        process_parallel(descriptors, self.df, self.label, self.out_path, self.seed, self.large, self.levels)