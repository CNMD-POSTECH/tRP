from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import os

from SISSO.src.code.etc import math_functions

# In this output, accuracy means rmse.

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def process_descriptor(folder_list, desc, train, test, label, seed):
    rf = RandomForestRegressor(random_state=seed,
                               bootstrap=True,
                               max_depth=1)
    
    fold_r2 = []
    fold_mse = []
    fold_mse_o = []
    fold_mse_x = []

    # fold 별 accuracy 계산
    for num, folder in enumerate(folder_list):
        num = num+1
        fold_train = train[train[f'fold_{num}']!='fold_test'].reset_index(drop=True)
        fold_test = train[train[f'fold_{num}']=='fold_test'].reset_index(drop=True)
        
        fold_train_x, fold_train_y = np.array(fold_train[[desc]]), np.array(fold_train[[label]]).ravel()
        fold_test_x, fold_test_y = np.array(fold_test[[desc]]), np.array(fold_test[[label]]).ravel()
        fold_test_xo, fold_test_yo = np.array(fold_test[fold_test['Label']==1][[desc]]), np.array(fold_test[fold_test['Label']==1][[label]]).ravel()
        fold_test_xx, fold_test_yx = np.array(fold_test[fold_test['Label']!=1][[desc]]), np.array(fold_test[fold_test['Label']!=1][[label]]).ravel() 

        rf.fit(fold_train_x, fold_train_y)
        y_pred = rf.predict(fold_test_x)
        y_pred_o = rf.predict(fold_test_xo)
        y_pred_x = rf.predict(fold_test_xx)
        fold_r2.append(r2_score(fold_test_y, y_pred))
        fold_mse.append(np.sqrt(mean_squared_error(fold_test_y, y_pred)))
        fold_mse_o.append(np.sqrt(mean_squared_error(fold_test_yo, y_pred_o)))
        fold_mse_x.append(np.sqrt(mean_squared_error(fold_test_yx, y_pred_x)))
            
    x, y = np.array(train[[desc]]), np.array(train[[label]]).ravel()
    x_test, y_test = np.array(test[[desc]]), np.array(test[[label]]).ravel()
 
    rf.fit(x,y)
    y_pred_test = rf.predict(x_test)
    r2_test = r2_score(y_test, y_pred_test)
    accuracy_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    y_pred_train = rf.predict(x)
    f1_train = r2_score(y, y_pred_train)
    accuracy_train = np.sqrt(mean_squared_error(y, y_pred_train))
    
    return accuracy_test, r2_test, accuracy_train, f1_train, np.mean(np.array(fold_mse)), np.mean(np.array(fold_r2)), np.mean(np.array(fold_mse_o)), np.mean(np.array(fold_mse_x))

class Out:
    def __init__(self,
                 seed=42,
                 splits=5,
                 label='f.e',
                 run_path='./',
                 out_path='./',
                 train_set='./train.csv',
                 test_set='./test.csv',
                 feature_list=['rA','rB','rX','xX'],
                 operator_list=['+','-','*','/',
                                '^-1','^2',
                                'sqrt','log',
                                'ln','exp'],
                 min_complexity=0,
                 max_complexity=10):
        
        self.seed = seed
        self.label = label
        self.splits = splits
        self.out_path = out_path
        self.descriptors = []
        self.feature_list = feature_list
        self.operator_list = operator_list
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
                    
        # descriptor 후보군 모두 추출
        self.folder_list = ['fold_{}'.format(num) for num in range(1, self.splits+1)]
        for folder in self.folder_list:
            self.run_path = os.path.join(run_path, folder)
            self.all_descriptor(self.run_path)
            print('all descriptor', len(self.descriptors))
         
        # descriptor들을 common descriptor로 변경   
        self.descriptors = self.common_descriptor()
        
        # accuracy 계산을 위한 데이터셋 생성
        self.train_df = pd.read_csv(train_set)
        self.test_df = pd.read_csv(test_set)
        for num, folder in enumerate(self.folder_list):
            num = num+1
            fold_df = pd.read_csv(os.path.join(run_path, folder, f'split_{num}.csv'))
            fold_test = fold_df[fold_df[f'fold_{num}']=='test'].reset_index(drop=True)
            fold_test_compound = fold_test['Substance'].tolist()
            self.train_df.loc[self.train_df['Substance'].isin(fold_test_compound), f'fold_{num}'] = 'fold_test'
        
        self.train_df['rf_label'] = 'train'
        self.test_df['rf_label'] = 'test'
        self.df = pd.concat([self.train_df, self.test_df], axis=0).reset_index(drop=True)
            
        # descriptor들의 complexity 평가
        self.descriptor_df = self.get_complexity()
        self.descriptor_df = self.descriptor_df.sort_values(by='Num_total', ascending=True)
        self.descriptor_df = self.descriptor_df[(self.descriptor_df['Num_total'] >= self.min_complexity) \
                                                & (self.descriptor_df['Num_total'] <= self.max_complexity)].reset_index(drop=True)
            
        # descriptor들의 값 계산
        self.get_calculation()
            
        # descriptor들의 정확도 계산
        self.get_accuracy()
        self.descriptor_df = self.descriptor_df.sort_values(by='Accuracy_avg_fold_test', ascending=True).reset_index(drop=True)
        self.descriptor_df = self.descriptor_df.sort_values(by='Accuracy_test', ascending=True).reset_index(drop=True)
        
        # save
        self.save()
            
    def all_descriptor(self, fold_path):
        features_space = '{}/feature_space/space_001d.name'.format(fold_path)
        with open(features_space, 'r') as file:
            self.descriptors += [line.split('corr=')[0].strip() for line in file]
        
    def common_descriptor(self):
        descriptor_count = pd.Series(self.descriptors).value_counts()
        common_descriptors = descriptor_count[descriptor_count == self.splits].index.tolist()
        print('common descriptor', len(common_descriptors))
        return common_descriptors
    
    def get_complexity(self):
        complexities = []
        for descriptor in self.descriptors:
            features_counts = sum(descriptor.count(feature) for feature in self.feature_list)
            operators_counts = sum(descriptor.count(operator) for operator in self.operator_list)
            exception_counts = sum(descriptor.count(exception) for exception in ['exp(-'])
            total_count = features_counts + operators_counts  - exception_counts
            complexities.append((descriptor, features_counts, operators_counts, total_count))
        return pd.DataFrame(complexities, columns=['Descriptor', 'Num_features', 'Num_operators', 'Num_total'])
            
    def get_calculation(self):
        calculate_desc = {}
        print(self.descriptor_df['Descriptor'])
        for i, descriptor in tqdm(enumerate(self.descriptor_df['Descriptor'])):
            try:
                for feature in self.feature_list:
                    descriptor = descriptor.replace(feature, f'self.df["{feature}"]')
                    descriptor = descriptor.replace('"self.df["n"]A"]', '"nA"]')
                    descriptor = descriptor.replace('"self.df["n"]B"]', '"nB"]')
                    descriptor = descriptor.replace('"self.df["n"]X"]', '"nX"]')
                    descriptor = descriptor.replace('self.df["n"]', 'self.df["n"].astype(float)')
                descriptor = descriptor.split(' ')[0]
                descriptor = math_functions(descriptor)
                descriptor_calculated = eval(descriptor)
                calculate_desc['cand_'+str(descriptor)] = descriptor_calculated
                if np.any(np.isinf(descriptor_calculated)) or np.any(descriptor_calculated > np.finfo(np.float32).max):
                    inf_indices = np.where(np.isinf(descriptor_calculated))[0]
                    large_value_indices = np.where(descriptor_calculated > np.finfo(np.float32).max)[0]
                    print(f"경고: Descriptor '{descriptor}' 계산 결과에 np.inf 또는 float32 범위를 초과하는 값이 있습니다.")
                    if len(inf_indices) > 0:
                        print(f"무한대 값이 있는 행의 인덱스: {inf_indices}")
                        print(f"무한대 값: {self.df['Substance'][inf_indices]}")
                    if len(large_value_indices) > 0:
                        print(f"float32 범위를 초과하는 값이 있는 행의 인덱스: {large_value_indices}")
                        print(f"float32 범위를 초과하는 값: {self.df['Substance'][large_value_indices]}")
                    problematic_indices = np.union1d(inf_indices, large_value_indices)
                    self.df = self.df.drop(problematic_indices).reset_index(drop=True)
                else:
                    calculate_desc['cand_'+str(descriptor)] = descriptor_calculated
            except Exception as e:
                print('error_descriptor: ',e, descriptor)
                
        self.df = pd.concat([self.df, pd.DataFrame(calculate_desc)], axis=1)
    
    def get_accuracy(self):        
        def process_parallel(descriptors, train, test, label, seed):
            with Pool() as pool:
                results = [pool.apply_async(process_descriptor, (self.folder_list, desc, train, test, label, seed)) for desc in descriptors]
                results = [r.get() for r in results]
            return results
        
        def update_descriptor_df(results):
            for idx, (accuracy_test, f1_test, accuracy_train, f1_train, fold_accuracy, fold_f1, fold_accuracy_o, fold_accuracy_x,) in enumerate(results):
                self.descriptor_df['Accuracy_avg_fold_test'][idx] = fold_accuracy
                self.descriptor_df['Accuracy_test'][idx] = accuracy_test

        train = self.df[
                self.df['rf_label']=='train'
                ].reset_index(drop=True)
        test = self.df[
                self.df['rf_label']=='test'
                ].reset_index(drop=True)
        
        descriptors = [cols for cols in self.df.columns if 'cand_' in cols]
        results = process_parallel(descriptors, train, test, self.label, self.seed)
        self.descriptor_df[['index', 
                            'Accuracy_avg_fold_test',
                            'Accuracy_test',
                            ]] = ['', 0.0, 0.0]
        update_descriptor_df(results)
         
    def save(self):
        self.df.to_csv(os.path.join(self.out_path,'descriptors_value.csv'), index=False)
        self.descriptor_df = self.descriptor_df[['Descriptor', 'Accuracy_test', 'Accuracy_avg_fold_test',
                                                 'Num_features', 'Num_operators', 'Num_total']]
        self.descriptor_df.to_csv(os.path.join(self.out_path, 'descriptors.csv'), index=False)
            
