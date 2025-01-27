import os
import numpy as np
from sklearn.tree import export_graphviz

def math_functions(expression):
    replacements = {
        'sqrt': 'np.sqrt',
        'log': 'np.log',
        'exp': 'np.exp',
        '^-1': '**-1',
        '^': '**'}
    
    for old, new in replacements.items():
        expression = expression.replace(old, new)
    return expression

def bias_functions(estimators,
                   path='./',
                   num=0):
    bias_list = []
    for length in range(len(estimators)):
        best_rf = estimators[length]
        os.makedirs(f'{path}/Tree_{num}', exist_ok=True)
        
        export_graphviz(best_rf, 
                        out_file=f'{path}/Tree_{num}/{length}_tree.dot', 
                        feature_names=['Descriptor'],
                        class_names=['class_1','class_2'], 
                        rounded = True, 
                        filled = True)
        
        with open(f'{path}/Tree_{num}/{length}_tree.dot','r') as f:
            line = f.readlines()
            value = float(str(line[3].split('entropy')[0].split(' ')[-1].strip('\\n').strip(' ')))
        bias_list.append(value)
        bias = np.array(bias_list)
        min_bias, max_bias = np.min(bias), np.max(bias)
    return np.arange(min_bias, max_bias+0.01, 0.01).tolist()

def set_bias_functions(df, desc_column, label_column, bias=0.0, large=False):
    all_rp = df[df[label_column]==1].reset_index(drop=True)
    all_non_rp = df[df[label_column]!=1].reset_index(drop=True)
    if large:
        rp = df[(df[desc_column]>=bias)&(df[label_column]==1)].reset_index(drop=True)
        rp_err = df[(df[desc_column]<bias)&(df[label_column]==1)].reset_index(drop=True)
        non_rp = df[(df[desc_column]<bias)&(df[label_column]!=1)].reset_index(drop=True)
        non_rp_err = df[(df[desc_column]>=bias)&(df[label_column]!=1)].reset_index(drop=True)
    else:
        rp = df[(df[desc_column]<=bias)&(df[label_column]==1)].reset_index(drop=True)
        rp_err = df[(df[desc_column]>bias)&(df[label_column]==1)].reset_index(drop=True)
        non_rp = df[(df[desc_column]>bias)&(df[label_column]!=1)].reset_index(drop=True)
        non_rp_err = df[(df[desc_column]<=bias)&(df[label_column]!=1)].reset_index(drop=True)
    
    TP = len(rp)
    TN = len(non_rp)
    FP = len(non_rp_err)
    FN = len(rp_err)
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    accuracy_rp = len(rp)/len(all_rp)
    accuracy_non_rp = len(non_rp)/len(all_non_rp)
    
    precision = TP/(TP+FP) if TP+FP != 0 else 0
    recall = TP/(TP+FN) if TP+FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    
    assert len(rp)+len(rp_err)==len(all_rp)
    assert len(non_rp)+len(non_rp_err)==len(all_non_rp)
    
    return f1_score, accuracy, accuracy_rp, accuracy_non_rp

def get_max(biases=[], acc_list=[], large_list=[]):
    acc_np = np.array(acc_list)
    max_idx = np.where(acc_np==np.max(acc_np))[0][0]
    large_val = np.array(large_list)[max_idx]
    return biases[max_idx], max_idx, large_val

def get_bias_functions(df, desc_column, label_column, biases=[], large=None):
    b_f1, b_accuracy, b_accuracy_rp, b_accuracy_non_rp, large_vals, bias_vals = [], [], [], [], [], []
    for bias in biases:
        s_f1, s_accuracy, s_accuracy_rp, s_accuracy_non_rp = set_bias_functions(df, desc_column, label_column, bias, large=False)
        l_f1, l_accuracy, l_accuracy_rp, l_accuracy_non_rp = set_bias_functions(df, desc_column, label_column, bias, large=True)
        if large is None:    
            if s_f1 > l_f1:
                f1, accuracy, accuracy_rp, accuracy_non_rp = s_f1, s_accuracy, s_accuracy_rp, s_accuracy_non_rp
                large_val, bias_val = 'x', bias
            else:
                f1, accuracy, accuracy_rp, accuracy_non_rp = l_f1, l_accuracy, l_accuracy_rp, l_accuracy_non_rp
                large_val, bias_val = 'o', bias
        else:
            if large=='o':
                f1, accuracy, accuracy_rp, accuracy_non_rp = l_f1, l_accuracy, l_accuracy_rp, l_accuracy_non_rp
                large_val, bias_val = 'o', bias
            elif large=='x':
                f1, accuracy, accuracy_rp, accuracy_non_rp = s_f1, s_accuracy, s_accuracy_rp, s_accuracy_non_rp
                large_val, bias_val = 'x', bias
            else:
                bias_val = 0.0
                large_val = ('x' if large=='o' else 'o')
                f1 = 0.0
                accuracy = 0.0
                accuracy_rp = 0.0
                accuracy_non_rp = 0.0

        bias_vals.append(bias_val)       
        large_vals.append(large_val)
        b_f1.append(f1)
        b_accuracy.append(accuracy)        
        b_accuracy_rp.append(accuracy_rp)
        b_accuracy_non_rp.append(accuracy_non_rp)
    
    max_f1, f1_idx, max_large = get_max(bias_vals, b_f1, large_vals)
    max_bias, acc_idx, _ = get_max(bias_vals, b_accuracy, large_vals)
    max_bias_rp, idx_rp, _ = get_max(bias_vals, b_accuracy_rp, large_vals)
    max_bias_non_rp, idx_non_rp, _ = get_max(bias_vals, b_accuracy_non_rp, large_vals)
    
    bias_dict = {'len': len(df),
                 'max_large': max_large,
                 'max_bias': max_bias,
                 'max_accuracy': b_accuracy[acc_idx],
                 }
        
    return bias_dict
            