import pandas as pd

df = pd.read_csv('/scratch/x3100a06/tRP/Data/train_source.csv')
df['Substance'] = df['Compound']
del df['Compound']
del df['Source'], df['Type']
del df['criterion'], df['t_RP'], df['p_Label']
del df['cnA1'], df['cnA2'], df['cnB1'], df['cnB2'], df['cnX']

fold_1 = pd.read_csv('/scratch/x3100a06/tRP/SISSO/run/fold_1/split_1.csv')
fold_2 = pd.read_csv('/scratch/x3100a06/tRP/SISSO/run/fold_2/split_2.csv')
fold_3 = pd.read_csv('/scratch/x3100a06/tRP/SISSO/run/fold_3/split_3.csv')
fold_4 = pd.read_csv('/scratch/x3100a06/tRP/SISSO/run/fold_4/split_4.csv')
fold_5 = pd.read_csv('/scratch/x3100a06/tRP/SISSO/run/fold_5/split_5.csv')

def merge_df(fold_df, index):
    try:
        del fold_df['Type']
    except KeyError:
        pass

    # 병합 기준 컬럼 설정
    common_keys = ['Substance', 'Label', 'n', 'A1', 'A2', 'B1', 'B2', 'X1', 'X2']

    # 데이터 타입 변환 (문자열과 숫자 구분)
    for col in common_keys:
        if col in ['Substance', 'A1', 'A2', 'B1', 'B2', 'X1', 'X2']:
            df[col] = df[col].astype(str).str.strip()
            fold_df[col] = fold_df[col].astype(str).str.strip()
        else:
            df[col] = round(df[col].astype(float), 5)
            fold_df[col] = round(fold_df[col].astype(float), 5)

    # 병합 수행 (outer join)
    merge_fold = df.merge(fold_df, on=common_keys, how='outer', suffixes=('', '_fold'))

    # fold_df에 있는 값들을 우선적으로 사용
    for col in fold_df.columns:
        if col not in common_keys:  # 기준 컬럼 제외
            if col in merge_fold.columns and f"{col}_fold" in merge_fold.columns:
                merge_fold[col] = merge_fold[f"{col}_fold"].combine_first(merge_fold[col])
                merge_fold.drop(columns=[f"{col}_fold"], inplace=True)

    # fold_X 값이 존재하는 행만 유지
    merge_fold = merge_fold[merge_fold[f'fold_{index}'].notna()].reset_index(drop=True)

    # 검증: `Substance` 기준으로 데이터 일관성 확인
    fold_df_comp = sorted(set(fold_df['Substance'].to_list()))
    merge_df_comp = sorted(set(merge_fold['Substance'].to_list()))

    if fold_df_comp != merge_df_comp:
        print('Error: Mismatch in Substance values')
        print(f'fold_df_comp: {fold_df_comp}')
        print(f'merge_df_comp: {merge_df_comp}')

    # `train` 컬럼 제거 (존재하면)
    if 'train' in merge_fold.columns:
        del merge_fold['train']

    # fold_{index} 값이 'train'인 행만 유지
    merge_fold = merge_fold[merge_fold[f'fold_{index}'] == 'train'].reset_index(drop=True)
    fold_df = fold_df[fold_df[f'fold_{index}']=='train'].reset_index(drop=True)

    # fold_df의 Substance 순서를 유지하도록 정렬
    merge_fold = merge_fold.set_index('Substance').loc[fold_df['Substance']].reset_index()

    # 최종 저장할 컬럼 리스트 설정 (df와 fold_df에서 자동으로 가져옴)
    final_columns = ['Substance', 
                     'rA', 'rB', 'rX', 
                     'nA', 'nB', 'nX', 
                     'xA', 'xB', 'xX',
                     'iA', 'iB', 'iX',
                     'vA', 'vB', 'vX']

    # NaN 값 검출 및 오류 발생
    if merge_fold[final_columns].isna().any().any():
        nan_rows = merge_fold[merge_fold[final_columns].isna().any(axis=1)]
        print("Error: NaN values detected before saving:")
        print(nan_rows)
        raise ValueError("NaN values found in data. Check the above rows before proceeding.")

    # 데이터 저장
    output_path = f'/scratch/x3100a06/tRP/SISSO/run/2_1/fold_{index}/split_{index}.csv'
    merge_fold[final_columns].to_csv(output_path, index=False)

    return merge_fold

fold1 = merge_df(fold_1, 1)
fold2 = merge_df(fold_2, 2)
fold3 = merge_df(fold_3, 3)
fold4 = merge_df(fold_4, 4)
fold5 = merge_df(fold_5, 5)