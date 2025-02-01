import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ------------------------------
# 폰트 및 플롯 설정
# ------------------------------
arial_path = '/scratch/x3100a06/miniconda3/fonts/Arial.ttf'
fm.fontManager.addfont(arial_path)
mp.rcParams['font.family'] = 'Arial'

# 플롯 사이즈 관련 상수 (fig_width: 인치 단위)
fig_width = 4  
title_size_ratio = 0.08
axis_title_size_ratio = 0.07
tick_label_size_ratio = 0.055
legend_size_ratio = 0.055
text_size_ratio = 0.04

basis = fig_width
title_size = title_size_ratio * basis * 72      # 포인트 단위 (1인치 = 72pt)
axis_title_size = axis_title_size_ratio * basis * 72
tick_label_size = tick_label_size_ratio * basis * 72
legend_size = legend_size_ratio * basis * 72
text_size = text_size_ratio * basis * 72

tick_label_pad_size = 0.01

# ------------------------------
# 데이터 불러오기 및 전처리
# ------------------------------
data_path = '/scratch/x3100a06/tRP/Data/form_source.csv'
df = pd.read_csv(data_path)
print("Total records:", len(df))

# 'rp' descriptor 계산
df['rp'] = ((df['rX'] / df['xX']) * np.sqrt(df['rB'])) / (df['rA']**2)
descriptor_name = 'rp'

# 형식 통일
df['f.e'] = df['f.e'].astype(float)
df[descriptor_name] = df[descriptor_name].astype(float)

# 원소 그룹 정의
chalcogen = ['O', 'S', 'Se', 'Te']
halogen = ['F', 'Cl', 'Br', 'I']

# f.e 값이 0 또는 NaN 인 행 제거
df = df[(df['f.e'] != 0) & (df['f.e'].notna())].reset_index(drop=True)

# 할로겐 및 칼코젠 데이터셋 분리
halogen_df = df[df['X1'].isin(halogen) & df['X2'].isin(halogen)].reset_index(drop=True)
chalcogen_df = df[(df['X1'].isin(chalcogen)) | (df['X2'].isin(chalcogen))].reset_index(drop=True)

# 각 데이터셋에 anion 라벨 추가
halogen_df['anion'] = 'halogen'
chalcogen_df['anion'] = 'chalcogen'

# 필요에 따라 전체 데이터셋으로 합치기 (여기서는 개별 플롯에 사용)
df_combined = pd.concat([halogen_df, chalcogen_df], axis=0).reset_index(drop=True)

# ------------------------------
# 플롯팅 함수 정의
# ------------------------------
def plot_regression(data, descriptor, formation, xlim, xtick_step, ylim, ytick_step,
                    fill_color, regression_color, label_prefix, output_filename):
    """
    formation energy와 지정된 descriptor 사이의 선형 회귀 분석 및 플롯을 생성합니다.
    
    Parameters:
      data: 데이터가 저장된 DataFrame.
      descriptor: x축에 사용할 컬럼 (예: 'rp').
      formation: y축에 사용할 컬럼 (예: 'f.e').
      xlim: (xmin, xmax) 튜플, x축 범위.
      xtick_step: x축 눈금 간격.
      ylim: (ymin, ymax) 튜플, y축 범위.
      ytick_step: y축 눈금 간격.
      fill_color: 산점도, 채움영역, 기준선에 사용할 색상.
      regression_color: 회귀선 색상.
      label_prefix: 범례에 사용할 식별자 (예: 'C' 또는 'H').
      output_filename: 생성된 플롯을 저장할 파일 경로.
    """
    # 데이터 추출 및 회귀분석 수행
    x = data[descriptor].astype(float).values
    y = data[formation].astype(float).values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"{label_prefix} r-value: {r_value}")
    
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    bias = -coeffs[1] / coeffs[0]
    print(f"{label_prefix} bias: {bias}")
    
    # 플롯 생성
    fig, ax = plt.subplots(figsize=(10, 9))
    xmin, xmax = xlim
    ymin, ymax = ylim
    
    # bias 기준 영역 채우기
    ax.fill_between([xmin, bias, bias, xmin],
                    [ymin, ymin, 0, 0],
                    color=fill_color, alpha=0.2)
    ax.fill_between([bias, xmax, xmax, bias],
                    [0, 0, ymax, ymax],
                    color=fill_color, alpha=0.2)
    
    # 기준선 그리기
    ax.vlines(bias, ymin, ymax, color=fill_color, linestyle='-', linewidth=2, alpha=0.2)
    ax.hlines(0, xmin, xmax, color=fill_color, linestyle='-', linewidth=2, alpha=0.2)
    
    # 데이터 산점도: 'Label' 값에 따라 마커와 크기 결정
    for _, row in data.iterrows():
        if row['Label'] == 1:
            marker = '*'
            size = 150
        elif row['Label'] == 0:
            marker = 'o'
            size = 75
        else:
            continue
        ax.scatter(row[descriptor], row[formation], color=fill_color, s=size,
                   lw=1, marker=marker, alpha=0.7, edgecolor='black')
    
    # 회귀선 플롯
    reg_x = np.array([xmin, xmax + 0.1])
    ax.plot(reg_x, poly(reg_x), linestyle='--', color=regression_color,
            alpha=0.7, linewidth=3)
    
    # 축 범위 및 눈금 설정
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax, xtick_step))
    ax.tick_params(axis='x', labelsize=axis_title_size, length=5, direction='in')
    
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax, ytick_step))
    ax.tick_params(axis='y', labelsize=axis_title_size, length=5, direction='in')
    
    ax.tick_params(axis='both', which='both', direction='in', top=False, right=False)
    
    # 축 라벨 설정
    ax.set_xlabel(r'$\mathit{t}_{\mathit{RP}}$', fontsize=title_size, labelpad=tick_label_pad_size)
    ax.set_ylabel(r'$\mathit{E}_{\mathit{form}}$ (eV/f.u)', fontsize=title_size, labelpad=tick_label_pad_size)
    
    # 커스텀 범례 설정
    custom_lines = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=fill_color,
                   markersize=15, markeredgecolor='black', markeredgewidth=1.5, label=f'RP ({label_prefix})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=fill_color,
                   markersize=10, markeredgecolor='black', markeredgewidth=1.5, label=f'Non-RP ({label_prefix})')
    ]
    font_prop = fm.FontProperties(size=legend_size, stretch='condensed')
    ax.legend(handles=custom_lines, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.0),
              prop=font_prop, handlelength=0.5, handleheight=0.5, handletextpad=0.5,
              labelspacing=0.5, columnspacing=0.7)
    
    # 스파인과 눈금선 두께 조정
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='x', length=5, direction='out', width=2)
    ax.tick_params(axis='y', length=5, direction='out', width=2)
    
    # 결과 저장 및 출력
    plt.savefig(output_filename, dpi=400, bbox_inches='tight')
    plt.show()

# ------------------------------
# 칼코젠 데이터 플롯 (예: Figure.4.a_reg.png)
# ------------------------------
plot_regression(
    data=chalcogen_df,
    descriptor=descriptor_name,
    formation='f.e',
    xlim=(0.9, 13.2),
    xtick_step=2.0,
    ylim=(-4.0, 6.2),
    ytick_step=1.0,
    fill_color='#DF6E79',
    regression_color='#c92a2a',
    label_prefix='C',
    output_filename='./Figure.4.a_reg.png'
)

# ------------------------------
# 할로겐 데이터 플롯 (예: Figure.4.b_regression.png)
# ------------------------------
plot_regression(
    data=halogen_df,
    descriptor=descriptor_name,
    formation='f.e',
    xlim=(0.0, 11),
    xtick_step=2.0,
    ylim=(-0.7, 2.4),
    ytick_step=0.5,
    fill_color='#94BBC1',
    regression_color='#1864ab',
    label_prefix='H',
    output_filename='./Figure.4.b_reg.png'
)
