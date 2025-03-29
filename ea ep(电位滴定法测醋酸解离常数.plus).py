import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import csv
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
import pandas as pd

name1 = '刘兴盛.csv'  # 输入已经转好格式的包含V和pH的数据
V_buchong = []  # 需要补充的V数据
pH_buchong = []  # 需要补充的pH数据

V_shanchu=[]
pH_shanchu=[]
'''zrl
V_buchong = [18.55, 18.8]  # 需要补充的V数据
pH_buchong = [6.44, 6.85]  # 需要补充的pH数据
'''
# --------------以下请勿更改！！！---------------

ls = []
# 一个打开文件的程序块
with open(name1, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        str_data = ','.join(row)
        str_data = str_data.replace('\ufeff', '')
        result = str_data.split(',')
        float_row = [float(item) for item in result]
        ls.append(float_row)

pH = []  # 用于接受B列
V = []  # 用于接受A列

pH_yuan = []  # 用于盛放原数据
V_yuan = []  # 用于盛放原数据
for i in range(len(ls)):  # 用于把存在ls列表里的数据拆开存入上述四个列表
    pH.append(ls[i][1])
    pH_yuan.append(ls[i][1])
    V.append(ls[i][0])
    V_yuan.append(ls[i][0])

V.extend(V_buchong)  # 加入补充数据
pH.extend(pH_buchong)  # 加入补充数据

# 根据V的值对合并后的列表进行排序，pH的值会自动调整位置
combined = sorted(zip(V, pH), reverse=False)  # reverse=True表示从大到小排序

# 分离排序后的V和pH列表
V_sorted, pH_sorted = zip(*combined)

# 将结果转换为列表
V = list(V_sorted)
pH = list(pH_sorted)

np_pH = np.array(pH)  # 转化为np数组
np_V = np.array(V)

delta_V = []  # 创建空列表盛放上下体积差
# 原数据计算部分——————————————————————————————————————————————————————————————————————————————————————————————

for i in range((len(ls) - 1)):  # 计算原数据上下体积差并存入
    delta_i = ls[i + 1][0] - ls[i][0]
    print(f'{ls[i + 1][0]}mL时 ΔV为{delta_i:.2f}mL')
    delta_V.append(delta_i)
print()
delta_pH = []

for i in range((len(ls) - 1)):  # 计算原数据上下pH差并存入
    delta_j = ls[i + 1][1] - ls[i][1]
    print(f'{ls[i + 1][0]}mL时 ΔpH为{delta_j:.2f}')
    delta_pH.append(delta_j)
print()
delta_pH_V = []

for i in range(len(delta_V)):  # 计算原数据的一次微分并存入
    delta_j_i = delta_pH[i] / delta_V[i]
    print(f'{ls[i + 1][0]}mL时 ΔpH/ΔV为{delta_j_i:.3f}')
    delta_pH_V.append(delta_j_i)
npdelta_pH_V = np.array(delta_pH_V)  # 改成np数组

# 引入补充数据后计算部分————————————————————————————————————————————————————————————————————————————————————————————————
y_delta_pH_V_zeng1 = []  # 定义一个列表存储引入补充数据后的一次微分

for i in range((len(np_pH) - 1)):  # 计算包含补充数据的一次微分并存入
    delta_i = np_pH[i + 1] - np_pH[i]
    delta_j = np_V[i + 1] - np_V[i]
    y_delta_pH_V_zeng1.append(delta_i / delta_j)
y_delta_pH_V_zeng = np.array(y_delta_pH_V_zeng1)
# print(len(y_delta_pH_V_zeng))
'''for x, y in zip(np_V, y_delta_pH_V_zeng):
    print(f'x: {x:.2f}, y: {y:.3f}')'''


# pH处理
# 绘制pH-V曲线的插值函数——————————————————————————————————————————————————————————————————————————————————————————————
# 这是一个三次样条插值函数
def sancichazhi(np_x, np_y):
    x_1 = np.linspace(np_x.min(), np_x.max(), 500)
    f = make_interp_spline(np_x, np_y)
    y_1 = f(x_1)
    x_2 = np.linspace(x_1.min(), x_1.max(), 1000)
    f1 = make_interp_spline(x_1, y_1)
    y_2 = f1(x_2)
    x_3 = np.linspace(x_2.min(), x_2.max(), 1500)
    f2 = make_interp_spline(x_2, y_2)
    y_3 = f2(x_3)
    return x_3, y_3


x_V, y_pH = sancichazhi(np_V, np_pH)  # 对包含补充数据的x,y进行插值
V122 = []  # 用于原数据的一次微分图的x轴
for i in range(len(V_yuan) - 1):
    V122.append(V[i + 1])
np_V112 = np.array(V122)

V1 = []  # 用于包含补充数据的一次微分图x轴
for i in range(len(V) - 1):
    V1.append(V[i + 1])
np_V11 = np.array(V1)
# print(len(np_V11),len(y_delta_pH_V_zeng))
# 绘制δpH/δV-V曲线的插值函数

x_V1, y_delta_pH_V = sancichazhi(np_V11, y_delta_pH_V_zeng)  # 生成一次微分插值图的x,y数据
peaks, _ = find_peaks(y_delta_pH_V, prominence=1)

# 实验结果的计算
dic1 = {key: value for key, value in zip(y_delta_pH_V, x_V1)}  # 将微分值与体积打包成字典

x_V, y_pH = sancichazhi(np_V, np_pH)  # 对pH-V图的数据进行插值


def find_max_key(d):  # 定义一个寻找最大键并返回对应的值的函数
    if not d:  # Check if the dictionary is empty
        return None
    max_key = max(d.keys())  # Find the maximum key in the dictionary
    return d[max_key]


print(f'完全中和体积 {find_max_key(dic1):.2f} mL ')
V11 = find_max_key(dic1)  # 寻找出微分值最大时的体积，是为拐点体积
V12 = V11 / 2  # 拐点体积除以2为等浓度时的体积
dic2 = {key: value for key, value in zip(x_V, y_pH)}  # 将体积与pH打包


def find_closest_key(input_dict, target):  # 定义一个根据输入的数，寻找最接近的键所对应的值的函数
    if not input_dict:
        return None

    closest_key = min(input_dict.keys(), key=lambda x: abs(x - target))
    return input_dict[closest_key]


result = find_closest_key(dic2, V12)  # 找出等浓度时的pH是为pKa
print('Ve/2体积时的pH  ', result)
Ka = 10 ** (-result)
print('Ka=', Ka)
# 一些相对误差计算
Er = (Ka - 1.8 * 10 ** -5) / (1.8 * 10 ** -5) * 100
print(f'Ka下的Er={Er:.2f}%')
Er = (np.log(Ka) - np.log(1.8 * 10 ** -5)) / np.log(1.8 * 10 ** -5) * 100
print(f'pKa下的Er={Er:.2f}%')

# np_V11包含着补充数据，作为插值的x数据  y_delta_pH_V_zeng 包含着补充数据，作为插值y数据
np_V11 = np.delete(np_V11, 0)  # 删除第一个元素，因为它与其他值离得远
y_delta_pH_V_zeng = np.delete(y_delta_pH_V_zeng, 0)  # 删除第一个元素

# B样条插值法------------------------------------------------------------------------------------------------------------

x_data = pd.Series(np_V11)
y_data = pd.Series(y_delta_pH_V_zeng)

max_index = y_data.idxmax()
max_x = x_data[max_index]
max_y = y_data[max_index]

spline_smooth_corrected = UnivariateSpline(x_data, y_data)
spline_smooth_corrected.set_smoothing_factor(1)

x_smooth = np.linspace(min(x_data), max(x_data), 200)


y_spline_smooth_corrected = spline_smooth_corrected(x_smooth)


closest_index = (np.abs(x_smooth - max_x)).argmin()
y_spline_smooth_corrected[closest_index] = max_y
x_data = x_data.values.tolist()
y_data = y_data.values.tolist()
for i in V_buchong:
    index = x_data.index(i)
    x_data.remove(i)
    y_data.pop(index)

# 绘图区
plt.scatter(x_data, y_data, label='Data', color='blue')
# plt.scatter(np_V112, npdelta_pH_V, label='Data', color='black')
plt.plot(x_smooth, y_spline_smooth_corrected, label='Peak-Corrected Spline Curve', color='red')
plt.xlabel('V (mL)')
plt.ylabel('ΔpH/ΔV (1/mL)')
plt.title('ΔpH/ΔV-V figure')
plt.plot([find_max_key(dic1), find_max_key(dic1)], [0, max_y], linestyle='dashed', color='black')
plt.text(find_max_key(dic1), 0.01, f'{find_max_key(dic1):.2f}')
plt.ylim(0)
plt.show()
# 绘制pH-V曲线————————————————————————————————————————————————————————————————————————————————————————————————————————————

# # 垂线绘制
plt.plot(x_V, y_pH)
plt.scatter(V_yuan, pH_yuan, s=20)
plt.title('pH-V figure')
plt.xlabel('V (mL)')
plt.ylabel('pH')
plt.plot([V12, V12], [0, 8], linestyle='dashed', color='black')
plt.text(V12, 0.5, f'{V12:.2f}')
plt.plot([0, V12 + 1], [result, result], linestyle='dashed', color='black')
plt.text(0.5, result, f'{result:.2f}')
plt.xlim(0)
plt.ylim(0, 14)
plt.show()
