
import pandas as pd
import numpy as np
"""输入参数:由于即时调度时间太短，现将时间参数放大十倍"""

"""前置仓参数"""
L=15 #通道长度
W=2  #两通道之间距离
Vpick=15 #拣选速度
Vtravel=80 #行走速度
t_set_up=0.15*10 #拣货准备时间（并入拣货时间计算）
t_convey=round(3/4,1)*10 #分区传送时间
t_pack=0.05*10 #每个品项需要0.05分钟打包时间(单个订单打包时间控制在1分钟以内)
c_op=1.5 #单位时间订单拣选成本

"""配送参数"""
v_trans=500 #车辆行驶速度：500米/分钟 
Q=12 #配送车容量（最大载重量）
delivery_cost_perMin=0
c_d=0.005 # 配送员行驶成本（车耗、燃油、工资）5元/公里
f=3  #派车固定成本（配送员每单基础工资）
G=999
h1=0.3 #from o to i caution intensity
# h2= 0.75*h1  #from i to j caution intensity i,j!=0
h2= 0.15
"""其他参数"""

num=25 #客户订单/数

#订单履行时间期限
lm=30*10 #30min
c_od=2 #拣选逾期单位惩罚成本
c_od_d=5 #配送逾期单位惩罚成本
url="D://Onedrive/SEU/Paper_chen/Paper_2/data_figure_table/cus_data_origin.csv" #数据存放文件路径
rawdata=pd.read_csv(url,nrows =num+1, header=None)
# 坐标
X = list(rawdata.iloc[:, 1])
Y = list(rawdata.iloc[:, 2])
# 最早到达时间
eh = list(rawdata.iloc[:, 4])
# 最晚到达时间
lh = list(rawdata.iloc[:, 5])
demands=list(rawdata.iloc[:, 3]) #q[i]
srtime=list(rawdata.iloc[:, 3])
location=list(zip(X,Y))
time_windows =[(eh[i], lh[i]) for i in  range(len(rawdata))]
def distance(location):
    row=len(location)
    Dis=np.zeros((row,row))
    for i in range(row):
        for j in range(i+1,row):
            Dis[i,j]=(abs(location[i][0]-location[j][0])+abs(location[i][1]-location[j][1]))*300
            Dis[j,i]=Dis[i,j]
    return Dis
D=distance(location)
def traveltime(D,v,h1,h2):
    row=D.shape[0]
    T=np.zeros((row,row))
    for i in range(row):
        for j in range(row):
            if i==0:
                T[i,j]=D[i,j]/(v*(1-h1))
            elif j==0:
                T[i,j]=D[i,j]/v
            else:
                T[i,j]=D[i,j]/(v*(1-h2))
    return T
T=traveltime(D,v_trans,h1,h2)