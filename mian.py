# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import geatpy as ea # import geatpy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns
from MyProblem import MyProblem # 导入自定义问题接口
from My_soea_psy_EGA_templet import My_soea_psy_EGA_templet

"""================================外部环境设置============================"""
sns.set()#切换到seaborn的默认运行配置
sns.set_style("whitegrid")

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        # 'figure.dpi': 300,
        # 'savefig.dpi': 300,
        # 'font.style':'normal', # ‘normal’, ‘italic’ or ‘oblique’.
        # 'font.weight':'normal', #or 'blod'
        'font.size':12, #or large,small
        }
rcParams.update(params)


"""================================实例化问题对象============================"""
problem = MyProblem() # 生成问题对象
"""==================================种群设置==============================="""
NIND = 60             # 种群规模
# 创建区域描述器，这里需要创建两个，前25个变量用P编码，剩余变量用RI编码
Encodings = ['P', 'RI']
Field1 = ea.crtfld(Encodings[0], problem.varTypes[:problem.num] ,problem.ranges[:,:problem.num], problem.borders[:,:problem.num])
Field2 = ea.crtfld(Encodings[1], problem.varTypes[problem.num:], problem.ranges[:,problem.num:], problem.borders[:,problem.num:])
Fields = [Field1, Field2]
population = ea.PsyPopulation(Encodings, Fields, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）

# prophetPop = ea.PsyPopulation(Encodings, Fields, 2) # 实例化先知种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
# from MyFunction import Chroms_pro
# prophetPop.initChrom( NIND = 2)
# prophetPop.Chroms=Chroms_pro
"""================================算法参数设置============================="""
myAlgorithm = My_soea_psy_EGA_templet(problem, population) # 实例化一个算法模板对象
myAlgorithm.MAXGEN = 400 # 最大进化代数
myAlgorithm.trappedValue = 1e-10 # “进化停滞”判断阈值
myAlgorithm.maxTrappedCount = 80 # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
myAlgorithm.drawing = 1 # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）


"""=======================调用算法模板进行种群进化======================="""
[population, obj_trace, var_trace] = myAlgorithm.run() # 执行算法模板


population.save() # 把最后一代种群的信息保存到文件中
"""===============================输出结果及绘图============================"""
# 输出结果
best_gen = np.argmin(problem.maxormins * obj_trace[:, 1]) # 记录最优种群个体是在哪一代
best_ObjV = np.min(obj_trace[:, 1])
ind_best=var_trace[best_gen, :] #最优个体（基因型）
routes_best=problem.decodeInd(ind_best) #最优车辆调度
C_ol=problem.loadPenalty(routes_best)
C_od_d=problem.timePenalty(routes_best)
C_d=best_ObjV-C_ol-C_od_d

print('有效进化代数：%s'%(obj_trace.shape[0]))
print('最优的一代是第 %s 代'%(best_gen + 1))
print('评价次数：%s'%(myAlgorithm.evalsNum))
print('时间已过 %s 秒'%(myAlgorithm.passTime))


print('最优车辆调度为：',routes_best)
# print("各点访问时间：",problem.timeTable(routes_best))
print('最优目标函数值：%s'%(best_ObjV))
print("超载惩罚成本：",C_ol)
print("时间窗惩罚成本：",C_od_d)
print("配送成本：",C_d)


# 绘图
def plot(routes):
  """
  :data 配送批次顺序, eg：data=[0,10,4,14,12,0,7,8,19,3,0,9,16,0,15,5,6,0,13,11,18,20,2,0,17,1,0]
  :url：客户数据文件路径, eg:url = "D://Onedrive/SEU/Paper_chen/Paper_2/data_figure_table/cus_data_origin.csv"
  :num 客户数
  """
  rawdata = pd.read_csv(problem.url,nrows =problem.num+1, header=None)
  temp=[i[1::] for i in routes]
  ind=[0]+[item for sublist in temp for item in sublist]
  # 坐标
  X = list(rawdata.iloc[:, 1])
  Y = list(rawdata.iloc[:, 2])
  Xorder = [X[i] for i in ind]
  Yorder = [Y[i] for i in ind]
  plt.plot(Xorder, Yorder, c='black', lw=1,zorder=1)
  plt.scatter(X, Y, c='black',marker='*',zorder=2)
  plt.scatter([X[0]], [Y[0]], c='black',marker='o', zorder=3)
  # plt.scatter(X[,m:], Y[,m:], marker='^', zorder=3)
 
  plt.xticks(range(11))
  plt.yticks(range(11))
  # plt.xlabel('x坐标')
  # plt.ylabel('y坐标')
  # plt.title(self.name)
  plt.savefig('roadmap.eps', dpi=600, bbox_inches='tight')
  plt.show()

plt.figure()
plot(routes_best)



