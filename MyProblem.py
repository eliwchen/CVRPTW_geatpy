# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import geatpy as ea

"""
    一个带时间窗和载重约束的单目标车辆路径优化问题：CVRPTW

"""

class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = 'CVRPTW' # 初始化name（函数名称，可以随意设置）
        M = 1 # 初始化M（目标维数）
        maxormins = [1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 50 # 初始化Dim（决策变量维数）(订单排列以及对应指派的车辆)
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1] * Dim # 决策变量下界
        ub =[25] * int(Dim/2)+[20]*int(Dim/2) # 决策变量上界, 25张订单，20辆车
        lbin = [1] * Dim # 决策变量下边界
        ubin = [1] * Dim # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        
        """前置仓参数"""
        self.L=15 #通道长度
        self.W=2  #两通道之间距离
        self.Vpick=15 #拣选速度
        self.Vtravel=80 #行走速度
        self.t_set_up=0.15*10 #拣货准备时间（并入拣货时间计算）
        self.t_convey=round(3/4,1)*10 #分区传送时间
        self.t_pack=0.05*10 #每个品项需要0.05分钟打包时间(单个订单打包时间控制在1分钟以内)
        self.c_op=1.5 #单位时间订单拣选成本
        
        """配送参数"""
        self.v_trans=500 #车辆行驶速度：500米/分钟 
        self.Q=12 #配送车容量（最大载重量）
        self.c_d=0.005 # 配送员行驶成本（车耗、燃油、工资）5元/公里
        self.f=3  #派车固定成本（配送员每单基础工资）#3
        self.G=999
        self.h1=0.3 #from o to i caution intensity
        # self.h2= 0.75*h1  #from i to j caution intensity i,j!=0
        self.h2= 0.15
        
        """其他参数"""
        self.num=25 #客户订单/数
        self.lm=30*10 #30min #订单履行时间期限
        self.c_od_p=2 #拣选逾期惩罚因子 #2
        self.c_od_d=5 #配送逾期惩罚因子 #5
        self.c_ol=20 #超载惩罚因子 #20
        self.url="D://Onedrive/SEU/Paper_chen/Paper_2/data_figure_table/cus_data_origin.csv" #数据存放文件路径
        self.rawdata=pd.read_csv(self.url,nrows =self.num+1, header=None)
        # 坐标
        self.X = list(self.rawdata.iloc[:, 1])
        self.Y = list(self.rawdata.iloc[:, 2])
        self.locations=list(zip(self.X,self.Y)) #各点坐标
        self.eh = list(self.rawdata.iloc[:, 4])  # 最早到达时间  
        self.lh = list(self.rawdata.iloc[:, 5]) # 最晚到达时间
        self.time_windows =[(self.eh[i], self.lh[i]) for i in  range(len(self.rawdata))]
        self.demands=list(self.rawdata.iloc[:, 3]) #各点需求量
        self.svtime=list(self.rawdata.iloc[:, 6]) #各点服务时间
        
        
            
    def decodeInd(self,ind): 
        """
        功能：用于解构染色体成为路线集
        输入：染色体，例如：[1,2,4,3,5,6,3,2,1,1,2,4]
        输出：解构后的路线集，例如：[[0, 6, 0], [0, 1, 0], [0, 4, 3, 0], [0, 2, 5, 0]]
        
        """
        lind=int(len(ind)/2) #单条染色体长度（客户数）
        ind_cus=ind[:lind] 
        ind_vhi=ind[lind:] 
        Idx_vhi=[] #车辆索引值,定位承运订单
        for i in ind_vhi:
            Idx_vhi.append(tuple([m for m,x in enumerate(ind_vhi) if x==i]))
            Idx_vhi=list(set(Idx_vhi))
        routes=[]  #根据索引值，将订单进行合并  
        for i in Idx_vhi:
            temp=[0]
            for j in i:
                temp.append(int(ind_cus[j]))
            temp.append(0)    
            routes.append(temp)
        return routes
    

    def loadPenalty(self,routes):
        '''
        功能:辅助函数，因为在交叉和突变中可能会产生不符合负载约束的个体，需要对不合要求的个体进行惩罚
        输入：实例，单个个体的路径集
        输出：每个该个体的超载惩罚成本
        '''
        overload = 0
        # 计算每条路径的负载，取max(0, routeLoad - maxLoad)计入惩罚项
        for eachRoute in routes:
            routeLoad = np.sum([self.demands[i] for i in eachRoute])
            overload += max(0, routeLoad - self.Q)
        penalty=overload * self.c_ol  
        return penalty
    
    
    def calDist(self,pos1, pos2):
        '''
        功能：计算距离的辅助函数，根据给出的坐标pos1和pos2，返回两点之间的距离
        输入：  pos1, pos2 -- (x,y)元组 ;
        输出： 曼哈顿距离
        '''
        return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))*300

    def calcRouteVisitTime(self,route):
        '''
        功能：辅助函数，根据给定路径，计算该路径上各点访问时间(实际开始服务的时间)
        输入：实例，单条路径
        输出：该路径上各点访问时间(实际开始服务的时间)
        '''
        # 初始化visitTime数组，其长度应该比eachRoute小2(去除头尾的0节点)
        visitTime = [0] * len(route) 
        arrivalTime=0
        arrivalTime += MyProblem.calDist(self,self.locations[0], self.locations[route[1]])/(self.v_trans*(1-self.h1)) #从0出发到下一点的行驶时间
        arrivalTime = max(arrivalTime, self.time_windows[route[1]][0]) #实际开始服务时间（需等到左侧时间窗开启）
        visitTime[1] = arrivalTime
        arrivalTime += self.svtime[route[1]] # 在出发前往下个节点前完成服务，该值等于离开正在访问节点的时间
        
        for i in range(1, len(route)-1):
            
            # 计算从路径上当前节点[i]到下一个节点[i+1]的花费的时间 
           if route[i+1]==0:
                arrivalTime += MyProblem.calDist(self,self.locations[route[i]], self.locations[route[i+1]])/(self.v_trans*(1-0))
           else: 
                arrivalTime += MyProblem.calDist(self,self.locations[route[i]], self.locations[route[i+1]])/(self.v_trans*(1-self.h2))
           arrivalTime = max(arrivalTime, self.time_windows[route[i+1]][0])
           visitTime[i+1] = arrivalTime
           arrivalTime += self.svtime[route[i+1]] # 在前往下个节点前完成服务
        return visitTime

    def timeTable(self,routes):
        '''
        功能：辅助函数，依照给定配送计划，返回每个顾客受到服务的时间
        输入：单个个体路径集合
        输出：每个节点的访问时间
        '''
        visitTimeArrangement = [] #容器，用于存储每个顾客受到服务的时间
        for eachRoute in routes:
            visitTime = MyProblem.calcRouteVisitTime(self,eachRoute)
            visitTimeArrangement.append(visitTime)
        return visitTimeArrangement

    def timePenalty(self,routes):
        '''
        功能：辅助函数，对不能按服务时间到达顾客的情况进行惩罚
        输入:单个个体路径集合
        输出:逾期惩罚成本
        '''
        visitTimeArrangement = MyProblem.timeTable(self,routes) # 对给定路线，计算到达每个客户的时间
        visitTimeArrangement_flatten=[item for sublist in visitTimeArrangement for item in sublist] #合并子列表，拉平
        # 索引给定的最迟到达时间
        lateTimeArrangement=[]
        for eachRoute in routes:
            lateTime=[]
            for j in eachRoute:
                lateTime.append(self.time_windows[j][1])
            lateTime[-1]=100
            lateTimeArrangement.append(lateTime)
        lateTimeArrangement_flatten=[item for sublist in lateTimeArrangement for item in sublist] #合并子列表，拉平
        # 计算各节点延迟时间
        timeDelay_flatten=[max(visitTimeArrangement_flatten[i]-lateTimeArrangement_flatten[i],0) for i in range(len(visitTimeArrangement_flatten))]
        return np.sum(timeDelay_flatten)*self.c_od_d

    def calRouteLen(self,routes):
        '''辅助函数，返回给定路径的总长度'''
        totalDistance = 0 # 记录各条路线的总长度
        for eachRoute in routes:
            # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
            for i,j in zip(eachRoute[0:-1], eachRoute[1::]):
                totalDistance += MyProblem.calDist(self,self.locations[int(i)], self.locations[int(j)])    
        return totalDistance


    def evaluate(self,routes):
        '''
        功能：评价函数，目标函数值+惩罚值 
        输入：单个个体的路径集
        输出：该个体的评价值
        '''
        totalDistance = MyProblem.calRouteLen(self,routes)
        cost_num_node=0
        for i in routes:
            if len(i)<=3:
                cost_num_node=+10 #惩罚一条线路只有1个订单的情况
        C_d=self.f*len(routes)+self.c_d*totalDistance
        C_penalty=MyProblem.loadPenalty(self,routes)+MyProblem.timePenalty(self,routes)  
        return C_d+C_penalty+cost_num_node
    
    def aimFunc(self, pop): 
        '''
        功能：计算种群的目标函数
        输入：实例，种群
        输出：种群的目标函数值
        '''
        inds = pop.Phen # 得到决策变量矩阵，种群的表现型矩阵
        routess=[MyProblem.decodeInd(self,i) for i in inds] #解构后的所有个体路径集的集合
        ObjV =np.array( [MyProblem.evaluate(self,routes) for routes in routess]).T # 存储所有种群个体对应的目标函数值
        pop.ObjV=ObjV.reshape(len(ObjV),1)
   
    