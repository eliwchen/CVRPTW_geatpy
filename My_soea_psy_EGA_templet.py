# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea # 导入geatpy库
from sys import path as paths
from os import path
import time
paths.append(path.split(path.split(path.realpath(__file__))[0])[0])

class My_soea_psy_EGA_templet(ea.SoeaAlgorithm):
    
    """
Msoea_psy_EGA_templet.py - Polysomy Elitist Reservation GA templet(精英保留的多染色体遗传算法模板)

模板说明:
    该模板基于内置算法模板soea_EGA_templet的多染色体版本，
    因此里面的种群对象为支持混合编码的多染色体种群类PsyPopulation类的对象。
    
算法描述:
    本模板实现的是基于杰出保留的单目标遗传算法。算法流程如下：
    1) 根据编码规则初始化N个个体的种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 独立地从当前种群中选取N-1个母体。
    5) 独立地对这N-1个母体进行交叉操作。
    6) 独立地对这N-1个交叉后的个体进行变异。
    7) 计算当代种群的最优个体，并把它插入到这N-1个交叉后的个体的第一位，得到新一代种群。
    8) 回到第2步。

模板使用注意:
    本模板调用的目标函数形如：aimFunc(pop), 
    其中pop为种群类的对象，代表一个种群，
    pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵，
    该函数根据该Phen计算得到种群所有个体的目标函数值组成的矩阵，并将其赋值给pop对象的ObjV属性。
    例如：population为一个种群对象，则调用aimFunc(population)即可完成目标函数值的计算，
         此时可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。
    若不符合上述规范，则请修改算法模板或自定义新算法模板。
    
"""
    
    def __init__(self, problem, population):
        ea.SoeaAlgorithm.__init__(self, problem, population) # 先调用父类构造方法
        if str(type(population)) != "<class 'PsyPopulation.PsyPopulation'>":
            raise RuntimeError('传入的种群对象必须为PsyPopulation类型')
        self.name = 'psy-EGA'
        self.selFunc = 'tour' # 锦标赛选择算子
        # 由于有多个染色体，因此需要用多个重组和变异算子
        self.recOpers = []
        self.mutOpers = []
        for i in range(population.ChromNum):
            if population.Encodings[i] == 'P':
                recOper = ea.Xovpmx(XOVR = 0.8) # 生成部分匹配交叉算子对象
                mutOper = ea.Mutinv(Pm = 0.7) # 生成逆转变异算子对象
            else:
                recOper = ea.Xovdp(XOVR = 1) # 生成两点交叉算子对象
                if population.Encodings[i] == 'BG':
                    mutOper = ea.Mutbin(Pm = None) # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
                elif population.Encodings[i] == 'RI':
                    mutOper = ea.Mutbga(Pm = 1/self.problem.Dim, MutShrink = 0.5, Gradient = 20) # 生成breeder GA变异算子对象
                else:
                    raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
            self.recOpers.append(recOper)
            self.mutOpers.append(mutOper)
            
    def finishing(self, population): # 重新改写,进化完成后调用的函数
        # 处理进化记录器
        delIdx = np.where(np.isnan(self.obj_trace))[0]
        self.obj_trace = np.delete(self.obj_trace, delIdx, 0)
        self.var_trace = np.delete(self.var_trace, delIdx, 0)
        if self.obj_trace.shape[0] == 0:
            raise RuntimeError('error: No feasible solution. (有效进化代数为0，没找到可行解。)')
        self.passTime += time.time() - self.timeSlot # 更新用时记录
        # 绘图
        if self.drawing != 0:
            #调用绘图函数，用C++写的底层函数
            ea.trcplot(self.obj_trace,
                       [['The Mean Value', 'The Best Value']], 
                       xlabels = [['Number of Generation']],
                       ylabels = [['Value']], 
                       gridFlags = [[False]])
        # 返回最后一代种群、进化记录器、变量记录器以及执行时间
        return [population, self.obj_trace, self.var_trace]
              
    
        
    def run(self, prophetPop = None): 
        # prophetPop为先知种群（即包含先验知识的种群）
        #==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization() # 初始化算法模板的一些动态参数
        #===========================准备进化============================
        population.initChrom(NIND) # 初始化种群染色体矩阵（内含染色体解码，详见PsyPopulation类的源码）
        
        #插入较优初始解
        
        from MyFunction import Chroms_pro
        population.setChrom(Chroms_pro) #插入想要初始解
        # from MyFunction import Chroms1,Chroms2
        # population.setChrom(Chroms2)
        # population.setChrom(Chroms1)
        # print(population.Chroms)
       
        self.problem.aimFunc(population) # 计算种群的目标函数值
        self.evalsNum = population.sizes # 记录评价次数
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查，故应确保prophetPop是一个种群类且拥有合法的Chrom、ObjV、Phen等属性）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND] # 插入先知种群
        population.FitnV = ea.scaling(self.problem.maxormins * population.ObjV) # 计算适应度
        #===========================开始进化============================
        """EGA"""
        # while self.terminated(population) == False:
        #     bestIndi = population[np.argmax(population.FitnV, 0)] # 得到当代的最优个体
        #     # 选择
        #     offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND - 1)]
        #     # 进行进化操作，分别对各种编码的染色体进行重组和变异
        #     for i in range(population.ChromNum):
        #         offspring.Chroms[i] = self.recOpers[i].do(offspring.Chroms[i]) # 重组
        #         offspring.Chroms[i] = self.mutOpers[i].do(offspring.Encodings[i], offspring.Chroms[i], offspring.Fields[i]) # 变异
        #     # 求进化后个体的目标函数值
        #     offspring.Phen = offspring.decoding() # 染色体解码
        #     self.problem.aimFunc(offspring) # 计算目标函数值
        #     self.evalsNum += offspring.sizes # 更新评价次数
        #     population = bestIndi + offspring # 更新种群
        #     population.FitnV = ea.scaling(self.problem.maxormins * population.ObjV) # 计算适应度
        
        """SEGA""" #增强版精英保留
        while self.terminated(population) == False:
            # 选择
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # 进行进化操作，分别对各种编码的染色体进行重组和变异
            for i in range(population.ChromNum):
                offspring.Chroms[i] = self.recOpers[i].do(offspring.Chroms[i]) # 重组
                offspring.Chroms[i] = self.mutOpers[i].do(offspring.Encodings[i], offspring.Chroms[i], offspring.Fields[i]) # 变异
            # 求进化后个体的目标函数值
            offspring.Phen = offspring.decoding() # 染色体解码
            self.problem.aimFunc(offspring) # 计算目标函数值
            self.evalsNum += offspring.sizes # 更新评价次数
            population = population + offspring # 父子合并
            population.FitnV = ea.scaling(self.problem.maxormins * population.ObjV, population.CV) # 计算适应度
            # 得到新一代种群
            population = population[ea.selecting('dup', population.FitnV, NIND)] # 采用基于适应度排序的直接复制选择生成新一代种群
        
        return self.finishing(population) # 调用finishing完成后续工作并返回结果
    