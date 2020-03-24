# -*- coding: utf-8 -*-

import numpy as np
from CVRPDV_ortools import vehicle_routing_OR
from CVRPV_CW import vehicle_routing_CW
def Ind2Chroms(vehicle_routing):
    Chrom1=[i for i in vehicle_routing if i !=0]
    Chrom2=[]
    indx=0
    for i in vehicle_routing:
        if i==0:
            indx+=1
        else:
            Chrom2.append(indx)
    Chroms=[Chrom1,Chrom2]    
    return Chroms  


Chroms1=Ind2Chroms(vehicle_routing=vehicle_routing_OR)
Chroms2=Ind2Chroms(vehicle_routing=vehicle_routing_CW)

Chroms_pro=[np.array([Chroms1[0],Chroms2[0]]),np.array([Chroms1[1],Chroms2[1]])]
# print(Chroms_pro)

