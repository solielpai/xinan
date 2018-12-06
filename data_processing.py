# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:08:31 2018

@author: Solielpai
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
col_numeric  = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10',
                'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 
                'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 
                'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 
                'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 
                'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 
                'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_69', 'x_70', 
                'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'x_77', 'x_78', 'x_79', 'x_80',
                'x_81', 'x_82', 'x_83', 'x_84', 'x_85', 'x_86', 'x_87', 'x_88', 'x_89', 'x_90', 
                'x_91', 'x_93', 'x_95','x_92','x_94' ]

col_discrete = ['x_96', 'x_97', 'x_98', 'x_99', 'x_100', 'x_101','x_102', 'x_103', 'x_104', 'x_105', 'x_106', 
                'x_107', 'x_108', 'x_109', 'x_110', 'x_111', 'x_112', 'x_113', 'x_114', 'x_115', 'x_116', 
                'x_117', 'x_118', 'x_119', 'x_120', 'x_121', 'x_122', 'x_123', 'x_124', 'x_125', 'x_126', 
                'x_127', 'x_128', 'x_129', 'x_130', 'x_131', 'x_132', 'x_133', 'x_134', 'x_135', 'x_136', 
                'x_137', 'x_138', 'x_139', 'x_140', 'x_141', 'x_142','x_143', 'x_144', 'x_145', 'x_146',
                'x_147', 'x_148', 'x_149', 'x_150', 'x_151', 'x_152','x_153', 'x_154', 'x_155', 'x_156',
                'x_157',]

fal_dis=['x_28','x_94','x_23','x_34','x_12','x_18','x_6','x_49','x_90','x_91','x_3','x_5','x_82','x_9',
         'x_92',
         'x_85']#dfdf
#数据路径
#aa=[]
#for x in range(102,139):
#    y=str('x_')+str(x)
#    aa.append(y)
  
path_train = r'C:\Users\Administrator\Desktop\da\data\da\train_xy.csv'
path_test = r'C:\Users\Administrator\Desktop\da\data\da\test_all.csv'
path_va=r'C:\Users\Administrator\Desktop\da\data\da\train_x.csv'
#读取数据
train_data = pd.read_csv(path_train,encoding='gbk')
test_data=pd.read_csv(path_test,encoding='gbk')
valid=pd.read_csv(path_va,encoding='gbk')
#label=train_data['y']
test_data['y']=-1
#valid['y']=-2
data=pd.concat([train_data,test_data],axis=0,ignore_index=True)
cc=data.nunique()
data.drop('cust_group',axis=1,inplace=True) 
#data=pd.concat([data,valid],axis=0,ignore_index=True)
d1=data[['cust_id']+col_numeric]
a=d1.columns
data=data.drop(col_numeric,axis=1)
for var in col_numeric:
    na_model=Imputer(missing_values=-99,strategy='mean',axis=0)
    d1=na_model.fit_transform(d1)
d2=pd.DataFrame(d1,columns=a)
data=pd.merge(data,d2,on='cust_id')

for var in fal_dis:
    var_dummies = pd.get_dummies(data[var])
    var_dummies.columns = [var+'_'+str(i) for i in range(var_dummies.shape[1])]
#    col_numeric.remove(var)
#    data.drop(var,axis=1,inplace=True)
    data = pd.concat([data,var_dummies],axis=1)
    
for var in col_discrete :
    var_dummies = pd.get_dummies(data[var])
    var_dummies.columns = [var+'_'+str(i) for i in range(var_dummies.shape[1])]
#    if var not in ['draft_param4','draft_param6','draft_param7']:
    data.drop(var,axis=1,inplace=True)
    data = pd.concat([data,var_dummies],axis=1) 
#    x=int(x.lstrip('group_'))
#data['cust_group']=data['cust_group'].str.lstrip('group_')
#lis=['x_94','x_92','x_96', 'x_97', 'x_98', 'x_99', 'x_100', 'x_101', 'x_102', 'x_103', 'x_104', 'x_105', 'x_106', 'x_107', 'x_108', 'x_109', 'x_110', 'x_111', 'x_112', 'x_113', 'x_114', 'x_115', 'x_116', 'x_117', 'x_118', 'x_119', 'x_120', 'x_121', 'x_122', 'x_123', 'x_124', 'x_125', 'x_126', 'x_127', 'x_128', 'x_129', 'x_130', 'x_131', 'x_132', 'x_133', 'x_134', 'x_135', 'x_136', 'x_137', 'x_138', 'x_139', 'x_140', 'x_141', 'x_142', 'x_143', 'x_144', 'x_145', 'x_146', 'x_147', 'x_148', 'x_149', 'x_150', 'x_151', 'x_152', 'x_153', 'x_154', 'x_155', 'x_156', 'x_157']
#l=['x_16','x_19','x_13']
##'x_18',,'x_12','x_23','x_6','x_28','x_34'
#data.drop(l,axis=1,inplace=True)
#a=[]
#for x in col_numeric:
#    if x not in l:
#        a.append(x)
#col_numeric=a        
#a=[]
#for x in col_discrete:
#    if x not in l:
#        a.append(x)
#col_discrete=a

data_Raw = data.copy()



############训练集start#############
data_id=data_Raw[['cust_id']]

data_num =  data_Raw[['cust_id']+col_numeric].copy()
data_rank = pd.DataFrame(data_id,columns=['cust_id'])
for feature in col_numeric:
    data_rank['r'+feature] = data_num[feature].rank(method='max')
    


# 离散数据
data_x_discretization = data_rank.copy()
data_x_discretization = data_x_discretization.drop(['cust_id'],axis=1)
data_x_discretization[data_x_discretization<1500] = 1
data_x_discretization[(data_x_discretization>=1500)&(data_x_discretization<3000)] = 2
data_x_discretization[(data_x_discretization>=3000)&(data_x_discretization<4500)] = 3
data_x_discretization[(data_x_discretization>=4500)&(data_x_discretization<6000)] = 4
data_x_discretization[(data_x_discretization>=6000)&(data_x_discretization<7500)] = 5
data_x_discretization[(data_x_discretization>=7500)&(data_x_discretization<9000)] = 6
data_x_discretization[(data_x_discretization>=9000)&(data_x_discretization<10500)] = 7
data_x_discretization[(data_x_discretization>=10500)&(data_x_discretization<12000)] = 8
data_x_discretization[(data_x_discretization>=12000)&(data_x_discretization<13500)] = 9
data_x_discretization[data_x_discretization>=13500] = 10
#离散特征的命名：在原始特征前加'd',如'x1'的离散特征为'dx1'
rename_dict = {s:'d'+s[1:] for s in data_x_discretization.columns.tolist()}
data_x_discretization = data_x_discretization.rename(columns=rename_dict)
data_x_discretization['cust_id'] =  data_id




 


#计数数据
data_x_nd = data_x_discretization.copy()
data_x_nd['n1'] = (data_x_nd==1).sum(axis=1)
data_x_nd['n2'] = (data_x_nd==2).sum(axis=1)
data_x_nd['n3'] = (data_x_nd==3).sum(axis=1)
data_x_nd['n4'] = (data_x_nd==4).sum(axis=1)
data_x_nd['n5'] = (data_x_nd==5).sum(axis=1)
data_x_nd['n6'] = (data_x_nd==6).sum(axis=1)
data_x_nd['n7'] = (data_x_nd==7).sum(axis=1)
data_x_nd['n8'] = (data_x_nd==8).sum(axis=1)
data_x_nd['n9'] = (data_x_nd==9).sum(axis=1)
data_x_nd['n10'] = (data_x_nd==10).sum(axis=1)
data_x_nd = data_x_nd[['cust_id','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]




data_eleven = pd.merge(data_x_nd,data_id,on='cust_id')[['cust_id','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]

rank_data_xx =data_rank.drop('cust_id',axis = 1)
rank_data = rank_data_xx 
rank_data['cust_id'] = data_id

discret_data = data_x_discretization.copy()


#将原始特征，排序特征，离散特征，以及其他11维特征（n1～n10，discret_null）合并
data_xy = pd.merge(data_Raw,rank_data,on='cust_id')
data_xy = pd.merge(data_xy,discret_data,on='cust_id')
data_xy = pd.merge(data_xy,data_eleven,on='cust_id')
#data_xy['cust_group']=data_xy['cust_group'].astype('int')
#data_xy.drop('cust_group',axis=1,inplace=True)
dee=data_xy.describe()
for x in data_xy.columns:
    deee=data_xy[x]
#    if x!='cust_group':
        
    if deee.std()<0.005:
        data_xy.drop(x,axis=1,inplace=True)
l=['x_80','x_2','x_81','x_92','x_1','x_48','x_63','x_54','x_52','x_55','x_62',
   'n6','n7','n8','n2','n3','n5','n10','n4','n9']
i=0
m=l.copy()
abab=[]
#for x in l:
#    
#    m.remove(x)
#    for y in m:
#       # print(l,'sdsdsdsd')
#        i=str(x)+str(y)
#        abab.append(i)
#        data_xy[i]=data_xy[x]*data_xy[y]
#        


#train_data=data[data['y']!=-1]
#test_data=data[data['y']==-1]

aaa= data.corr()
train=data_xy[data_xy['y']>-1]
test=data_xy[data_xy['y']==-1]
valid=data_xy[data_xy['y']==-2]