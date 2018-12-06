# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:08:31 2018

@author: Solielpai
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter

from sklearn.preprocessing import Imputer
#col_numeric  = ['x_94','x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10',
#                'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 
#                'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 
#                'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 
#                'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 
#                'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 
#                'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_69', 'x_70', 
#                'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'x_77', 'x_78', 'x_79', 'x_80',
#                'x_81', 'x_82', 'x_83', 'x_84', 'x_85', 'x_86', 'x_87', 'x_88', 'x_89', 'x_90', 
#                'x_91', 'x_93', 'x_95','x_92' ]
#
#col_discrete = ['x_96', 'x_97', 'x_98', 'x_99', 'x_100', 'x_101', 'x_139', 'x_140', 'x_141', 'x_142',
#                'x_143', 'x_144', 'x_145', 'x_146', 'x_147', 'x_148', 'x_149', 'x_150', 'x_151', 'x_152',
#                'x_153', 'x_154', 'x_155', 'x_156', 'x_157','x_102', 'x_103', 'x_104', 'x_105', 'x_106', 
#                'x_107', 'x_108', 'x_109', 'x_110', 'x_111', 'x_112', 'x_113', 'x_114', 'x_115', 'x_116', 
#                'x_117', 'x_118', 'x_119', 'x_120', 'x_121', 'x_122', 'x_123', 'x_124', 'x_125', 'x_126', 
#                'x_127', 'x_128', 'x_129', 'x_130', 'x_131', 'x_132', 'x_133', 'x_134', 'x_135', 'x_136', 
#                'x_137', 'x_138']
#fal_dis=['x_28','x_94','x_23','x_34','x_12','x_18','x_6','x_90','x_91','x_3','x_5','x_82','x_9','x_92',
#         'x_85','x_49']#dfdf
##数据路径
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
valid_data=pd.read_csv(path_va,encoding='gbk')
train_data.drop(['cust_group','cust_id','y'],axis=1,inplace=True)
valid_data.drop(['cust_group','cust_id'],axis=1,inplace=True)
test_data.drop(['cust_group','cust_id'],axis=1,inplace=True)
#train_data=train_data.replace(-99, '')
a=train_data.count()
aa=train_data.describe()
ab=pd.DataFrame()
ac=pd.DataFrame()
ad=pd.DataFrame()
aaa=[]
c=[]
for x in train_data.columns:
    z=0
    b=train_data[x].values
    for y in b:
        if y==-99:
            z+=1
    z=z/15000
    if z not in c:
        c.append(z)
    aaa.append(z)
#ab['loss']=aaa
c=np.sort(c)
print(c)
ab['train_xy']=aaa
c=[]
aaa=[]
for x in valid_data.columns:
    z=0
    b=valid_data[x].values
    for y in b:
        if y==-99:
            z+=1
    z=z/10000
    if z not in c:
        c.append(z)
    aaa.append(z)
#ab['loss']=aaa
c=np.sort(c)
print(c)
ab['train_x']=aaa
c=[]
aaa=[]
for x in test_data.columns:
    z=0
    b=test_data[x].values
    for y in b:
        if y==-99:
            z+=1
    z=z/10000
    if z not in c:
        c.append(z)
    aaa.append(z)
#ab['loss']=aaa
c=np.sort(c)
print(c)
ab['test_all']=aaa
#d=[]
#for y in c:
#    w=0
#    for z in aaa:
#        if y==z:
#            w+=1
#    d.append(w)
##cc=aaa.unique()
#ab['b']=d
#ab.to_csv(r'C:\Users\Administrator\Desktop\queshi.csv',index=False)
    




#a1=train_data[train_data['cust_group']=='group_1']
#a2=train_data[train_data['cust_group']=='group_2']
#a3=train_data[train_data['cust_group']=='group_3']
#a=train_data['y'].ravel()
#a=np.sum(a)
#print(a)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
plt.figure(figsize=(50, 50)) 
ab.plot()

plt.savefig(r'C:\Users\Administrator\Desktop\2.png', dpi=200)
#tsne = TSNE(learning_rate=100, n_components=2, random_state=0, perplexity=20)
#tsne.embedding_=tsne.fit_transform(train_data)
# 
#tsne = pd.DataFrame(tsne.embedding_,index = train_data.index)#tsne.embedding_即降维后的二维数据
#plt.figure(figsize=(10, 10)) 
#d = tsne[train_data['y']==0]
#plt.plot(d[0],d[1],'r.')
#
#
#d = tsne[train_data['y']==1]
#plt.plot(d[0],d[1],'b*')
#plt.savefig(r'C:\Users\Administrator\Desktop\1.png', dpi=200)
##d = tsne[data['labels']==2]
##plt.plot(d[0],d[1],'b*')
#plt.show()
#from sklearn.manifold import TSNE
# 9 
#10 tsne=TSNE()
#11 tsne.fit_transform(data_zs)  #进行数据降维,降成两维
#12 #a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
#13 tsne=pd.DataFrame(tsne.embedding_,index=data_zs.index) #转换数据格式
#14 
#15 import matplotlib.pyplot as plt 
#16 
#17 d=tsne[r[u'聚类类别']==0]
#18 plt.plot(d[0],d[1],'r.')
#19 
#20 d=tsne[r[u'聚类类别']==1]
#21 plt.plot(d[0],d[1],'go')
#22 
#23 d=tsne[r[u'聚类类别']==2]
#24 plt.plot(d[0],d[1],'b*')
#25 
#26 plt.show()