# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:13:51 2018

@author: Solielpai
"""
from data_processing import train,test
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
#test=data_xy[data_xy['y']==-1]
#
result=pd.DataFrame()#最终的测试集的预测概率df
predict_result=pd.DataFrame()#训练集的预测概率
predict_result['cust_id']=train['cust_id'].values
a1=pd.DataFrame()
a2=pd.DataFrame()
a3=pd.DataFrame()
b1=pd.DataFrame()
b2=pd.DataFrame()
b3=pd.DataFrame()
lab=train['y'].values
id_cols = ['cust_id','y','cust_group']
l1=train[train['cust_group']==1]
l1=l1['y']
l2=train[train['cust_group']==2]['y'].copy()
l3=train[train['cust_group']==3]['y'].copy()
t1=train[train['cust_group']==1].copy()
t2=train[train['cust_group']==2].copy()
t3=train[train['cust_group']==3].copy()
te1=test[test['cust_group']==1].copy()
te2=test[test['cust_group']==2].copy()
te3=test[test['cust_group']==3].copy()
b1['cust_id']=te1['cust_id']
b2['cust_id']=te2['cust_id']
b3['cust_id']=te3['cust_id']
a1['cust_id']=t1['cust_id']
a2['cust_id']=t2['cust_id']
a3['cust_id']=t3['cust_id']
train =t1.drop(id_cols,axis =1).values
lable = l1.values
#train=preprocessing.scale(train,axis=0) 
test=te1.drop(id_cols,axis =1).values

#print(test.columns.tolist())
#
#test=test.drop(id_cols,axis =1).values
test=preprocessing.scale(test,axis=0)
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.01, n_estimators=5000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)

t=100

skf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
baseloss = []
loss = 0
a1['pred_prob']=0
for i, (train_index, test_index) in enumerate(skf.split(train, lable)):
    print("Fold", i)
    lgb_model = model.fit(train[train_index], lable[train_index],
                          eval_names =['train','valid'],
                          eval_metric='auc',
                          eval_set=[(train[train_index], lable[train_index]), 
                                    (train[test_index], lable[test_index])],early_stopping_rounds=100)
    baseloss.append(lgb_model.best_score_['valid']['auc'])
    loss += lgb_model.best_score_['valid']['auc']
#    train_p=lgb_model.predict_proba(train, num_iteration=lgb_model.best_iteration_)[:, 1]
#    test_pred= lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
#    predict_result['pred_prob']=predict_result['pred_prob']+train_p
    if loss<t:
        t=loss
        train_p=lgb_model.predict_proba(train, num_iteration=lgb_model.best_iteration_)[:, 1]
        test_pred= lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
        a1['pred_prob']  = train_p
        b1['pred_prob']  = test_pred
        print('test mean:', test_pred.mean())
    else:
        pass
train =t2.drop(id_cols,axis =1).values
lable = l2.values
#train=preprocessing.scale(train,axis=0) 
test=te2.drop(id_cols,axis =1).values
baseloss = []
loss = 0
a2['pred_prob']=0
for i, (train_index, test_index) in enumerate(skf.split(train, lable)):
    print("Fold", i)
    lgb_model = model.fit(train[train_index], lable[train_index],
                          eval_names =['train','valid'],
                          eval_metric='auc',
                          eval_set=[(train[train_index], lable[train_index]), 
                                    (train[test_index], lable[test_index])],early_stopping_rounds=100)
    baseloss.append(lgb_model.best_score_['valid']['auc'])
    loss += lgb_model.best_score_['valid']['auc']
#    train_p=lgb_model.predict_proba(train, num_iteration=lgb_model.best_iteration_)[:, 1]
#    test_pred= lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
#    predict_result['pred_prob']=predict_result['pred_prob']+train_p
    if loss<t:
        t=loss
        train_p=lgb_model.predict_proba(train, num_iteration=lgb_model.best_iteration_)[:, 1]
        test_pred= lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
        a2['pred_prob']  = train_p
        b2['pred_prob']  = test_pred
        print('test mean:', test_pred.mean())
    else:
        pass
#predict_result['pred_prob']=0
#loss=pd.Series(losses)
#lo=loss.rank(method='max')
#for i in lo:
#    i=5-i.astype(int)
#    print(i)
#    a=w[i]
#    predict_result['pred_prob']=predict_result['pred_prob']+predict_result['pred_prob'+str(i)]*a
#
train =t3.drop(id_cols,axis =1).values
lable = l3.values
#train=preprocessing.scale(train,axis=0) 
test=te3.drop(id_cols,axis =1).values
baseloss = []
loss = 0
a3['pred_prob']=0
for i, (train_index, test_index) in enumerate(skf.split(train, lable)):
    print("Fold", i)
    lgb_model = model.fit(train[train_index], lable[train_index],
                          eval_names =['train','valid'],
                          eval_metric='auc',
                          eval_set=[(train[train_index], lable[train_index]), 
                                    (train[test_index], lable[test_index])],early_stopping_rounds=100)
    baseloss.append(lgb_model.best_score_['valid']['auc'])
    loss += lgb_model.best_score_['valid']['auc']
#    train_p=lgb_model.predict_proba(train, num_iteration=lgb_model.best_iteration_)[:, 1]
#    test_pred= lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
#    predict_result['pred_prob']=predict_result['pred_prob']+train_p
    if loss<t:
        t=loss
        train_p=lgb_model.predict_proba(train, num_iteration=lgb_model.best_iteration_)[:, 1]
        test_pred= lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
        a3['pred_prob']  = train_p
        b3['pred_prob']  = test_pred
        print('test mean:', test_pred.mean())
    else:
        pass
predict_result=pd.concat([a1,a2],axis=0,ignore_index=0)
predict_result=pd.concat([predict_result,a3],axis=0,ignore_index=0)
result=pd.concat([b1,b2],axis=0,ignore_index=0)
result=pd.concat([result,b3],axis=0,ignore_index=0)
#result为test_all的cust_id及标签
a=predict_result['pred_prob'].values
print(roc_auc_score(lab, a))


#predict_result['pred_prob']=predict_result['pred_prob']/5 
print('Fold_5:', baseloss, '\navg_auc:',loss/5)
result[['cust_id', 'pred_prob']].to_csv(r'C:\Users\Administrator\Desktop\da\cup\sub1.csv'  , index=False)