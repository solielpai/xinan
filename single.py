# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:10:14 2018

@author: Solielpai
"""
from data_processing import train,test
from xgboost.sklearn import XGBRegressor
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

from sklearn.preprocessing import Imputer
predict_result=pd.DataFrame()
predict_result['cust_id']=test['cust_id'].values
id_cols = ['cust_id','y']
#,'rx_75','dx_72','x_101_1','x_100_0','x_100_2',
#           'x_101_0','x_99_0','x_98_3','x_98_2','x_98_1','x_98_0','x_100_1','x_141_6','x_139_0',
#           'x_139_2','x_146_2','x_146_1','x_146_0','x_145_2','x_145_1','x_145_0','x_144_3',
#           'x_144_2','x_144_1','x_144_0','x_143_2','x_143_1','x_143_0','x_142_1','x_92x_63',
#           'x_92x_54','x_92x_52','x_92x_55','x_92x_62','x_92n7','x_92n3','x_142_0','x_92n9',
#           'x_97_0','x_141_5','x_141_4','x_141_3','x_141_2','x_140_2','x_97_3','rx_11','x_96_1',
#           'x_125','x_119','x_120','x_121','x_122','x_123','x_124','x_126','x_117','x_127','x_128',
#           'x_129','x_130','x_131','x_132','x_118','x_116','x_134','x_108','x_102','x_103','x_104',
#           'x_105','x_106','x_107','x_109','x_115','x_11','x_110','x_111'
#           rx_75','dx_72','x_101_1','x_100_0','x_100_2',
#'x_101_0','x_99_0','x_98_3','x_98_2','x_98_1','x_98_0','x_100_1','x_141_6','x_139_0',
#'x_139_2','x_146_2','x_146_1','x_146_0','x_145_2','x_145_1','x_145_0','x_144_3',
#'x_144_2','x_144_1','x_144_0','x_143_2','x_143_1','x_143_0','x_142_1','x_92x_63',
#'x_92x_54','x_92x_52','x_92x_55','x_92x_62','x_92n7','x_92n3','x_142_0','x_92n9',
#'x_97_0','x_141_5','x_141_4','x_141_3','x_141_2','x_140_2','x_97_3','rx_11','x_96_1',
#'x_125','x_119','x_120','x_121','x_122','x_123','x_124','x_126','x_117','x_127','x_128',
#'x_129','x_130','x_131','x_132','x_118','x_116','x_134','x_108','x_102','x_103','x_104',
#'x_105','x_106','x_107','x_109','x_115','x_11','x_110','x_111',
#'x_112','x_113','x_114','x_133','x_135','x_98','x_74','x_33','x_37','x_147_1','x_55',
#'x_70','x_71','x_75','x_3','x_85','x_86','x_89','x_9','x_92','x_94','x_31','x_27',
#'x_136','x_148','x_137','x_138','x_143','x_144','x_145','x_146','x_149','x_25','x_15',
#'x_150','x_152','x_17','x_21','x_22','x_147_0','x_154_3','x_147_2','dx_7','dx_17',
#'dx_15','dx_14','dx_11','dx_10','dx_9','dx_8','dx_5','dx_48','dx_4','dx_3','dx_2',
#'dx_1','dx_94','rx_3','rx_90','dx_21','dx_22','dx_24','dx_25','dx_46','dx_42','dx_41',
#'dx_40','dx_39','dx_38','dx_37','dx_36','dx_35','dx_33','dx_32','dx_31','dx_29','dx_27',
#'dx_26','rx_89','rx_88','rx_87','rx_50','rx_48','rx_47','rx_37','rx_33','rx_32','rx_31',
#'rx_9','rx_29','rx_27','rx_26','rx_25','rx_10','rx_22','rx_21','rx_17','rx_49','rx_51',
#'rx_86','rx_52','rx_85','rx_77','rx_76','x_10','rx_74','rx_73','rx_71','rx_70','rx_66',
#'rx_65','rx_63','rx_62','rx_5','rx_55','rx_54','dx_47','dx_49','x_147_3','x_155_3',
#'x_153_0','x_153_2','x_153_3','x_154_0','rx_15','x_155_0','x_155_2','x_156_1','dx_51',
#'x_156_3','x_157_1','x_157_5','x_157_6','rx_94','n7','n6','x_152_2','x_152_1','x_152_0',
#'x_151_2','x_147_4','x_148_0','x_148_1','x_148_2','x_148_3','x_149_0','x_149_1',
#'x_149_2','x_149_3','x_150_0','x_150_1','x_150_2','x_150_3','x_151_0','x_151_1',
#'n2','rx_1','dx_90','dx_69','dx_67','dx_66','dx_65','dx_64','dx_63','dx_62','dx_61',
#'dx_60','dx_59','dx_58','dx_56','dx_55','dx_54','dx_53','dx_52','dx_68','dx_70',
#'dx_89','dx_71','dx_88','dx_87','dx_86','dx_85','dx_83','dx_82','dx_81','dx_79',
#'dx_78','dx_77','dx_76','dx_75','dx_74','dx_73','dx_72','rx_75'
tr_x = train.drop(id_cols,axis =1)
aaaa=tr_x.corr()
tr_y = train['y']

train =tr_x.values
lable = tr_y.values
#train=preprocessing.scale(train,axis=0) 
 
 
print(test.columns.tolist())

test=test.drop(id_cols,axis =1).values
#test=preprocessing.scale(test,axis=0)
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.01, n_estimators=5000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)

t=100

skf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
baseloss = []
loss = 0
predict_result['pred_prob']=0
for i, (train_index, test_index) in enumerate(skf.split(tr_x, tr_y)):
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
        predict_result['pred_prob']  = test_pred
        
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

#a=predict_result['pred_prob'].values
#print(roc_auc_score(lable, a))


#predict_result['pred_prob']=predict_result['pred_prob']/5 
print('Fold_5:', baseloss, '\navg_auc:',loss/5)
predict_result[['cust_id', 'pred_prob']].to_csv(r'C:\Users\Administrator\Desktop\da\sub1.csv'  , index=False)