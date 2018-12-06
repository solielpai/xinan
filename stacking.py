# coding=UTF-8
from __future__ import division
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, linear_model
from sklearn.metrics import roc_auc_score
import pandas as pd
from data_processing import train,test
N_TREES = 500
XGBC_TREES = 1500
LM_CV_NUM = 100
MAX_ITERS = 400

INITIAL_PARAMS = {
    'XGBC:one': {
        'n_estimators': XGBC_TREES,
        'scale_pos_weight': 6,
        'objective': 'binary:logistic',
        'learning_rate': 0.02,
        'gamma': 0.7,
        'reg_lambda': 800,
        'colsample_bytree': 0.75,
        'max_depth': 5,
        'min_child_weight': 4,
        'subsample': 0.8,
        },
    
    'ETC:one': {
        'n_estimators': N_TREES, 'n_jobs': -1, 'min_samples_leaf': 2,  'criterion': 'entropy',
        'max_depth': 20, 'min_samples_split': 5, 'max_features': 0.4,
        'bootstrap': False,
        },
    'GBC:one': {
        'n_estimators': int(N_TREES / 2), 'learning_rate': .08, 'max_features': 0.5,
        'min_samples_leaf': 1, 'min_samples_split': 3, 'max_depth': 5,
        },
    'lgb:one':{
            'boosting_type':'gbdt', 'num_leaves':31, 'max_depth':-1, 'learning_rate':0.01, 
               'n_estimators':5000,'max_bin':425, 'subsample_for_bin':50000, 'objective':'binary', 
               'min_split_gain':0,'min_child_weight':5, 'min_child_samples':10,
               'subsample':0.8, 'subsample_freq':1,'colsample_bytree':1, 'reg_alpha':3,
               'reg_lambda':5, 'seed':1000,' n_jobs':10, 'silent':'True'}
    }

import xgboost as xgb

MODEL_NAME = 'blend_ensemble'


def modell(X_org, y_org, test_x):
    n_folds = 5
    verbose = True
    shuffle = False

    X = X_org
    y = y_org
    X_submission = test_x
    #X_submission = X_org

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
    skf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
    a=list(skf.split(X, y))
    
#    skf = StratifiedKFold(y=y, n_folds=n_folds)

    clfs = [
        #RandomForestClassifier().set_params(**INITIAL_PARAMS.get("RFC:one", {})),
        ExtraTreesClassifier().set_params(**INITIAL_PARAMS.get("ETC:one", {})),
        GradientBoostingClassifier().set_params(**INITIAL_PARAMS.get("GBC:one", {})),
        #LogisticRegression().set_params(**INITIAL_PARAMS.get("LR:one", {})),
        xgb.XGBClassifier().set_params(**INITIAL_PARAMS.get("XGBC:two", {})),
        #xgb.XGBClassifier().set_params(**INITIAL_PARAMS.get("XGBC:one", {})),
        lgb.LGBMClassifier().set_params(**INITIAL_PARAMS.get("LGB:one", {}))
        ]

    print ("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print (j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(a)))
        for i, (train, test) in enumerate(a):
            print ("Fold", i)
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print ("Blending.")
    clf = LogisticRegression(C=2, penalty='l2', class_weight='balanced', n_jobs=-1)
#    clf = linear_model.RidgeCV(
#            alphas=np.linspace(0, 200), cv=LM_CV_NUM)
    #clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=100)
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict(dataset_blend_test)
    x_submission = clf.predict(dataset_blend_train)
#    final_model = LinearRegression()
#final_model.fit(stacked_train, y_train)
#test_prediction = final_model.predict(stacked_test)

    print ("Linear stretch of predictions to [0,1]")
       
    print ("blend result")
    #save_submission.to_csv(r'C:\Users\Administrator\Desktop\da\su.csv', index=False)
    return y_submission,dataset_blend_train,dataset_blend_test,x_submission


if __name__ == '__main__':
    
    np.random.seed(0)  # seed to shuffle the train set

    result=pd.DataFrame()
    
    y_org=train['y'].values
    X_org=train.drop(['y','cust_id'],axis=1).values
    result['cust_id']=test['cust_id']
    test_x=test.drop(['y','cust_id'],axis=1).values
    result['pred-prob']=0
   


    y,a,b,d=modell(X_org, y_org, test_x)
    result['pred-prob']=y
    result.to_csv(r'C:\Users\Administrator\Desktop\da\cd.csv',index=False)
    a = np.mean(a, axis=1, keepdims=True)
    b=np.mean(b, axis=1, keepdims=True)
    c=roc_auc_score(y_org,a)
    e=roc_auc_score(y_org,d)
    print(c)
 

