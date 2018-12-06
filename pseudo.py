from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
from data_processing import train,test,valid
import heapq
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
import xgboost as xgb
skf = StratifiedKFold(n_splits=5, random_state=21, shuffle=True)
class PseudoLabeler(BaseEstimator, RegressorMixin):
    
    def __init__(self, model, valid,test, features, target, sample_rate=0.81, seed=42):
        self.sample_rate = sample_rate#取值范围为0.0-1.0
        self.seed = seed
        self.model = model
        self.model.seed = seed
        self.valid=valid
        self.test = test
        self.features = features
        self.target = target
        
    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "test": self.test,
            'valid':self.valid,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        
    def fit(self, X, y):
        if self.sample_rate > 0.0:
            baseloss=[]
            t=10
            loss=0
            numone=int(self.sample_rate*10000)
            te,augemented_train= self.__create_augmented_train(X, y)
            ltest=te['y'].reshape((numone,1))
            ldata=te.drop('y',axis=1).values
            tr_y=augemented_train['y']
            y=tr_y.reshape((15000,1))
            tr_x=augemented_train.drop('y',axis=1)
            x=tr_x.values
#            param=
#             'XGBC:one': {
#        'n_estimators': XGBC_TREES,
#        'scale_pos_weight': 6,
#        'objective': 'binary:logistic',
#        'learning_rate': 0.02,
#        
#        'reg_lambda': 800,
#        'colsample_bytree': 0.75,
#        'max_depth': 5,
#        'min_child_weight': 4,
#        'subsample': 0.75,
#        },
#            model1= xgb.XGBClassifier(boosting_type='gbdt', num_leaves=40, max_depth=5, learning_rate=0.01, n_estimators=5000,
#                           max_bin=425, gamma= 0.7, objective='binary:logistic', min_split_gain=0,
#                           min_child_weight=5, min_child_samples=10, subsample=0.8,  seed=1000, n_jobs=10, silent=True)
            skf = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
            for i, (train_index, test_index) in enumerate(skf.split(tr_x, tr_y)):
                print(x[train_index].shape,ldata.shape)       
                train=np.vstack((x[train_index],ldata))
                print(y[train_index].shape,ltest.shape)
                lable=np.vstack((y[train_index],ltest))
                val_x=x[test_index]#验证集数据全都来自test_all
                val_y=y[test_index]
            
                print("Fold", i)
                slef.model.fit(train,lable,
                          
                          eval_metric='auc',
                          eval_set=[(train, lable), 
                                    (val_x, val_y)],early_stopping_rounds=100)
                baseloss.append(self.model.best_score_['valid']['auc'])
                loss =self.model.best_score_['valid']['auc']
#            train_p=self.model.predict_proba(train, num_iteration=self.model.best_iteration_)[:, 1]
#            test_pred= self.model.predict_proba(test, num_iteration=self.model.best_iteration_)[:, 1]
#    #predict_result['pred_prob']=predict_result['pred_prob']+train_p
                if loss<t:
                    t=loss
                    train_p=self.model.predict(x, num_iteration=self.model.best_iteration_)
                    test_pred= np.abs(self.model.predict(self.test, num_iteration=self.model.best_iteration_))
        #predict_result['pred_prob']  = train_p
#            
#            print('test mean:', test_pred.mean())
#            sampled_test = augmented_test.sample(n=num_of_samples)
#            
#            self.model.fit(
#                augemented_train[self.features],
#                augemented_train[self.target]
#            )
#        else:
#            self.model.fit(X, y)
        
        return test_pred,baseloss,loss
#    def __create_augmented_train(self, X, y):
#        numone=500#pseudo-lable中的最大的numone个数据为1，与train_xy同分布
#        num_of_samples = int(len(test) * self.sample_rate)+int(len(valid) * self.sample_rate)
#        #从valid和test数据中分别选出sample_rate比例的数据加入训练+int(len(valid) * self.sample_rate)
#        skf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
#        baseloss = []
#        loss = 0
#        t=10
#        train=X.values
#        lable=y.values
#        
#        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#            
#            print("Fold", i)
#            self. model.fit(train[train_index], lable[train_index],
#                          eval_names =['train','valid'],
#                          eval_metric='auc',
#                          eval_set=[(train[train_index], lable[train_index]), 
#                                    (train[test_index], lable[test_index])],early_stopping_rounds=100)
#            baseloss.append(self.model.best_score_['valid']['auc'])
#            loss = self.model.best_score_['valid']['auc']
##            train_p=self.model.predict_proba(train, num_iteration=self.model.best_iteration_)[:, 1]
##            test_pred= self.model.predict_proba(test, num_iteration=self.model.best_iteration_)[:, 1]
#    #predict_result['pred_prob']=predict_result['pred_prob']+train_p
#            if loss<t:
#                t=loss
#                train_p=self.model.predict_proba(train, num_iteration=self.model.best_iteration_)[:, 1]
#                test_pred= self.model.predict_proba(self.test, num_iteration=self.model.best_iteration_)[:, 1]
#                valid_pred= self.model.predict_proba(self.valid, num_iteration=self.model.best_iteration_)[:, 1]
#        #predict_result['pred_prob']  = train_p
#            
#            print('test mean:', test_pred.mean())
##            else:
##                pass
#         #Train the model and creat the pseudo-labels
#         
#         #lable=np.vstack((y[train_index],ltest))
##        self.model.fit(X, y)
#        b=heapq.nlargest(693, train_p)
#        c=0.163
#        
##        pseudo_labels = valid_pred
#        pseudo_labels = np.vstack((test_pred,valid_pred)).reshape((20000,1))
#        #print(pseudo_lables)
#        a=heapq.nlargest(numone, pseudo_labels)
#        #np.savetxt(r'C:\Users\Administrator\Desktop\da\re.csv',pseudo_labels ,delimiter=',')
#        #pseudo_labels.save_txt(r'C:\Users\Administrator\Desktop\da\re.csv')
#        #self.model.predict(self.test[self.features])
##        threshold = 0.4 
#        for i in range(len(pseudo_labels)):  
#            pseudo_labels[i] = 1 if  pseudo_labels[i]>= c else 0  
#        np.savetxt(r'C:\Users\Administrator\Desktop\da\re.csv',pseudo_labels ,delimiter=',')
#        #print(pseudo_labels,'adadasds')
##
#        a=np.sum(pseudo_labels)
#        # Add the pseudo-labels to the test set
#        augmented_test = pd.concat([test.copy(deep=True),valid.copy(deep=True)],axis=0)
#       # augmented_test =test.copy(deep=True)
#        augmented_test[self.target] = pseudo_labels
#        
#        # Take a subset of the test set with pseudo-labels and append in onto
#        # the training set
#        sampled_test = augmented_test.sample(n=num_of_samples)
#        temp_train = pd.concat([X, y], axis=1)
#        #augemented_train = pd.concat([sampled_test, temp_train])
#        
#        return sampled_test ,temp_train,c,a

    def __create_augmented_train(self, X, y):
        numone=461#pseudo-lable中的最大的numone个数据为1，与train_xy同分布
        num_of_samples = int(len(test) * self.sample_rate)
        #从valid和test数据中分别选出sample_rate比例的数据加入训练+int(len(valid) * self.sample_rate)
        skf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
        baseloss = []
        loss = 0
        t=10
        train=X.values
        lable=y.values
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            
            print("Fold", i)
            self. model.fit(train[train_index], lable[train_index],
                          eval_names =['train','valid'],
                          eval_metric='auc',
                          eval_set=[(train[train_index], lable[train_index]), 
                                    (train[test_index], lable[test_index])],early_stopping_rounds=100)
            baseloss.append(self.model.best_score_['valid']['auc'])
            loss = self.model.best_score_['valid']['auc']
#            train_p=self.model.predict_proba(train, num_iteration=self.model.best_iteration_)[:, 1]
#            test_pred= self.model.predict_proba(test, num_iteration=self.model.best_iteration_)[:, 1]
    #predict_result['pred_prob']=predict_result['pred_prob']+train_p
            if loss<t:
                t=loss
                train_p=self.model.predict_proba(train, num_iteration=self.model.best_iteration_)[:, 1]
                test_pred= self.model.predict_proba(test, num_iteration=self.model.best_iteration_)[:, 1]
                valid_pred= self.model.predict_proba(valid, num_iteration=self.model.best_iteration_)[:, 1]
        #predict_result['pred_prob']  = train_p
            
            print('test mean:', test_pred.mean())
#            else:
#                pass
         #Train the model and creat the pseudo-labels
         
         #lable=np.vstack((y[train_index],ltest))
#        self.model.fit(X, y)
        pseudo_labels=test_pred
       # pseudo_labels = np.vstack((test_pred,valid_pred)).reshape((20000,1))
        #print(pseudo_lables)
        a=heapq.nlargest(numone, pseudo_labels)
        #np.savetxt(r'C:\Users\Administrator\Desktop\da\re.csv',pseudo_labels ,delimiter=',')
        #pseudo_labels.save_txt(r'C:\Users\Administrator\Desktop\da\re.csv')
        #self.model.predict(self.test[self.features])
#        threshold = 0.4 
        for i in range(len(pseudo_labels)):  
            pseudo_labels[i] = 1 if  pseudo_labels[i] in a else 0  
        np.savetxt(r'C:\Users\Administrator\Desktop\da\re.csv',pseudo_labels ,delimiter=',')
        #print(pseudo_labels,'adadasds')
#        
        # Add the pseudo-labels to the test set
        augmented_test = test
       # augmented_test = pd.concat([test.copy(deep=True),valid.copy(deep=True)],axis=0)
        augmented_test[self.target] = pseudo_labels
        
        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_test = augmented_test.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        #augemented_train = pd.concat([sampled_test, temp_train])
        
        return sampled_test ,shuffle(temp_train)
#        
    def predict(self, X):
        #self.model.predict(X).to_csv(r'C:\Users\Administrator\Desktop\da\re.csv'  , index=False)
        return self.fit(X,y)
    
    def get_model_name(self):
        return self.model.__class__.__name__
predict_result=pd.DataFrame()

predict_result['cust_id']=test['cust_id'].values

target = 'y'
test.drop(['y','cust_id','cust_group'],axis=1,inplace=True)
valid.drop(['y','cust_id','cust_group'],axis=1,inplace=True)
features=test.columns
# Preprocess the data
X_train, X_test,X_valid = train[features], test[features],valid[features]
y_train = train[target]

# Create the PseudoLabeler with XGBRegressor as the base regressor
model = PseudoLabeler(
    lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=-1, learning_rate=0.01, n_estimators=5000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.7, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True),
    
                       
    X_valid,
    X_test,
    features,
    target
)
from sklearn.metrics import roc_auc_score
# Train the model and use it to predict
result,baseloss,loss=model.fit(X_train, y_train)
#result即为test_all的预测标签
#a=model.predict(X_test)
predict_result['pred_prob']  = result
print('Fold_5:', baseloss, '\navg_auc:',np.mean(baseloss))
predict_result[['cust_id', 'pred_prob']].to_csv(r'C:\Users\Administrator\Desktop\da\sub1.csv'  , index=False)