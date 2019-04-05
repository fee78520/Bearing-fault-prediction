# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import copy
import os
import gc
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import mode
from sklearn import svm
import lightgbm as lgb
scaler = StandardScaler()



train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test_data.csv')


def creat_fea(data):
    train = data

    return train

def get_result(train_data, test_data, label, mymodel, split_nums=3):
    sub = test_data[['id']]
    train = train_data
    test = test_data

    oof = np.zeros(len(label))

    train = train.values
    test = test.values

    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)

    train = abs(np.fft.fft(train))
    test = abs(np.fft.fft(test))

    label = label.values

    mean_re = []

    k_fold = KFold(n_splits=split_nums, shuffle=True, random_state=2019)
    for index, (train_index, test_index) in enumerate(k_fold.split(train)):
        print(index)
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = label[train_index], label[test_index]

        model = mymodel(X_train, y_train, X_test, y_test)

        X_test_pre = model.predict(X_test)
        oof[test_index] =X_test_pre

        score = accuracy_score(y_test, X_test_pre)
        mean_re.append(score)

        pred_result = model.predict(test)

        sub['label'] = pred_result

        if index == 0:
            re_sub = copy.deepcopy(sub)
        else:
            re_sub['label'] = re_sub['label'] + sub['label']

    re_sub['label'] = re_sub['label'] / split_nums

    print('score list:', mean_re)
    return re_sub, oof

label = train[['label']]
train = creat_fea(train).drop(['label'], axis=1)
test = creat_fea(test)

def svm_model(X_train, y_train, X_test, y_test):
    model = svm.SVC(C=1.0, kernel='linear', decision_function_shape='ovr')
    model.fit(X_train, y_train)
    return model

def lgb_multi(X_train, y_train, X_test, y_test):
    lgb_model = lgb.LGBMClassifier(
        boosting_type="gbdt", num_leaves=240, reg_alpha=0, reg_lambda=0.,
        max_depth=-1, n_estimators=800, objective='multiclass', class_weight='balanced',
        subsample=0.9, colsample_bytree=0.5, subsample_freq=1,
        learning_rate=0.03, random_state=2018, n_jobs=-1, metric="None", importance_type='gain', verbose=1
    )
    lgb_model.fit(X_train, y_train)
    return  lgb_model


sub, oof = get_result(train, test, label, mymodel=svm_model, split_nums=3)
sub['label'] = sub['label'].apply(lambda x:int(x+0.5))
sub.to_csv('../result/sub.csv', index=False)



