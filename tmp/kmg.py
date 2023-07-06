# 주제 : 신용카드 자격이 있는 사람과 없는 사람에 대한 데이터 분석하기
# 목표 : train.csv만 갖고서, accuracy 0.8이상 나오도록 할 것

'''
<전체 로직 순서>
[1] NaN값 처리하기
[2] Labeling(object값 제거하기(문자 -> 숫자화))
[3] 상관계수 히트맵(heatmap) 데이터 시각화하기(변수 간의 연관성 식별하여 체크하기)
[4] outliers 데이터 전처리하기(최솟값, 최댓값, 이상치 값 체크하기)
[5] feature importance 데이터 표시하기(중요 데이터 값 체크하기)
'''

import time, datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# DL
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import Accuracy
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder
from sklearn.covariance import EllipticEnvelope

# ML
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 1. 데이터
# data_path = "../data/credit_card_prediction/"  # 리눅스 환경
data_path = "./data/credit_card_prediction/"
datasets = pd.read_csv(data_path + "train.csv")

# 1-1. 데이터 확인
# Columns
# ['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active', 'Is_Lead']
# Target = Is_Lead, value=[0, 1]
# print(datasets.shape)  # (245725, 11)

# 1-2. 라벨링
string_columns = ["ID", "Gender", "Region_Code", "Occupation", "Channel_Code", "Credit_Product", "Is_Active"]
label_encoder = LabelEncoder()
for i in range(len(string_columns)):
    datasets[string_columns[i]] = label_encoder.fit_transform(datasets[string_columns[i]])

# 1-3. outliers - EllipticEnvelop 적용
outliers_data = np.array(datasets.Vintage)
outliers_data = outliers_data.reshape(-1, 1)

outliers = EllipticEnvelope(contamination=.37)  # .2 -> .37
outliers.fit(outliers_data)
result = outliers.predict(outliers_data)

# 1-3-1. 행 삭제
datasets.drop(np.where(result == -1)[0], axis=0, inplace=True)

# 1-4 NaN값 처리 - 최빈값으로 처리
x = datasets.drop(columns=['ID', 'Gender', 'Region_Code', 'Avg_Account_Balance', 'Is_Lead'])    # selectFromModel에서 나온 컴럼들로 구성
imputer = SimpleImputer(strategy="most_frequent")  # 최빈값
imputer.fit(x)
x = imputer.transform(x)

y = datasets.Is_Lead

# 1-4. 스케일링
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 1-5. 데이터 시각화 (Heatmap)
n_splits = 11
random_state = 72
kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)


# 2. 모델, 훈련
lgbm_model = LGBMClassifier(boost_from_average=True, n_estimators=64, num_leaves=16, n_jobs=-1)
xgb_model = XGBClassifier(eta=0.1, max_depth=6, subsample=1)
cat_model = CatBoostClassifier(verbose=0)

model = VotingClassifier(
    estimators=[('lgbb', lgbm_model), ('xgb', xgb_model), ('cat', cat_model)],
    voting='soft',
    n_jobs=-1,
    verbose=1
)

classifier = [lgbm_model, xgb_model, cat_model]
for model_c in classifier:
    model_c.fit(x, y)
    name = model_c.__class__.__name__
    acc = cross_val_score(model_c, x, y, cv=kfold)
    print(name, ' : ', round(np.mean(acc), 16))
    
model.fit(x, y)
acc = cross_val_score(model, x, y, cv=kfold)
print('votting acc :', round(np.mean(acc), 16))