'''
주제 : 신용카드 자격이 있는 사람과 없는 사람에 대한 데이터 분석하기
목표 : train.csv만 갖고서, accuracy 0.85 이상 나오도록 할 것
'''

import datetime, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

# DL
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# ML
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# 1. 데이터 전처리
# data_path = "../data/credit_card_prediction/"  # 리눅스 환경
data_path = "./data/credit_card_prediction/"
datasets = pd.read_csv(data_path + "train.csv")

# 1-1. 데이터 확인
# Columns = ['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active', 'Is_Lead']
# Target = 'Is_Lead'  # value = [0, 1]
# print(datasets.shape)  # (245725, 11)

# 1-2. 라벨링
string_columns = ["ID", "Gender", "Region_Code", "Occupation", "Channel_Code", "Credit_Product", "Is_Active"]
label_encoder = LabelEncoder()
for i in range(len(string_columns)):
    datasets[string_columns[i]] = label_encoder.fit_transform(datasets[string_columns[i]])

# 1.3 Outliers 처리 - EllipticEnvelop 적용
outliers_data = np.array(datasets.Vintage)
outliers_data = outliers_data.reshape(-1, 1)

outliers = EllipticEnvelope(contamination=.34)  # DL Best Value
outliers = EllipticEnvelope(contamination=.37)  # ML Best Value
outliers.fit(outliers_data)
result = outliers.predict(outliers_data)

datasets.drop(np.where(result == -1)[0], axis=0, inplace=True)  # 행 삭제

# 1.4 불필요 Feature 제거
# x = datasets.drop(columns=["Is_Lead"])  # Drop x
x = datasets.drop(columns=['ID', 'Gender', 'Region_Code', 'Avg_Account_Balance', 'Is_Lead'])  # selectFromModel에서 나온 컴럼들로 구성
y = datasets.Is_Lead

# 1.5 NaN값 처리 - 최빈값으로 처리
# imputer = SimpleImputer()  # 평균값
# imputer = SimpleImputer(strategy="mean")  # default
# imputer = SimpleImputer(strategy="median")  # 중간값
imputer = SimpleImputer(strategy="most_frequent")  # 최빈값
imputer.fit(x)
x = imputer.transform(x)

# 1.6 train_test_split (DL)
# train_size = 0.6
# test_size = 0.2
# tts_random_state = 7211
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, test_size=test_size, random_state=tts_random_state, shuffle=True)

# 1.6 KFold (ML)
# n_splits = 7
# kf_random_state = 72
# kfold = StratifiedKFold(n_splits=n_splits, random_state=kf_random_state, shuffle=True)

# 1-7. 스케일링
scaler = StandardScaler()  # Best Value
# scaler = RobustScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()

# DL
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# ML
# x = scaler.fit_transform(x)


# 2. 모델 구성
# DL
# model = Sequential()
# model.add(Dense(128, input_dim=9, activation="relu"))
# model.add(Dense(1, activation="sigmoid"))

# ML
# model = SVC()
# model = RandomForestClassifier()
# model = DecisionTreeClassifier()


# 3. 컴파일 (DL)
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# early_stopping = EarlyStopping(
#     monitor="val_loss",
#     mode="min",
#     patience=10,
#     restore_best_weights=True
# )


# 4. 훈련
# DL
# start_time = time.time()
# model.fit(x_train, y_train, epochs=500, batch_size=128, validation_split=0.2, callbacks=[early_stopping])
# end_time = time.time()

# ML
# start_time = time.time()
# model.fit(x, y)
# end_time = time.time()


# 5. 평가, 예측
# DL
# loss, acc = model.evaluate(x_test, y_test)
# print("loss:", loss)
# print("acc:", acc)
# print("time:", end_time-start_time)

# ML
# acc = cross_val_score(model, x, y, cv=kfold)
# print("acc:", round(np.mean(acc), 4))
# print("time:", end_time-start_time)


# 6. 데이터 시각화
# 6-1. Heatmap
# sns.set(font_scale=1.2)
# sns.set(rc = {"figure.figsize":(12, 9)})
# sns.heatmap(data=datasets.corr(), square=True, annot=True, cbar=True)
# plt.show()

# 6-2. Outliers
# def outliers(data_out):
#     q1, q2, q3 = np.percentile(data_out, [25, 50, 75])
#     print('1사분위 : ', q1)
#     print('2사분위 : ', q2)
#     print('3사분위 : ', q3)
    
#     iqr =  q3 - q1
#     print(iqr)

#     lower_bound = q1 - (iqr * 1.5)
#     upper_bound = q3 + (iqr * 1.5)
#     print('lower_bound : ', lower_bound)
#     print('upper_bound : ', upper_bound)

#     return np.where((data_out > upper_bound) | (data_out < lower_bound))

# outlers_loc = outliers(datasets)
# print('이상치의 위치 : ', outlers_loc)
# plt.boxplot(outlers_loc)
# plt.show()

# 6-3. Feature Importances
# temp_x = datasets.drop(columns=["Is_Lead"])
# model = DecisionTreeClassifier()
# model.fit(temp_x, y)
# n_features = temp_x.shape[1]
# plt.barh(range(n_features), model.feature_importances_, align="center")
# plt.yticks(np.arange(n_features), temp_x.columns)
# plt.title("Credit Card Feature Importances")
# plt.ylabel("Feature")
# plt.xlabel("Importance")
# plt.ylim(-1, n_features)
# plt.show()


# 7. 결과
# 7-1. Deep Learning
# DL - / NaN(최빈값) / Drop["Id"] / train_test_split.random_state=7211 / StandardScaler / Model: 128(relu) 8 16 32 64 128 256 512(relu) 256 64 32 15 8 1 / epochs=100, batch_size=128 / EarlyStopping=10
# loss: 0.342851459980011
# acc: 0.8620002269744873
# time: 337.30227875709534

# DL - / EllipticEnvelope(contamination=.34) / Drop['ID', 'Gender', 'Region_Code', 'Avg_Account_Balance', 'Is_Lead'] / train_test_split(train_size=0.68, test_size=0.12, random_state=7211) / Dense: 128(relu) 64 32(relu) 64 1 / model.fit(epochs=100)
# loss: 0.27720940113067627
# acc: 0.9039530754089355
# time: 67.09893560409546

# DL - / EllipticEnvelope(contamination=.34) / Drop['ID', 'Gender', 'Region_Code', 'Avg_Account_Balance', 'Is_Lead'] / train_test_split(train_size=0.68, test_size=0.12, random_state=2) / Dense: 128(relu) 64 32(relu) 64 1 / model.fit(epochs=100)
# loss: 0.27680519223213196
# acc: 0.9050339460372925*
# time: 67.05005598068237

# 7-2. Machine Learning
# ML - / DecisionTreeClassfier
# acc: 0.71
# time: 0.8604931831359863

# ML - Votting
# LGBMClassifier  :  0.8856119493979576
# XGBClassifier  :  0.8853833257125439
# CatBoostClassifier  :  0.8850276888685668
# votting acc : 0.8856373520296703*