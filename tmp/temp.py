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

# 1.3 NaN값 처리 - 최빈값으로 처리
# datasets = datasets.fillna(datasets["Credit_Product"].value_counts().idxmax())
x = datasets.drop(columns=["Is_Lead", "ID"])

# x = datasets.drop(columns=["Is_Lead", "ID", "Gender"])
# loss: 0.3437422811985016
# acc: 0.8611659407615662
# time: 131.74617338180542
# x = datasets.drop(columns=["Is_Lead", "ID", "Occupation"])
# loss: 0.3578159511089325
# acc: 0.8533116579055786
# time: 129.39727115631104
# x = datasets.drop(columns=["Is_Lead", "ID", "Channel_Code"])
# loss: 0.3449936509132385
# acc: 0.8609217405319214
# time: 51.05740666389465
# x = datasets.drop(columns=["Is_Lead", "ID", "Is_Active"])
# loss: 0.34696170687675476
# acc: 0.861084520816803
# time: 134.08618688583374
# x = datasets.drop(columns=["Is_Lead", "ID", "Gender", "Occupation"])
# loss: 0.35880422592163086
# acc: 0.853270947933197
# time: 45.12754011154175
# x = datasets.drop(columns=["Is_Lead", "ID", "Gender", "Channel_Code"])
# loss: 0.34473952651023865
# acc: 0.8620408773422241
# time: 78.28723955154419
# x = datasets.drop(columns=["Is_Lead", "ID", "Gender", "Is_Active"])
# loss: 0.3478316068649292
# acc: 0.8603113293647766
# time: 31.685577392578125
# x = datasets.drop(columns=["Is_Lead", "ID", "Occupation", "Channel_Code"])
# loss: 0.3579028844833374
# acc: 0.8540645241737366
# time: 61.75207996368408
# x = datasets.drop(columns=["Is_Lead", "ID", "Occupation", "Is_Active"])
# loss: 0.3602629601955414
# acc: 0.8522128462791443
# time: 126.29113626480103
# x = datasets.drop(columns=["Is_Lead", "ID", "Channel_Code", "Is_Active"])
# loss: 0.3477625250816345
# acc: 0.8612473011016846
# time: 63.73282337188721
# x = datasets.drop(columns=["Is_Lead", "ID", "Gender", "Occupation", "Channel_Code"])
# loss: 0.35943078994750977
# acc: 0.8531692028045654
# time: 66.96075963973999
# x = datasets.drop(columns=["Is_Lead", "ID", "Gender", "Occupation", "Is_Active"])
# loss: 0.36129942536354065
# acc: 0.8513582348823547
# time: 90.90782642364502
# x = datasets.drop(columns=["Is_Lead", "ID", "Gender", "Channel_Code", "Is_Active"])
# x = datasets.drop(columns=["Is_Lead", "ID", "Occupation", "Channel_Code", "Is_Active"])
x = datasets.drop(columns=["Is_Lead", "ID", "Gender", "Occupation", "Channel_Code", "Is_Active"])

# imputer = SimpleImputer()  # 평균값
# imputer = SimpleImputer(strategy="mean")  # default
# imputer = SimpleImputer(strategy="median")  # 중간값
imputer = SimpleImputer(strategy="most_frequent")  # 최빈값
imputer.fit(x)
x = imputer.transform(x)

y = datasets.Is_Lead
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.2, random_state=7211, shuffle=True)

# 1-4. 스케일링
scaler = StandardScaler()  # yet best
# scaler = RobustScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 1-5. 데이터 시각화 (Heatmap)
# 1-6. Outlier

# 2. 모델 구성
# DL
model = Sequential()
model.add(Dense(128, input_dim=5, activation="relu"))
model.add(Dense(64))
model.add(Dense(32, activation="relu"))
model.add(Dense(64))
model.add(Dense(1, activation="sigmoid"))

# 3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=128, validation_split=0.2, callbacks=[early_stopping])
end_time = time.time()

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss:", loss)
print("acc:", acc)
print("time:", end_time-start_time)

# 4-1. 데이터 시각화 (Feature Importances, ML)


# 5. 결과