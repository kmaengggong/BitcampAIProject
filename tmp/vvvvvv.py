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
from sklearn.preprocessing import LabelEncoder, StandardScaler

# DL
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split


# 1. 데이터 전처리
# data_path = "../data/credit_card_prediction/"  # 리눅스 환경
data_path = "./data/credit_card_prediction/"
datasets = pd.read_csv(data_path + "train.csv")

# 1-1. 데이터 확인

# 1-2. 라벨링
string_columns = ["ID", "Gender", "Region_Code", "Occupation", "Channel_Code", "Credit_Product", "Is_Active"]
label_encoder = LabelEncoder()
for i in range(len(string_columns)):
    datasets[string_columns[i]] = label_encoder.fit_transform(datasets[string_columns[i]])

# 1.3 Outliers 처리 - EllipticEnvelop 적용
print("Before:", datasets.shape)
outliers_data = np.array(datasets.Vintage)
outliers_data = outliers_data.reshape(-1, 1)
outliers = EllipticEnvelope(contamination=.2)  # ML Best Value
outliers.fit(outliers_data)
result = outliers.predict(outliers_data)
datasets.drop(np.where(result == -1)[0], axis=0, inplace=True)  # 행 삭제
outliers_columns = ["Occupation", "Channel_Code"]

for outlier_column in datasets.columns:#outliers_columns:
    q1 = datasets[outlier_column].quantile(0.25)
    q3 = datasets[outlier_column].quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = 1.5 * iqr
    outliers = datasets[(datasets[outlier_column] < q1 - outlier_threshold) | (datasets[outlier_column] > q3 + outlier_threshold)]
    # datasets = datasets.drop(outliers.index)
    plt.boxplot(datasets[outlier_column])
    outliers_indices = outliers.index.tolist()
    plt.scatter(outliers_indices, outliers[outlier_column], color='red')
    plt.title(outlier_column)
    plt.legend()
    plt.show()
    # outliers_low = datasets[(datasets['Vintage'] < q1 - outlier_threshold)]
    # outliers_high = datasets[(datasets['Vintage'] > q3 + outlier_threshold)]

# print("Before:", datasets['Vintage'].min(), datasets['Vintage'].max())
# datasets.loc[outliers_low.index, 'Vintage'] = datasets['Vintage'].min()
# datasets.loc[outliers_high.index, 'Vintage'] = datasets['Vintage'].max()
# print("After:", datasets['Vintage'].min(), datasets['Vintage'].max())

# plt.boxplot(datasets['Vintage'])
# outliers_indices = outliers.index.tolist()
# plt.scatter(outliers_indices, outliers['Vintage'], color='red')
# plt.legend()
# plt.show()
# outliers_columns = datasets.drop(columns=["ID", "Is_Lead"]).columns#["Vintage"]#, "Age"]
# contaminations = [.2]#[.2]#, .1]

# for i in range(len(outliers_columns)):
#     outliers_data = np.array(datasets[outliers_columns[i]])
#     outliers_data = outliers_data.reshape(-1, 1)

#     outliers = EllipticEnvelope(contamination=contaminations[i])  # DL Best Value
#     outliers.fit(outliers_data)
#     result = outliers.predict(outliers_data)

#     drop_indices = np.where(result == -1)[0]
#     datasets.drop(datasets.index[drop_indices], inplace=True)

# fig, axes = plt.subplots(nrows=1, ncols=len(outliers_columns), figsize=(12, 6))

# datasets = datasets.drop(datasets[datasets["Credit_Product"] == 2].index)
# print(datasets[datasets["Credit_Product"] == 2])
# for i, column in enumerate(outliers_columns):
#     # axes[i].hist(datasets[column], bins=10)
#     sns.kdeplot(data=datasets[column], fill=True, ax=axes[i])
#     axes[i].set_xlabel(column)
#     axes[i].set_ylabel('Value')

# plt.tight_layout()
# plt.show()

print("After:", datasets.shape)

# 1.4 불필요 Feature 제거
# x = datasets.drop(columns=["ID", "Is_Lead"])
x = datasets.drop(columns=['ID', 'Gender', 'Region_Code', 'Is_Lead'])
#x = datasets.drop(columns=['ID', 'Gender', 'Region_Code', 'Is_Lead', 'Avg_Account_Balance'])  # selectFromModel에서 나온 컴럼들로 구성
y = datasets.Is_Lead

# 1.5 NaN값 처리 - 최빈값으로 처리
# imputer = SimpleImputer(strategy="most_frequent")  # 최빈값
# imputer.fit(x)
# x = imputer.transform(x)

# 1.6 train_test_split (DL)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.68, test_size=0.12, random_state=7211, shuffle=True)

# 1.6 KFold (ML)

# 1-7. 스케일링
scaler = StandardScaler()  # Best Value
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
model = Sequential()
# model.add(Dense(128, input_dim=9, activation="relu"))
model.add(Dense(128, input_dim=7, activation="relu"))
model.add(Dense(64))
model.add(Dense(32, activation="relu"))
model.add(Dense(64))
model.add(Dense(1, activation="sigmoid"))


# 3. 컴파일 (DL)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    restore_best_weights=True
)


# 4. 훈련
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stopping])
end_time = time.time()


# 5. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss:", loss)
print("acc:", acc)
print("time:", end_time-start_time)


# 6. 데이터 시각화
# 6-1. Heatmap
# 6-2. Outliers
# 6-3. Feature Importances


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