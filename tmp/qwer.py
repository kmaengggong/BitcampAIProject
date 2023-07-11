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
from sklearn.svm import SVC


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
outliers_data = np.array(datasets.Vintage)
outliers_data = outliers_data.reshape(-1, 1)

outliers = EllipticEnvelope(contamination=.34)  # DL Best Value
outliers.fit(outliers_data)
result = outliers.predict(outliers_data)

datasets.drop(np.where(result == -1)[0], axis=0, inplace=True)  # 행 삭제

# 1.4 불필요 Feature 제거
x = datasets.drop(columns=['Is_Lead'], axis=1)  # selectFromModel에서 나온 컴럼들로 구성
y = datasets.Is_Lead

# 1.5 NaN값 처리 - 최빈값으로 처리
imputer = SimpleImputer(strategy="most_frequent")  # 최빈값
imputer.fit(x)
x = imputer.transform(x)

# 1.6 train_test_split (DL)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.68, test_size=0.12, random_state=7211, shuffle=True)

# 1.6 KFold (ML)

# 1-7. 스케일링
scaler = StandardScaler()  # Best Value
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=10, activation="relu"))
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
# start_time = time.time()
# model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stopping])
# end_time = time.time()


# 6-3. Feature Importances
temp_x = datasets.drop(columns=["Is_Lead"])
model = SVC()
model.fit(temp_x, y)
n_features = temp_x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align="center")
plt.yticks(np.arange(n_features), temp_x.columns)
plt.title("Credit Card Feature Importances")
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.ylim(-1, n_features)
plt.show()