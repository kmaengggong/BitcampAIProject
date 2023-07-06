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

# 1-2. 라벨링
string_columns = ["ID", "Gender", "Region_Code", "Occupation", "Channel_Code", "Credit_Product", "Is_Active"]
label_encoder = LabelEncoder()
for i in range(len(string_columns)):
    datasets[string_columns[i]] = label_encoder.fit_transform(datasets[string_columns[i]])

# 1.3 Outliers 처리 - EllipticEnvelop 적용
outliers_data = np.array(datasets.Vintage)
outliers_data = outliers_data.reshape(-1, 1)

outliers = EllipticEnvelope(contamination=.37)  # ML Best Value
outliers.fit(outliers_data)
result = outliers.predict(outliers_data)

datasets.drop(np.where(result == -1)[0], axis=0, inplace=True)  # 행 삭제

# 1.4 불필요 Feature 제거
x = datasets.drop(columns=['ID', 'Gender', 'Region_Code', 'Avg_Account_Balance', 'Is_Lead'])  # selectFromModel에서 나온 컴럼들로 구성
y = datasets.Is_Lead

# 1.5 NaN값 처리 - 최빈값으로 처리
imputer = SimpleImputer(strategy="most_frequent")  # 최빈값
imputer.fit(x)
x = imputer.transform(x)

# 1.6 train_test_split (DL)

# 1.6 KFold (ML)
n_splits = 8  # Best Value
random_state = 88
kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

# 1-7. 스케일링
scaler = StandardScaler()  # Best Value
x = scaler.fit_transform(x)


# 2. 모델 구성 - VotingClssifier(LGBM, XGB, CatBoot) 사용
lgbm_model = LGBMClassifier(boost_from_average=True, n_estimators=64, num_leaves=16, n_jobs=-1)
xgb_model = XGBClassifier(eta=0.1, max_depth=6, subsample=1)
cat_model = CatBoostClassifier(verbose=0)

model = VotingClassifier(
    estimators=[('lgbb', lgbm_model), ('xgb', xgb_model), ('cat', cat_model)],
    voting='soft',
    n_jobs=-1,
    verbose=1
)


# 3. 컴파일 (DL)


# 4. 훈련
classifier = [lgbm_model, xgb_model, cat_model]
for model_c in classifier:
    model_c.fit(x, y)
    name = model_c.__class__.__name__
    acc = cross_val_score(model_c, x, y, cv=kfold)
    print(name + ":", round(np.mean(acc), 16))

start_time = time.time()
model.fit(x, y)
end_time = time.time()


# 5. 평가, 예측
acc = cross_val_score(model, x, y, cv=kfold)
print("votting acc:", round(np.mean(acc), 16))
print("time:", end_time-start_time)


# 6. 데이터 시각화
# 6-1. Heatmap
# 6-2. Outliers
# 6-3. Feature Importances


# 7. 결과
# 7-2. Machine Learning
# ML - / DecisionTreeClassfier
# acc: 0.71
# time: 0.8604931831359863

# ML - Votting
# LGBMClassifier  :  0.8856119493979576
# XGBClassifier  :  0.8853833257125439
# CatBoostClassifier  :  0.8850276888685668
# votting acc : 0.8856373520296703*

# ML - Votting / EllipticEnvelope(contamination=.37) / KFold(n_splits=8, random_state=72)
# votting acc: 0.9016545624086946
# time: 22.835328578948975