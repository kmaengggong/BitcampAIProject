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
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ML
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# 1. 데이터 전처리
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

outliers = EllipticEnvelope(contamination=.2)
outliers.fit(outliers_data)
result = outliers.predict(outliers_data)

datasets.drop(np.where(result == -1)[0], axis=0, inplace=True)  # 행 삭제
datasets = datasets.reset_index(drop=True)

# 1.4 불필요 Feature 제거
x = datasets.drop(columns=['ID', 'Gender', 'Region_Code', 'Avg_Account_Balance', 'Is_Lead'])  # selectFromModel에서 나온 컴럼들로 구성
y = datasets.Is_Lead

# 1.5 train_test_split (DL)

# 1.5 KFold (ML)
n_splits = 8  # Best Value
random_state = 623
kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

# 1-6. 스케일링
scaler = StandardScaler()  # Best Value
x = scaler.fit_transform(x)

# 2. 모델 구성 - VotingClssifier(LGBM, XGB, CatBoot) 사용
lgbm_model = LGBMClassifier(num_leaves=116, reg_alpha=17, reg_lambda=54)
xgb_model = XGBClassifier(alpha=47, colsample_bylevel=0.636389058165331)
cat_model = CatBoostClassifier(learning_rate=0.0085912018661793, max_depth=7, random_strength=0, verbose=0)

model = VotingClassifier(
    estimators=[('lgbm', lgbm_model), ('xgb', xgb_model), ('cat', cat_model)],
    voting='hard',
    n_jobs=-1,
    verbose=0
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

# 6-3. Feature Importances
# fi_datasets = datasets.drop(columns=['Is_Lead'])
# n_features = fi_datasets.shape[1]
# 6-3-1.XGBClassifier
# plt.barh(range(n_features), xgb_model.feature_importances_, align="center")
# 6-3-2.LGBMClassifier
# plt.barh(range(n_features), lgbm_model.feature_importances_, align="center")
# 6-3-3.CatBoostClassifier
# plt.barh(range(n_features), cat_model.feature_importances_, align="center")

# plt.yticks(np.arange(n_features), fi_datasets.columns)
# plt.title("Credit Card Feature Importances")
# plt.ylabel("Feature")
# plt.xlabel("Importance")
# plt.ylim(-1, n_features)
# plt.show()


# 7. 결과
# 7-1. Deep Learning

# 7-2. Machine Learning
# ML - Votting
# LGBMClassifier: 0.8857936776211077
# XGBClassifier: 0.8859054474454434
# CatBoostClassifier: 0.8858800483758165
# votting acc: 0.885946093721178
# time: 83.06704425811768

# 7-3. 최종 결과
# ML - VottingClassifier(XGB, LGBM, Cat) / EllipticEnvelope(contamination=.2) / Drop['ID', 'Gender', 'Region_Code', 'Avg_Account_Balance', 'Is_Lead'] / StratifiedKfold(n_splits=8, random_state=623)
# votting acc: 0.8857936765886167
# time: 206.40080451965332