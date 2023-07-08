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
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ML
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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


# 1-3. Heatmat & Outliers 처리 - EllipticEnvelop 적용
# 1-3-1. Heatmap
# sns.set(font_scale=1.2)
# sns.set(rc={'figure.figsize' : (9, 6)}) # 가로,세로 사이즈 세팅
# sns.heatmap(data=datasets.corr(), square=True, annot=True, cbar=True)
# plt.show()

# 1-3-2. Outliers
outliers_data = np.array(datasets.Vintage)
outliers_data = outliers_data.reshape(-1, 1)

# plt.boxplot(outliers_data)
# plt.show()

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
random_state = 623
kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

# 1-7. 스케일링
scaler = StandardScaler()  # Best Value
x = scaler.fit_transform(x)

# 2. 모델 구성 - VotingClssifier(LGBM, XGB, CatBoot) 사용
lgbm_model = LGBMClassifier(num_leaves=70, reg_alpha=28, feature_fraction=0.5983661429456115)
xgb_model = XGBClassifier(max_depth=9, subsample=0.8, n_estimators=2700, eta=0.060000000000000005, reg_alpha=24, reg_lambda=73, min_child_weight=4, colsample_bytree=0.5511772671161806)
cat_model = CatBoostClassifier(learning_rate=0.9475161771345497, bagging_temperature=75.84084011451753, n_estimators=288, max_depth=3, random_strength=1, colsample_bylevel=0.9105476377311172, l2_leaf_reg=2.3355249344481578e-05, min_child_samples=80, subsample=0.6296258880786431, leaf_estimation_iterations=4, verbose=0)

model = VotingClassifier(
    estimators=[('lgbm', lgbm_model), ('xgb', xgb_model), ('cat', cat_model)],
    voting='soft',
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

# optuna 모두 적용
# LGBMClassifier: 0.9014498956876806
# XGBClassifier: 0.9016033954830142
# CatBoostClassifier: 0.9009638244557125
# votting acc: 0.9016161792789648
# time: 40.226014852523804