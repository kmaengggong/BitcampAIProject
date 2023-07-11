import numpy as np
import pandas as pd

from sklearn.covariance import EllipticEnvelope
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ML
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# 1. 데이터 전처리
# data_path = "../data/credit_card_prediction/"  # 리눅스 환경
data_path = "./data/credit_card_prediction/"
# data_path = "/home/ncp/workspace/tf_study/_data/credit_card_prediction/"
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

outliers = EllipticEnvelope(contamination=.2)  # ML Best Value
outliers.fit(outliers_data)
result = outliers.predict(outliers_data)

datasets.drop(np.where(result == -1)[0], axis=0, inplace=True)  # 행 삭제

# 1.4 불필요 Feature 제거
# x = datasets.drop(columns=['Is_Lead']) # 6, 7. feature Importances 관련 사항 확인시에는 컬럼 삭제 없이 진행
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

# optuna
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

def objectiveCAT(trial: Trial, x, y):
    # XGB
    # param = {
    #     'max_depth': trial.suggest_int('max_depth', 2, 15),
    #     'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05),
    #     'n_estimators': trial.suggest_int('n_estimators', 1000, 10000, 100),
    #     'eta': trial.suggest_discrete_uniform('eta', 0.01, 0.1, 0.01),
    #     'reg_alpha': trial.suggest_int('reg_alpha', 1, 50),
    #     'reg_lambda': trial.suggest_int('reg_lambda', 5, 100),
    #     'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
    #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
    # }


    # LGBMClassifier
    # param = {
    #     "objective": "binary",
    #     "metric": "binary_logloss",
    #     "verbosity": -1,
    #     "boosting_type": "gbdt",
    #     "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
    #     "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    #     "num_leaves": trial.suggest_int("num_leaves", 2, 256),
    #     "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
    #     "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
    #     "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
    #     "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
    #     'reg_alpha': trial.suggest_int('reg_alpha', 1, 50),
    #     'reg_lambda': trial.suggest_int('reg_lambda', 5, 100),
    # }

    # CatBoostClassifier
    param = {
        'learning_rate' : trial.suggest_uniform('learning_rate',0.01, 1),
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        "n_estimators":trial.suggest_int("n_estimators", 1, 1000),
        "max_depth":trial.suggest_int("max_depth", 1, 15),
        'random_strength' :trial.suggest_int('random_strength', 0, 100),
        "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        'subsample': trial.suggest_uniform('subsample',0,1),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5, 1e-3),
    }

    # 학습 모델 생성
    model = LGBMClassifier(**param)
    model.fit(x, y, verbose=True) # 학습 진행
    # 모델 성능 확인
    score = cross_val_score(model, x, y)
    return round(np.mean(score), 4)                                                                                                                                                      


# MAE가 최소가 되는 방향으로 학습을 진행
study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(lambda trial : objectiveCAT(trial, x, y), n_trials = 5)
print('Best trial : score {}, /nparams {}'.format(study.best_trial.value, study.best_trial.params))

# vscode에서 확인불가, jupyter에서 실행
# HyperParameter Importances - 하이퍼파라미터의 중요도를 시각화하여 출력
optuna.visualization.plot_param_importances(study)