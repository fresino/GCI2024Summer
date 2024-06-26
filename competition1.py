import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# データの読み込み
path = "/Users/hayase/Library/CloudStorage/OneDrive-筑波大学/松尾研/01.（公開）コンペ1/"
df = pd.read_csv(path + 'data/train.csv')
df_test = pd.read_csv(path + 'data/test.csv')

# データ前処理
age = pd.concat([df['Age'], df_test['Age']])
fare = pd.concat([df['Fare'], df_test['Fare']])

df['Age'].fillna(age.mean(), inplace=True)
df_test['Age'].fillna(age.mean(), inplace=True)

df['Fare'].fillna(fare.mean(), inplace=True)
df_test['Fare'].fillna(fare.mean(), inplace=True)

df.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)

df['Embarked'].fillna('S', inplace=True)
df_test['Embarked'].fillna('S', inplace=True)

df.drop(['Name', 'Ticket'], axis=1, inplace=True)
df_test.drop(['Name', 'Ticket'], axis=1, inplace=True)

df.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
df_test.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)

embarked = pd.concat([df['Embarked'], df_test['Embarked']])
embarked_ohe = pd.get_dummies(embarked)
embarked_ohe_train = embarked_ohe[:len(df)]
embarked_ohe_test = embarked_ohe[len(df):]

df = pd.concat([df, embarked_ohe_train], axis=1)
df_test = pd.concat([df_test, embarked_ohe_test], axis=1)

df.drop('Embarked', axis=1, inplace=True)
df_test.drop('Embarked', axis=1, inplace=True)

# 新しい特徴量の作成
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1

df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df_test['IsAlone'] = (df_test['FamilySize'] == 1).astype(int)

# スケーリング
scaler = StandardScaler()
X = df.drop(['PassengerId', 'Perished'], axis=1)
y = df['Perished']
X_test = df_test.drop('PassengerId', axis=1)

X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# テストデータを7:3に分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

# モデルの定義とハイパーパラメータチューニング
rfc = RandomForestClassifier(random_state=42)
lr = LogisticRegression(random_state=42)
mlpc = MLPClassifier(random_state=42)
gbc = GradientBoostingClassifier(random_state=42)

# パイプラインとグリッドサーチ
pipe_rfc = Pipeline([('rfc', rfc)])
param_rfc = {'rfc__n_estimators': [100, 200, 300],
             'rfc__max_depth': [7, 10, 15],
             'rfc__min_samples_leaf': [1, 2, 4]}

pipe_lr = Pipeline([('lr', lr)])
param_lr = {'lr__C': [0.1, 1, 10, 100]}

pipe_mlpc = Pipeline([('mlpc', mlpc)])
param_mlpc = {'mlpc__hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
              'mlpc__alpha': [0.0001, 0.001, 0.01]}

pipe_gbc = Pipeline([('gbc', gbc)])
param_gbc = {'gbc__n_estimators': [100, 200],
             'gbc__learning_rate': [0.01, 0.1],
             'gbc__max_depth': [3, 4, 5]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_rfc = GridSearchCV(pipe_rfc, param_rfc, cv=cv, scoring='accuracy', n_jobs=-1)
grid_lr = GridSearchCV(pipe_lr, param_lr, cv=cv, scoring='accuracy', n_jobs=-1)
grid_mlpc = GridSearchCV(pipe_mlpc, param_mlpc, cv=cv, scoring='accuracy', n_jobs=-1)
grid_gbc = GridSearchCV(pipe_gbc, param_gbc, cv=cv, scoring='accuracy', n_jobs=-1)

grid_rfc.fit(X_train, y_train)
grid_lr.fit(X_train, y_train)
grid_mlpc.fit(X_train, y_train)
grid_gbc.fit(X_train, y_train)

# ベストモデルの選択
best_rfc = grid_rfc.best_estimator_
best_lr = grid_lr.best_estimator_
best_mlpc = grid_mlpc.best_estimator_
best_gbc = grid_gbc.best_estimator_

# モデルの評価
models = [best_rfc, best_lr, best_mlpc, best_gbc]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    print(f'{model.steps[0][0]}:')
    print(f'  Accuracy: {accuracy_score(y_valid, y_pred):.3f}')
    print(f'  F1 Score: {f1_score(y_valid, y_pred):.3f}')
    print(f'  ROC AUC: {roc_auc_score(y_valid, y_pred):.3f}')

# アンサンブル
ensemble = VotingClassifier(estimators=[('rfc', best_rfc), ('lr', best_lr), ('mlpc', best_mlpc), ('gbc', best_gbc)], voting='soft')
ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_valid)
print(f'Ensemble:')
print(f'  Accuracy: {accuracy_score(y_valid, y_pred):.3f}')
print(f'  F1 Score: {f1_score(y_valid, y_pred):.3f}')
print(f'  ROC AUC: {roc_auc_score(y_valid, y_pred):.3f}')

# テストデータ予測
pred_proba = ensemble.predict_proba(X_test)[:, 1]
pred = (pred_proba > 0.5).astype(int)

# 読み込むデータが格納されたディレクトリのパス，必要に応じて変更の必要あり
path = '/Users/hayase/Library/CloudStorage/OneDrive-筑波大学/松尾研/01.（公開）コンペ1/'

submission = pd.read_csv(path + 'gender_submission.csv')

# 上書き
submission['Perished'] = pred
print(submission)

# CSV化
submission.to_csv('/Users/hayase/Library/CloudStorage/OneDrive-筑波大学/松尾研/01.（公開）コンペ1/submission.csv', index=False)

# Kaggle用
submission['Survived'] = -(pred) + 1  # Survivedを追加，omnicampus用と0,1が逆なので対応するように変換
submission.drop(['Perished'], axis=1, inplace=True)  # Perishedは不要なので削除
submission.to_csv('/Users/hayase/Library/CloudStorage/OneDrive-筑波大学/松尾研/01.（公開）コンペ1/submission_kaggle.csv', index=False)
