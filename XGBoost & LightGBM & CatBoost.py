# Install: pip install xgboost & lightgbm & catboost

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc'
}

xgb_model = xgb.train(
    params, dtrain,
    num_boost_round=200,
    evals=[(dtest, 'eval')],
    early_stopping_rounds=20,
    verbose_eval=False
)
y_pred_xgb = (xgb_model.predict(dtest) > 0.5).astype(int)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

# XGBoost sklearn API
xgb_sk = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss'
)
xgb_sk.fit(X_train, y_train)
print("XGBoost Sklearn API:", xgb_sk.score(X_test, y_test))

# LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=200,
                      valid_sets=[lgb_test], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
y_pred_lgb = (lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) > 0.5).astype(int)
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))

# CatBoost
cb = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.01, verbose=False, random_state=42)
cb.fit(X_train, y_train)
print("CatBoost Accuracy:", cb.score(X_test, y_test))
