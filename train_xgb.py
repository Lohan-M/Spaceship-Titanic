from data_preprocessing import X, y, preprocessing1, preprocessing2, preprocessing3, preprocessing4
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import random
import pandas as pd

random.seed(42)

model = XGBClassifier(objective="binary:logistic", eval_metric="logloss")

param_grid = {
    'preprocessor': [preprocessing1, preprocessing2, preprocessing3, preprocessing4],
    'model__n_estimators': [400, 600],
    'model__max_depth': [2, 4],
    'model__learning_rate': [0.01, 0.02, 0.03],
    'model__subsample': [0.6, 0.8],
    'model__colsample_bytree': [0.6, 0.8],
    'model__min_child_weight': [1, 2, 3, 4, 5],
    'model__gamma': [0, 0.1, 0.2, 0.3],
    'model__reg_lambda': [0, 0.001, 0.01, 0.1, 1, 10, 100],
    'model__reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10, 100]
}

def random_params(param_grid):
    return {key: random.choice(val) for key, val in param_grid.items()}

final_pipeline = Pipeline(steps=[('preprocessor', preprocessing1), ('model', model)])

def train(params, X_train, y_train, X_val):
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss")
    pipeline = Pipeline(steps=[('preprocessor', preprocessing1), ('model', model)])
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)
    outputs = pipeline.predict_proba(X_val)[:, 1]
    return outputs

if __name__ == "__main__":
    """search = RandomizedSearchCV(final_pipeline, param_grid, n_iter=5, scoring="accuracy", verbose=1)
    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)

    top10 = results.sort_values("mean_test_score", ascending=False).head(10)"""

    """for _, row in top10.iterrows():
        print(f"Score: {row['mean_test_score']:.4f}")
        print(f"Params: {row['params']}")
        print()
    joblib.dump(search, "search.joblib")
    for param in ['model__n_estimators','model__max_depth','model__learning_rate','model__subsample',
                'model__colsample_bytree','model__min_child_weight','model__gamma','model__reg_lambda','model__reg_alpha']:
        print(f"\n=== {param} ===")
        print(
            results.groupby(f"param_{param}")["mean_test_score"]
                .agg(["mean", "std", "count"])
                .sort_values("mean", ascending=False)
        )"""

    """search = joblib.load("search.joblib")
    print("Best CV score:")
    print(search.best_score_)
    for param, value in search.best_params_.items():
        print(f"{param}: {value}")"""
