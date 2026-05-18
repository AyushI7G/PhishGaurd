"""Basic ensemble wrapper for XGBoost, RandomForest, and LightGBM models."""
import os
import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost as xgb

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def _make_estimators():
    estimators = [
        ('xgb', xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, use_label_encoder=False, eval_metric='logloss')),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
    if LGBMClassifier is not None:
        estimators.append(('lgbm', LGBMClassifier(n_estimators=100)))
    return estimators


class EnsembleClassifier:
    def __init__(self, voting='soft', version='v1.1-Ensemble'):
        self.voting = voting
        self.version = version
        self.model = VotingClassifier(estimators=_make_estimators(), voting=voting, flatten_transform=True)
        self.scaler = None
        self.last_votes = {}

    def fit(self, X, y):
        self.scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        prob = self.model.predict_proba(X)
        if hasattr(self.model, 'estimators_'):
            votes = {}
            for name, estimator in zip(self.model.estimators, self.model.estimators_):
                try:
                    # just save the prediction of the first item for UI
                    p = float(estimator.predict_proba(X[0:1])[0][1])
                    votes[name[0]] = round(p, 3)
                except Exception:
                    pass
            self.last_votes = votes
        return prob

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path, scaler_path=None):
        joblib.dump(self, path)
        if scaler_path and self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
