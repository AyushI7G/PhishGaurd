import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from feature_engineering import extract_features

data = pd.read_csv("data/urls.csv")

X = data['url'].apply(extract_features).tolist()
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier(n_estimators=200, max_depth=6)
model.fit(X_train, y_train)

joblib.dump(model, "models/url_xgb.pkl")