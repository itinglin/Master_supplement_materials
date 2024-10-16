from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from prediction import predict
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression



df = pd.read_csv('model_training.csv')
# selecting features and target data
X = df.iloc[:, :9]
y = df.iloc[:, 9]

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X, y)


clf = RandomForestClassifier(max_depth = 3, n_estimators=30)
clf.fit(X_train, y_train)
joblib.dump(clf,'rf_model.sav')