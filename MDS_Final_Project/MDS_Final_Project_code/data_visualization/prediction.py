import joblib
def predict(data):
    clf = joblib.load('rf_model.sav')
    return clf.predict(data)

def predict_xgb(data):
    clf = joblib.load('xgb_model.sav')
    return clf.predict(data)

def predict_lr(data):
    clf = joblib.load('lr_model.sav')
    return clf.predict(data)