from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import MODEL_FEATURES, PROCESSED_DATA, MODEL_PATH, METRICS_PATH, FEATURE_IMPORTANCE_PATH

CATEGORICAL=["schoolsup","famsup","internet","higher"]
NUMERIC=[c for c in MODEL_FEATURES if c not in CATEGORICAL]

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([( "imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("cat", Pipeline([( "imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])

def train(data_path=PROCESSED_DATA, model_path=MODEL_PATH):
    df=pd.read_csv(data_path)
    X=df[MODEL_FEATURES]
    y=df["target_score_pct"]
    y_cls=(y<50).astype(int)
    X_train,X_test,y_train,y_test,cy_train,cy_test=train_test_split(X,y,y_cls,test_size=.2,random_state=42,stratify=y_cls)

    reg=Pipeline([( "preprocess", make_preprocessor()), ("model", RandomForestRegressor(n_estimators=400,min_samples_leaf=3,random_state=42,n_jobs=-1))])
    reg.fit(X_train,y_train)
    pred=reg.predict(X_test)

    clf=Pipeline([( "preprocess", make_preprocessor()), ("model", LogisticRegression(max_iter=3000,class_weight="balanced"))])
    clf.fit(X_train,cy_train)
    cprob=clf.predict_proba(X_test)[:,1]
    cpred=(cprob>=.5).astype(int)

    metrics={
        "regression":{"mae":float(mean_absolute_error(y_test,pred)),"r2":float(r2_score(y_test,pred))},
        "below_credit_classifier":{"accuracy":float(accuracy_score(cy_test,cpred)),"precision":float(precision_score(cy_test,cpred,zero_division=0)),"recall":float(recall_score(cy_test,cpred,zero_division=0)),"f1":float(f1_score(cy_test,cpred,zero_division=0))},
    }
    if len(np.unique(cy_test))==2:
        metrics["below_credit_classifier"]["roc_auc"]=float(roc_auc_score(cy_test,cprob))

    # Cross-validation on the full dataset; primary regression metric is MAE.
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    cv_mae=-cross_val_score(reg,X,y,cv=cv,scoring="neg_mean_absolute_error").mean()
    metrics["regression"]["5fold_cv_mae"]=float(cv_mae)

    model_path.parent.mkdir(parents=True,exist_ok=True)
    bundle={"regressor":reg,"failure_classifier":clf,"features":MODEL_FEATURES,"model_version":"1.0.0"}
    joblib.dump(bundle,model_path)
    METRICS_PATH.parent.mkdir(parents=True,exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics,indent=2))

    # Permutation importance gives model-agnostic feature importance in original feature space.
    from sklearn.inspection import permutation_importance
    pi=permutation_importance(reg,X_test,y_test,n_repeats=10,random_state=42,scoring="neg_mean_absolute_error")
    fi=pd.DataFrame({"feature":X_test.columns,"importance_mean":pi.importances_mean,"importance_std":pi.importances_std}).sort_values("importance_mean",ascending=False)
    fi.to_csv(FEATURE_IMPORTANCE_PATH,index=False)
    return metrics

def load_bundle(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    return joblib.load(model_path)

def predict(student: dict, bundle=None):
    bundle=bundle or load_bundle()
    row=pd.DataFrame([student],columns=bundle["features"])
    score=float(bundle["regressor"].predict(row)[0])
    score=max(0,min(100,score))
    p=float(bundle["failure_classifier"].predict_proba(row)[0,1])
    return score,p
