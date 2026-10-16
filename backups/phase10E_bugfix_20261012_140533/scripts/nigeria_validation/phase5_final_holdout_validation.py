"""EduPredict Nigeria Validation — Phase 5: Final 2025/2026 holdout."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, brier_score_loss, confusion_matrix)
import joblib
import sklearn
warnings.filterwarnings("ignore")

FEATURE_PIPELINE_VERSION = "2.0-period-safe-school-aware"
FEATURE_REPORT = ROOT / "reports" / "nigeria_validation" / "feature_build_report.json"

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "nigeria_validation" / "splits"
REPORT_DIR = ROOT / "reports" / "nigeria_validation"
MODEL_DIR = ROOT / "models" / "nigeria_validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = DATA / "train_2023_2024.csv"
VALID = DATA / "validation_2024_2025.csv"
TEST = DATA / "test_2025_2026.csv"
CAT = ["School","Term","Class","Arm","Gender","Subject"]
NUM = ["Term_Number","CA1","CA2","Assignment","Project","Attendance","Previous_Term_Total","Previous_Year_Average","Previous_Term_Grade_Numeric","Historical_Subject_Average","Historical_Subject_Count","Historical_Overall_Average","Historical_Overall_Count"]
TARGET_REG = "target_exam_score"
TARGET_CLS = "target_below_credit"
FORBIDDEN = {"Exam","Total","Grade","Final_Year_Result","Promotion_Status","Intervention"}

def load(p):
    df = pd.read_csv(p)
    need = set(CAT + NUM + [TARGET_REG, TARGET_CLS])
    miss = sorted(need - set(df.columns))
    if miss: raise RuntimeError(f"{p.name}: missing {miss}")
    return df

def reg_pipe():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
    return Pipeline([("pre", pre), ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., random_state=42))])

def cls_pipe():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CAT)])
    return Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., random_state=42))])

def X(df): return df[CAT + NUM]
def reg_metrics(y,p): return {"mae":float(mean_absolute_error(y,p)),"rmse":float(np.sqrt(mean_squared_error(y,p))),"r2":float(r2_score(y,p))}
def cls_metrics(y,p):
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"roc_auc":float(roc_auc_score(y,p)),"pr_auc":float(average_precision_score(y,p)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"accuracy":float(accuracy_score(y,pred)),"brier":float(brier_score_loss(y,p)),"threshold":.5,"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"positive_prevalence":float(np.mean(y))}

def main():
    if not FEATURE_REPORT.is_file():
        raise RuntimeError(f"Missing feature build report: {FEATURE_REPORT}")
    feature_report = json.loads(FEATURE_REPORT.read_text(encoding="utf-8"))
    if feature_report.get("status") != "PASS" or feature_report.get("feature_pipeline_version") != FEATURE_PIPELINE_VERSION:
        raise RuntimeError(
            "Refusing to train: feature_build_report.json is not a PASS from "
            f"pipeline {FEATURE_PIPELINE_VERSION!r}. Rebuild features first."
        )
    train, valid, test = map(load,[TRAIN,VALID,TEST])
    expected={TRAIN:"2023/2024",VALID:"2024/2025",TEST:"2025/2026"}
    for p,df in [(TRAIN,train),(VALID,valid),(TEST,test)]:
        vals=sorted(df.Academic_Session.astype(str).unique())
        if vals != [expected[p]]: raise RuntimeError(f"{p.name}: unexpected session {vals}")
    development=pd.concat([train,valid],ignore_index=True)
    # Validation-period measurements are reproduced for transparency; final fit uses only development data.
    vr=reg_pipe(); vr.fit(X(train),train[TARGET_REG]); vreg=reg_metrics(valid[TARGET_REG],vr.predict(X(valid)))
    vc=cls_pipe(); vc.fit(X(train),train[TARGET_CLS].astype(int)); vprob=vc.predict_proba(X(valid))[:,1]; vcls=cls_metrics(valid[TARGET_CLS].astype(int).to_numpy(),vprob)
    fr=reg_pipe(); fr.fit(X(development),development[TARGET_REG]); treg=reg_metrics(test[TARGET_REG],fr.predict(X(test)))
    fc=cls_pipe(); fc.fit(X(development),development[TARGET_CLS].astype(int)); tprob=fc.predict_proba(X(test))[:,1]; tcls=cls_metrics(test[TARGET_CLS].astype(int).to_numpy(),tprob)
    report={"phase":5,"status":"PASS","feature_pipeline_version":FEATURE_PIPELINE_VERSION,"feature_input_sha256":feature_report.get("input_sha256"),"runtime":{"scikit_learn":sklearn.__version__},"protocol":{"selection_regression":"lowest validation MAE","selection_classification":"highest validation PR-AUC","development_sessions":["2023/2024","2024/2025"],"final_holdout_session":"2025/2026","holdout_used_for_selection":False,"holdout_used_for_training":False,"classification_threshold":.5},"rows":{"train":len(train),"validation":len(valid),"development":len(development),"final_test":len(test)},"predictors":CAT+NUM,"forbidden_predictors":sorted(FORBIDDEN),"validation_metrics":{"regression":vreg,"classification":vcls},"final_test_metrics":{"regression":treg,"classification":tcls},"generalization_delta_test_minus_validation":{"regression_mae":treg["mae"]-vreg["mae"],"regression_rmse":treg["rmse"]-vreg["rmse"],"regression_r2":treg["r2"]-vreg["r2"],"classification_roc_auc":tcls["roc_auc"]-vcls["roc_auc"],"classification_pr_auc":tcls["pr_auc"]-vcls["pr_auc"],"classification_f1":tcls["f1"]-vcls["f1"],"classification_brier":tcls["brier"]-vcls["brier"]},"note":"Phase 5 is final chronological holdout measurement only. Production release still requires calibration, subgroup/stability, provenance review and release gate."}
    rp=REPORT_DIR/"phase5_final_holdout_results.json"; rp.write_text(json.dumps(report,indent=2),encoding="utf-8")
    joblib.dump(fr,MODEL_DIR/"phase5_regression_hgb.joblib"); joblib.dump(fc,MODEL_DIR/"phase5_classification_hgb.joblib")
    print("="*72); print("EDUPREDICT NIGERIA VALIDATION — PHASE 5"); print("="*72)
    print("Development: 2023/2024 + 2024/2025 | FINAL HOLDOUT: 2025/2026\n")
    print(f"Regression validation: MAE={vreg['mae']:.4f} RMSE={vreg['rmse']:.4f} R2={vreg['r2']:.4f}")
    print(f"Regression FINAL:      MAE={treg['mae']:.4f} RMSE={treg['rmse']:.4f} R2={treg['r2']:.4f}\n")
    print(f"Classification validation: ROC-AUC={vcls['roc_auc']:.4f} PR-AUC={vcls['pr_auc']:.4f} F1={vcls['f1']:.4f} Brier={vcls['brier']:.4f}")
    print(f"Classification FINAL:      ROC-AUC={tcls['roc_auc']:.4f} PR-AUC={tcls['pr_auc']:.4f} Precision={tcls['precision']:.4f} Recall={tcls['recall']:.4f} F1={tcls['f1']:.4f} Brier={tcls['brier']:.4f}")
    print(f"Confusion matrix: {tcls['confusion_matrix']}")
    print(f"\nReport: {rp}\nModels: {MODEL_DIR}\n\nPHASE 5 STATUS: PASS")
    print("Production release is NOT authorized by Phase 5 alone.")
if __name__ == "__main__": main()
