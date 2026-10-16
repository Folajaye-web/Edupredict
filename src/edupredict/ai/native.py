"""Native EduPredict pre-exam feature and model pipeline.

Target = actual examination score. Features are restricted to evidence available
before the current examination: CA components, attendance, engagement, profile,
and prior-period performance. Current exam/final-result fields are never used.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sqlalchemy.orm import Session
from ..db import Result, ContinuousAssessment, Attendance, Engagement, StudentProfile, AcademicSession
from ..config import NATIVE_MODEL_PATH, NATIVE_METRICS_PATH

NUMERIC_FEATURES = [
    'ca1','ca2','assignment','project','current_ca_average',
    'previous_subject_average','attendance_pct','days_absent',
    'homework_rate','participation','behaviour','study_hours_week',
    'age','previous_failures','study_time','family_relationship',
    'free_time','social_outings','self_reported_health','travel_time',
]
BINARY_FEATURES=['school_support','family_support','internet_access','higher_education_intent']
FEATURES=NUMERIC_FEATURES+BINARY_FEATURES

def _num(v, default=0.0):
    try:
        if v is None or pd.isna(v): return default
        return float(v)
    except Exception: return default

def _bool(v):
    return 1.0 if str(v).strip().lower() in {'yes','y','true','1','m'} else 0.0

def _profile(s, student_id):
    p=s.query(StudentProfile).filter_by(student_id=student_id).first()
    if not p: return {}
    try: return json.loads(p.profile_json or '{}')
    except Exception: return {}

def build_native_features(s: Session, *, session_id:int, student_id:int, subject_id:int) -> dict:
    ca=s.query(ContinuousAssessment).filter_by(session_id=session_id,student_id=student_id,subject_id=subject_id).first()
    att=s.query(Attendance).filter_by(session_id=session_id,student_id=student_id).first()
    eng=s.query(Engagement).filter_by(session_id=session_id,student_id=student_id).first()
    profile=_profile(s,student_id)
    vals=[getattr(ca,k,None) for k in ('ca1','ca2','assignment','project')] if ca else []
    vals=[_num(x,None) for x in vals if x is not None]
    current_ca=float(np.mean(vals)) if vals else None
    prior=(s.query(Result).filter(Result.student_id==student_id,Result.subject_id==subject_id,Result.session_id!=session_id).all())
    prior_scores=[_num(x.total,None) for x in prior if x.total is not None]
    previous=float(np.mean(prior_scores)) if prior_scores else None
    attendance=(100.0*att.days_present/att.school_days) if att and att.school_days else None
    age=_num(profile.get('age'),0)
    previous_failures=_num(profile.get('previous_failures'),0)
    return {
        'ca1': _num(getattr(ca,'ca1',None),0), 'ca2':_num(getattr(ca,'ca2',None),0),
        'assignment':_num(getattr(ca,'assignment',None),0), 'project':_num(getattr(ca,'project',None),0),
        'current_ca_average': _num(current_ca,0), 'previous_subject_average':_num(previous,current_ca or 0),
        'attendance_pct':_num(attendance,0), 'days_absent':_num(getattr(att,'days_absent',None),_num(profile.get('absence_count'),0)),
        'homework_rate':_num(getattr(eng,'homework_rate',None),0), 'participation':_num(getattr(eng,'participation',None),0),
        'behaviour':_num(getattr(eng,'behaviour',None),0), 'study_hours_week':_num(getattr(eng,'study_hours_week',None),0),
        'age':age, 'previous_failures':previous_failures, 'study_time':_num(profile.get('study_time'),2),
        'family_relationship':_num(profile.get('family_relationship'),3), 'free_time':_num(profile.get('free_time'),3),
        'social_outings':_num(profile.get('social_outings'),3), 'self_reported_health':_num(profile.get('self_reported_health'),3),
        'travel_time':_num(profile.get('travel_time'),2), 'school_support':_bool(profile.get('school_support')),
        'family_support':_bool(profile.get('family_support')), 'internet_access':_bool(profile.get('internet_access')),
        'higher_education_intent':_bool(profile.get('higher_education_intent')),
    }

def build_training_frame(s: Session, school_id:int) -> pd.DataFrame:
    rows=[]
    q=(s.query(Result, AcademicSession)
       .join(AcademicSession, Result.session_id==AcademicSession.id)
       .filter(AcademicSession.school_id==school_id, Result.exam.isnot(None)))
    for r,period in q.all():
        f=build_native_features(s,session_id=r.session_id,student_id=r.student_id,subject_id=r.subject_id)
        f['target_exam_score']=float(r.exam)
        f['session_id']=r.session_id; f['student_id']=r.student_id; f['subject_id']=r.subject_id
        rows.append(f)
    return pd.DataFrame(rows)

def train_native(s: Session, school_id:int, model_path=NATIVE_MODEL_PATH):
    df=build_training_frame(s,school_id)
    if len(df)<20: raise ValueError(f'At least 20 historical examination records are required; found {len(df)}.')
    X=df[FEATURES]; y=df.target_exam_score; cls=(y<50).astype(int)
    Xtr,Xte,ytr,yte,ctr,cte=train_test_split(X,y,cls,test_size=.2,random_state=42,stratify=cls if cls.nunique()>1 else None)
    prep=Pipeline([('imputer',SimpleImputer(strategy='median'))])
    reg=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestRegressor(n_estimators=400,min_samples_leaf=3,random_state=42,n_jobs=-1))])
    reg.fit(Xtr,ytr); pred=reg.predict(Xte)
    clf=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1))])
    clf.fit(Xtr,ctr); prob=clf.predict_proba(Xte)[:,1] if len(clf.classes_)==2 else np.full(len(Xte),float(clf.classes_[0]))
    cp=(prob>=.5).astype(int)
    metrics={'records':len(df),'regression':{'mae':float(mean_absolute_error(yte,pred)),'r2':float(r2_score(yte,pred))},
             'below_credit_classifier':{'accuracy':float(accuracy_score(cte,cp)),'f1':float(f1_score(cte,cp,zero_division=0))}}
    if len(np.unique(cte))==2: metrics['below_credit_classifier']['roc_auc']=float(roc_auc_score(cte,prob))
    model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'regressor':reg,'failure_classifier':clf,'features':FEATURES,'model_version':'native-1.0.0'},model_path)
    NATIVE_METRICS_PATH.write_text(json.dumps(metrics,indent=2))
    return metrics

def native_predict(features:dict, model_path=NATIVE_MODEL_PATH):
    bundle=joblib.load(model_path)
    row=pd.DataFrame([[features.get(k) for k in bundle['features']]],columns=bundle['features'])
    score=float(bundle['regressor'].predict(row)[0]); score=max(0,min(100,score))
    prob=float(bundle['failure_classifier'].predict_proba(row)[0,1]) if len(bundle['failure_classifier'].classes_)==2 else 0.0
    return score,prob,bundle['model_version']
