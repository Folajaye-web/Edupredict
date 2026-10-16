from pathlib import Path
import pandas as pd
from .config import RAW_DATA, PROCESSED_DATA
from .grading import score20_to_pct, grade_from_score

REQUIRED = ["G1","G2","G3","absences","studytime","failures","schoolsup","famsup","internet","famrel","freetime","goout","health","traveltime","higher"]

def build_training_data(input_path: Path = RAW_DATA, output_path: Path = PROCESSED_DATA) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep=";")
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out=df.copy()
    for c in ["G1","G2","G3","absences","studytime","failures","famrel","freetime","goout","health","traveltime"]:
        out[c]=pd.to_numeric(out[c], errors="coerce")
    for c in ["schoolsup","famsup","internet","higher"]:
        out[c]=out[c].astype(str).str.lower().str.strip()
    out["G1_pct"]=out["G1"].map(score20_to_pct)
    out["G2_pct"]=out["G2"].map(score20_to_pct)
    out["target_score_pct"]=out["G3"].map(score20_to_pct)
    out["grade"]=out["target_score_pct"].map(lambda x: grade_from_score(x)["grade"])
    out["credit_pass"]=out["target_score_pct"]>=50
    out=out.dropna(subset=["G1_pct","G2_pct","target_score_pct"])
    output_path.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(output_path,index=False)
    return out

if __name__ == "__main__":
    d=build_training_data(); print(f"Built {len(d)} records -> {PROCESSED_DATA}")
