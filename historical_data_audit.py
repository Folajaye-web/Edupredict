import pandas as pd

p = r"C:\Users\l2e\OneDrive\Desktop\Nigeria_Student_Records_2023_2026_Combined.csv"
df = pd.read_csv(p)

print("\n=== EDU PREDICT HISTORICAL DATA INTEGRITY AUDIT ===")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n--- Missing values ---")
print(df.isna().sum().to_string())

print("\n--- Score ranges ---")
cols = [
    "CA1", "CA2", "Assignment", "Project", "Exam",
    "Total", "Attendance", "Previous_Term_Total",
    "Previous_Year_Average"
]
print(df[cols].describe().T[["min", "max", "mean"]].to_string())

print("\n--- TOTAL CONSISTENCY ---")
calculated = df[["CA1", "CA2", "Assignment", "Project", "Exam"]].sum(axis=1)
print("Incorrect Total:", int((calculated != df["Total"]).sum()))
print("Maximum calculated total:", calculated.max())

print("\n--- GRADE CONSISTENCY ---")

def grade(x):
    if x >= 75:
        return "A1"
    elif x >= 70:
        return "B2"
    elif x >= 65:
        return "B3"
    elif x >= 60:
        return "C4"
    elif x >= 55:
        return "C5"
    elif x >= 50:
        return "C6"
    elif x >= 45:
        return "D7"
    elif x >= 40:
        return "E8"
    else:
        return "F9"

expected = df["Total"].apply(grade)
print("Incorrect Grades:", int((expected != df["Grade"]).sum()))

print("\nGrade counts:")
print(df["Grade"].value_counts(dropna=False).to_string())

print("\n--- ATTENDANCE ---")
print("Below 0:", int((df["Attendance"] < 0).sum()))
print("Above 100:", int((df["Attendance"] > 100).sum()))

print("\n--- SCORE LIMITS ---")
limits = {
    "CA1": 10,
    "CA2": 10,
    "Assignment": 10,
    "Project": 10,
    "Exam": 60,
    "Total": 100,
}

for column, maximum in limits.items():
    below = int((df[column] < 0).sum())
    above = int((df[column] > maximum).sum())
    print(f"{column}: below 0={below}, above max={above}")

print("\n--- KEY COUNTS ---")
print("Students:", df["Student_ID"].nunique())
print("Schools:", df["School"].nunique())
print("Sessions:", df["Academic_Session"].nunique())
print("Terms:", df["Term"].nunique())
print("Classes:", df["Class"].nunique())
print("Arms:", df["Arm"].nunique())
print("Subjects:", df["Subject"].nunique())

print("\n--- DUPLICATES ---")
duplicates = df.duplicated(
    ["Student_ID", "Academic_Session", "Term", "Subject"]
).sum()
print("Student/Session/Term/Subject duplicates:", int(duplicates))

print("\n=== END AUDIT ===")
