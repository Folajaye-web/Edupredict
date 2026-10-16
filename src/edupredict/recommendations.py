def recommendations(student: dict, predicted_score: float) -> list[str]:
    actions=[]
    if predicted_score < 40:
        actions.append("Schedule an immediate academic support review with the subject teacher.")
    elif predicted_score < 50:
        actions.append("Place the student on a targeted improvement plan and review progress weekly.")
    elif predicted_score < 60:
        actions.append("Provide focused revision on weak topics and monitor the next assessment.")
    if student.get("absences", 0) >= 10:
        actions.append("Review attendance barriers and agree on an attendance improvement plan.")
    elif student.get("absences", 0) >= 6:
        actions.append("Monitor attendance closely and follow up on repeated absence.")
    if student.get("failures", 0) >= 2:
        actions.append("Provide remedial support for prerequisite concepts from previously failed work.")
    if student.get("studytime", 2) <= 1:
        actions.append("Introduce a structured weekly study timetable with teacher/parent check-ins.")
    if student.get("schoolsup", "no") == "no":
        actions.append("Consider subject-specific school support or a peer study group.")
    if student.get("internet", "no") == "no":
        actions.append("Provide offline learning materials so lack of connectivity does not limit practice.")
    if not actions:
        actions.append("Maintain current support and monitor the next assessment for meaningful change.")
    return actions
