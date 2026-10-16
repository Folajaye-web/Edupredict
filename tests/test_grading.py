from edupredict.grading import grade_from_score
def test_grades():
 assert grade_from_score(80)[0]=="A1"
 assert grade_from_score(50)[0]=="C6"
 assert grade_from_score(39)[0]=="F9"
