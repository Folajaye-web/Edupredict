def grade_from_score(score: float):
    x=max(0,min(100,float(score)))
    bands=[(75,'A1',4.0,'Excellent'),(70,'B2',3.7,'Very Good'),(65,'B3',3.3,'Good'),(60,'C4',3.0,'Credit'),(55,'C5',2.7,'Credit'),(50,'C6',2.3,'Credit'),(45,'D7',1.5,'Pass'),(40,'E8',1.0,'Pass'),(0,'F9',0.0,'Fail')]
    for cut,g,gp,label in bands:
        if x>=cut: return g,gp
    return 'F9',0.0

def grade_label(score): return grade_from_score(score)[0]
