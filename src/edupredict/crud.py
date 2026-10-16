"""Public CRUD facade for EduPredict v3.3.

UI code should call these functions instead of embedding database mutations.
"""
from .services.assessment_service import get_assessment, upsert_assessment
from .services.examination_service import get_examination, upsert_examination
from .services.intervention_service import create_intervention, list_interventions, update_intervention, deactivate_intervention
from .services.prediction_service import create_prediction, list_predictions
from .services.evaluation_service import evaluate_prediction, evaluate_available

__all__=[
    'get_assessment','upsert_assessment','get_examination','upsert_examination',
    'create_intervention','list_interventions','update_intervention','deactivate_intervention',
    'get_engagement','upsert_engagement','student_report','class_report','subject_report',
    'create_prediction','list_predictions','evaluate_prediction','evaluate_available'
]
from .services.student_service import create_student, update_student, deactivate_student
from .services.academic_service import create_or_update_period, update_class_group, deactivate_class_group

from .services.engagement_service import get_engagement, upsert_engagement
from .services.reporting_service import student_report, class_report, subject_report
from .services.subject_service import deactivate_subject, restore_subject
__all__ += ['deactivate_subject','restore_subject']
