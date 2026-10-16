import sys, json
from pathlib import Path
from datetime import date
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pandas as pd
import streamlit as st

from edupredict.db import init_db, SessionLocal, School, AssessmentPolicy, Student, AcademicSession, ClassGroup, StudentPlacement, Subject, StudentSubjectEnrollment, Result, Attendance, ContinuousAssessment, FinalExamination, Prediction, Intervention, PredictionEvaluation
from edupredict.ingest import upsert_students, import_results, import_attendance, ensure_subjects, ensure_session
from edupredict.grading import grade_from_score
from edupredict.analytics import class_overview, broadsheet, student_subject_report, attendance_pct
from edupredict.structure import ensure_class_group, assign_student, enroll_class_subject, enroll_subject, unenroll_subject, TERMS
from edupredict.results import upsert_result
from edupredict.crud import create_student, update_student, deactivate_student, upsert_assessment, get_assessment, upsert_examination, get_examination, create_prediction, list_predictions, create_intervention, list_interventions, update_intervention, evaluate_available, deactivate_subject, restore_subject
from edupredict.ai.predictor import predict_pre_exam
from edupredict.crud import get_engagement, upsert_engagement, student_report, class_report, subject_report

st.set_page_config(page_title='EduPredict Nigeria', page_icon='🎓', layout='wide')
init_db()
CLASS_LEVELS=['Primary 1','Primary 2','Primary 3','Primary 4','Primary 5','Primary 6','JSS1','JSS2','JSS3','SSS1','SSS2','SSS3']

def get_school():
    with SessionLocal() as s: return s.query(School).first()

def get_policy():
    with SessionLocal() as s:
        p=s.query(AssessmentPolicy).first()
        if not p:
            p=AssessmentPolicy(school_id=s.query(School).first().id,name='Default configurable assessment policy',ca1_weight=10,ca2_weight=10,assignment_weight=10,project_weight=10,exam_weight=60,credit_threshold=50); s.add(p); s.commit()
        return p

def periods(s,school_id): return s.query(AcademicSession).filter_by(school_id=school_id).order_by(AcademicSession.session_name.desc(),AcademicSession.id.desc()).all()
def plabel(x): return f'{x.session_name} — {x.term}'
def current_period(ps): return next((x for x in ps if x.is_current), ps[0] if ps else None)
def select_period(s, ps, key, label='Academic period'):
    """Return a real AcademicSession row from the current SQLAlchemy session.

    Streamlit reruns the script frequently. Never rely on a detached/stale ORM
    object or a label-to-object lookup after a write. The selectbox stores only
    the stable database id, then we reload that row in the same DB session.
    """
    if not ps:
        return None
    clean = [x for x in ps if x is not None and getattr(x, 'id', None) is not None]
    if not clean:
        return None
    options = {plabel(x): int(x.id) for x in clean}
    labels = list(options.keys())
    default_index = next((i for i, x in enumerate(clean) if bool(x.is_current)), 0)
    choice = st.selectbox(label, labels, index=default_index, key=key)
    period_id = options.get(choice)
    if period_id is None:
        return None
    period = s.query(AcademicSession).filter(AcademicSession.id == period_id).first()
    return period


def require_period(sess, context='this page'):
    if sess is None:
        st.warning(f'No valid academic period is available for {context}. Please create or activate an academic period first.')
        return False
    return True
def students_for_period(s,period):
    if not period: return []
    ids=[x.student_id for x in s.query(StudentPlacement).filter_by(session_id=period.id).all()]
    return s.query(Student).filter(Student.active==True,Student.id.in_(ids)).order_by(Student.student_code).all() if ids else []

school=get_school(); policy=get_policy()
with SessionLocal() as s: ensure_subjects(s)

st.sidebar.title('EduPredict Nigeria')
st.sidebar.caption('CRUD-first academic intelligence & early warning')
page=st.sidebar.radio('Navigate',['Dashboard','Students','Academic Structure','Continuous Assessment','Attendance','Engagement','AI Early Warning','Interventions','Final Examination','Final Results','Prediction Evaluation','Reports & Analytics','Data Import','Settings'])

if page=='Dashboard':
    st.header('School Academic Dashboard')
    with SessionLocal() as s:
        ps=periods(s,school.id); selected=current_period(ps)
        if ps:
            selected=select_period(s, ps, 'dashboard_period')
        ov=class_overview(s,school.id,selected.id if selected else None)
        a,b,c,d,e=st.columns(5); a.metric('Active students',ov['students']); b.metric('Students with results',ov['students_with_results']); c.metric('Average',f"{ov['average']:.1f}%" if ov['average'] is not None else '—'); d.metric('Students below credit',ov['below_credit']); e.metric('Credit rate',f"{((ov['students_with_results']-ov['below_credit'])/max(1,ov['students_with_results']))*100:.1f}%")
        st.info('Dashboard is derived from current stored academic data. Missing results are not treated as failures.')
        if selected:
            df=broadsheet(s,school.id,selected.session_name,selected.term)
            if not df.empty: st.dataframe(df.head(20),width='stretch',hide_index=True)

elif page=='Students':
    st.header('Students — CRUD')
    with SessionLocal() as s:
        ps=periods(s,school.id); current=current_period(ps); students=s.query(Student).filter_by(school_id=school.id).order_by(Student.student_code).all()
        rows=[]
        for i,x in enumerate([z for z in students if z.active],1):
            placement=s.query(StudentPlacement,ClassGroup).join(ClassGroup,StudentPlacement.class_group_id==ClassGroup.id).filter(StudentPlacement.session_id==current.id,StudentPlacement.student_id==x.id).first() if current else None
            cg=placement[1] if placement else None
            rows.append({'S/N':i,'Student ID':x.student_code,'Student Name':f'{x.first_name} {x.last_name}'.strip(),'Class':cg.class_level if cg else x.class_level,'Arm / Stream':cg.arm or 'General' if cg else x.arm or 'General'})
        st.dataframe(pd.DataFrame(rows),width='stretch',hide_index=True)
        with st.expander('Create student'):
            with st.form('create_student'):
                c1,c2=st.columns(2); code=c1.text_input('Student ID / Code'); first=c2.text_input('First name'); last=c1.text_input('Last name'); cls=c2.selectbox('Initial class',CLASS_LEVELS); arm=c1.text_input('Initial arm / stream'); gender=c2.selectbox('Gender',['Female','Male','Other','Prefer not to say']); save=st.form_submit_button('Create student')
                if save:
                    try: create_student(s,school_id=school.id,student_code=code,first_name=first,last_name=last,gender=gender,class_level=cls,arm=arm); st.success('Student created.'); st.rerun()
                    except Exception as exc: st.error(str(exc))
        with st.expander('Update / deactivate student'):
            active=[x for x in students if x.active]
            if active:
                stx=st.selectbox('Student',active,format_func=lambda x:f'{x.student_code} — {x.first_name} {x.last_name}',key='stu_edit')
                c1,c2=st.columns(2); new_first=c1.text_input('First name',stx.first_name,key='edit_first'); new_last=c2.text_input('Last name',stx.last_name,key='edit_last'); new_class=c1.selectbox('Class',CLASS_LEVELS,index=CLASS_LEVELS.index(stx.class_level) if stx.class_level in CLASS_LEVELS else 0,key='edit_class'); new_arm=c2.text_input('Arm / Stream',stx.arm or '',key='edit_arm')
                if st.button('Update student',type='primary'):
                    try: update_student(s,stx.id,first_name=new_first,last_name=new_last,class_level=new_class,arm=new_arm); st.success('Student updated.'); st.rerun()
                    except Exception as exc: st.error(str(exc))
                if st.button('Deactivate student',key='deactivate_student'): deactivate_student(s,stx.id); st.warning('Student deactivated.'); st.rerun()

elif page=='Academic Structure':
    st.header('Academic Structure — CRUD')
    with SessionLocal() as s:
        ps=periods(s,school.id); subjects=s.query(Subject).filter_by(active=True).order_by(Subject.name).all(); all_subjects=s.query(Subject).order_by(Subject.name).all(); t1,t2,t3,t4=st.tabs(['Sessions & Terms','Classes / Streams','Subject Catalogue','Student Enrollment'])
        with t1:
            st.dataframe(pd.DataFrame([{'Session':x.session_name,'Term':x.term,'Current':x.is_current} for x in ps]),width='stretch',hide_index=True)
            with st.form('period_form'):
                a,b,c=st.columns(3); sn=a.text_input('Academic session'); term=b.selectbox('Term',TERMS); cur=c.checkbox('Set as current',True); save=st.form_submit_button('Create / activate period')
                if save:
                    try: obj=ensure_session(s,school.id,sn.strip(),term); s.query(AcademicSession).filter_by(school_id=school.id).update({AcademicSession.is_current:False}); obj.is_current=cur; s.commit(); st.success('Academic period saved.'); st.rerun()
                    except Exception as exc: st.error(str(exc))
        with t2:
            if not ps: st.info('Create a period first.')
            else:
                sess=select_period(s,ps,'cg_period')
                if not require_period(sess, 'Classes / Streams'):
                    st.stop()
                groups=s.query(ClassGroup).filter_by(session_id=sess.id,active=True).order_by(ClassGroup.class_level,ClassGroup.arm).all(); st.dataframe(pd.DataFrame([{'Class':g.class_level,'Arm / Stream':g.arm or 'General','Display name':g.display_name} for g in groups]),width='stretch',hide_index=True)
                with st.form('group_form'):
                    a,b,c=st.columns(3); level=a.selectbox('Class level',CLASS_LEVELS); arm=b.text_input('Arm / Stream'); display=c.text_input('Display name'); save=st.form_submit_button('Create class / stream')
                    if save:
                        try: g=ensure_class_group(s,school.id,sess.id,level,arm); g.display_name=display.strip() or g.display_name; s.commit(); st.success('Class/stream created.'); st.rerun()
                        except Exception as exc: st.error(str(exc))
                if groups:
                    target=st.selectbox('Target class / stream',groups,format_func=lambda x:x.display_name,key='assign_group'); all_st=s.query(Student).filter_by(school_id=school.id,active=True).order_by(Student.student_code).all(); selected=st.multiselect('Students',all_st,format_func=lambda x:f'{x.student_code} — {x.first_name} {x.last_name}',key='assign_st');
                    if st.button('Assign selected students'): [assign_student(s,x.id,target.id,sess.id) for x in selected]; st.success(f'Assigned {len(selected)} student(s).')
        with t3:
            active_subjects=[x for x in all_subjects if x.active]
            archived_subjects=[x for x in all_subjects if not x.active]

            st.subheader('Active Subject Catalogue')
            st.dataframe(
                pd.DataFrame([
                    {
                      'Code':x.code,
                      'Subject':x.name,
                      'Stage':x.stage,
                      'Compulsory':x.compulsory,
                      'Status':'Active'
                    }
            for x in active_subjects
        ]),
        width='stretch',
        hide_index=True
    )

    if active_subjects:
        subject_to_archive=st.selectbox(
            'Archive subject',
            active_subjects,
            format_func=lambda x:f'{x.code} — {x.name}',
            key='archive_subject'
        )

        if st.button('Archive subject',key='archive_subject_btn'):
            deactivate_subject(s,subject_to_archive.id)
            st.success(
                f'{subject_to_archive.name} archived. '
                'Historical records were preserved.'
            )
            st.rerun()
    else:
        st.info('No active subjects.')

    st.subheader('Archived Subjects')

    if archived_subjects:
        st.dataframe(
            pd.DataFrame([
                {
                    'Code':x.code,
                    'Subject':x.name,
                    'Stage':x.stage,
                    'Compulsory':x.compulsory,
                    'Status':'Archived'
                }
                for x in archived_subjects
            ]),
            width='stretch',
            hide_index=True
        )

        subject_to_restore=st.selectbox(
            'Restore archived subject',
            archived_subjects,
            format_func=lambda x:f'{x.code} — {x.name}',
            key='restore_subject'
        )

        if st.button('Restore subject',key='restore_subject_btn'):
            restore_subject(s,subject_to_restore.id)
            st.success(f'{subject_to_restore.name} restored.')
            st.rerun()
    else:
        st.info('No archived subjects.')

        with st.form('subject_form'):
                a,b,c,d=st.columns(4); code=a.text_input('Code'); name=b.text_input('Subject name'); stage=c.selectbox('Stage',['Primary','JSS','SSS','JSS/SSS']); comp=d.checkbox('Compulsory'); save=a.form_submit_button('Create subject')
                if save:
                    try:
                        if s.query(Subject).filter_by(code=code.strip().upper()).first(): raise ValueError('Subject code already exists.')
                        s.add(Subject(code=code.strip().upper(),name=name.strip(),stage=stage,compulsory=comp,active=True)); s.commit(); st.success('Subject created.'); st.rerun()
                    except Exception as exc: st.error(str(exc))
        with t4:
            if not ps: st.info('Create a period first.')
            else:
                sess=select_period(s,ps,'en_period')
                if not require_period(sess, 'Student Enrollment'):
                    st.stop()
                students=s.query(Student).filter_by(school_id=school.id,active=True).order_by(Student.student_code).all(); student=st.selectbox('Student',students,format_func=lambda x:f'{x.student_code} — {x.first_name} {x.last_name}',key='en_student'); enrolled={x.subject_id for x in s.query(StudentSubjectEnrollment).filter_by(session_id=sess.id,student_id=student.id,active=True).all()}
                st.caption('Individual-student-first enrollment. Tick only subjects this student actually takes.')
                for sub in subjects:
                    val=st.checkbox(f'{sub.code} — {sub.name}',value=sub.id in enrolled,key=f'en_{sess.id}_{student.id}_{sub.id}')
                    if val != (sub.id in enrolled): enroll_subject(s,school.id,sess.id,student.id,sub.id) if val else unenroll_subject(s,sess.id,student.id,sub.id)

elif page=='Continuous Assessment':
    st.header('Continuous Assessment — CRUD / PRE-EXAM')
    st.caption('Only pre-examination evidence belongs here. The current examination score is deliberately unavailable.')
    with SessionLocal() as s:
        ps=periods(s,school.id)
        if not ps: st.warning('Create an academic period first.')
        else:
            sess=select_period(s,ps,'ca_period')
            if not require_period(sess, 'Continuous Assessment'):
                st.stop()
            students=students_for_period(s,sess); student=st.selectbox('Student',students,format_func=lambda x:f'{x.student_code} — {x.first_name} {x.last_name}',key='ca_student'); enrolled=[x for x in s.query(Subject).filter(Subject.id.in_([e.subject_id for e in s.query(StudentSubjectEnrollment).filter_by(session_id=sess.id,student_id=student.id,active=True).all()])).order_by(Subject.name).all()];
            if enrolled:
                subject=st.selectbox('Subject',enrolled,format_func=lambda x:f'{x.code} — {x.name}',key='ca_subject'); existing=get_assessment(s,sess.id,student.id,subject.id); a,b,c,d=st.columns(4); ca1=a.number_input('CA 1',0.,100.,float(existing.ca1) if existing and existing.ca1 is not None else 0.,key='ca1'); ca2=b.number_input('CA 2',0.,100.,float(existing.ca2) if existing and existing.ca2 is not None else 0.,key='ca2'); ass=c.number_input('Assignment',0.,100.,float(existing.assignment) if existing and existing.assignment is not None else 0.,key='ca_ass'); proj=d.number_input('Project',0.,100.,float(existing.project) if existing and existing.project is not None else 0.,key='ca_proj'); comment=st.text_area('Teacher comment',existing.teacher_comment if existing else '')
                if st.button('Save / update CA',type='primary'):
                    try: upsert_assessment(s,session_id=sess.id,student_id=student.id,subject_id=subject.id,ca1=ca1,ca2=ca2,assignment=ass,project=proj,comment=comment); st.success('Continuous assessment saved.'); st.rerun()
                    except Exception as exc: st.error(str(exc))
            else: st.info('Enroll the student in subjects first.')

elif page=='Attendance':
    st.header('Attendance — CRUD')
    with SessionLocal() as s:
        ps=periods(s,school.id)
        if ps:
            sess=select_period(s,ps,'att_period')
            if not require_period(sess, 'Attendance'):
                st.stop()
            students=students_for_period(s,sess); student=st.selectbox('Student',students,format_func=lambda x:f'{x.student_code} — {x.first_name} {x.last_name}',key='att_student'); existing=s.query(Attendance).filter_by(session_id=sess.id,student_id=student.id).first(); days=st.number_input('School days',0,300,int(existing.school_days) if existing else 65); present=st.number_input('Days present',0,300,int(existing.days_present) if existing else 0)
            if present<=days:
                st.metric('Attendance',f'{present/days*100:.1f}%' if days else '—')
                if st.button('Save / update attendance',type='primary'):
                    if existing: existing.school_days,existing.days_present,existing.days_absent=days,present,days-present
                    else: s.add(Attendance(session_id=sess.id,student_id=student.id,school_days=days,days_present=present,days_absent=days-present))
                    s.commit(); st.success('Attendance saved.'); st.rerun()
            else: st.error('Days present cannot exceed school days.')
        else: st.warning('Create an academic period first.')

elif page=='Engagement':
    st.header('Engagement — PRE-EXAM CRUD')
    st.caption('Teacher-observed engagement evidence can support early warning; it is never populated from examination results.')
    with SessionLocal() as s:
        ps=periods(s,school.id)
        if ps:
            sess=select_period(s,ps,'eng_period')
            students=students_for_period(s,sess)
            if students:
                student=st.selectbox('Student',students,format_func=lambda x:f'{x.student_code} — {x.first_name} {x.last_name}',key='eng_student')
                existing=get_engagement(s,sess.id,student.id)
                a,b,c,d=st.columns(4)
                hw=a.number_input('Homework completion %',0.,100.,float(existing.homework_rate) if existing and existing.homework_rate is not None else 0.,key='eng_hw')
                part=b.number_input('Participation %',0.,100.,float(existing.participation) if existing and existing.participation is not None else 0.,key='eng_part')
                beh=c.number_input('Behaviour %',0.,100.,float(existing.behaviour) if existing and existing.behaviour is not None else 0.,key='eng_beh')
                hrs=d.number_input('Study hours / week',0.,100.,float(existing.study_hours_week) if existing and existing.study_hours_week is not None else 0.,key='eng_hrs')
                if st.button('Save / update engagement',type='primary'):
                    try:
                        upsert_engagement(s,session_id=sess.id,student_id=student.id,homework_rate=hw,participation=part,behaviour=beh,study_hours_week=hrs)
                        st.success('Engagement saved.'); st.rerun()
                    except Exception as exc: st.error(str(exc))
            else: st.info('No students are placed in this academic period.')
        else: st.warning('Create an academic period first.')

elif page=='AI Early Warning':
    st.header('AI Early Warning — PHASE 5 PRE-EXAM MODEL')
    st.caption('Uses the corrected 19-feature Phase 5 Nigerian model. The current examination score is never a predictor. School pilot authorisation remains separate and is currently not granted.')
    with SessionLocal() as s:
        ps = periods(s, school.id)
        if not ps:
            st.warning('Create an academic period first.')
        else:
            sess = select_period(s, ps, 'ai_period')
            if not require_period(sess, 'AI Early Warning'):
                st.stop()
            students = students_for_period(s, sess)
            if not students:
                st.info('No active students are placed in this academic period. Complete Students and Class / Stream placement first.')
            else:
                student = st.selectbox(
                    'Student', students,
                    format_func=lambda x: f'{x.student_code} — {x.first_name} {x.last_name}',
                    key='ai_student'
                )
                enrollments = s.query(StudentSubjectEnrollment).filter_by(
                    session_id=sess.id, student_id=student.id, active=True
                ).all()
                subject_ids = [en.subject_id for en in enrollments]
                enrolled = s.query(Subject).filter(Subject.id.in_(subject_ids)).order_by(Subject.name).all() if subject_ids else []
                if not enrolled:
                    st.info('This student has no active subject enrollments for the selected period. Use Academic Structure → Student Enrollment before predicting.')
                else:
                    subject = st.selectbox(
                        'Subject', enrolled,
                        format_func=lambda x: f'{x.code} — {x.name}',
                        key='ai_subject'
                    )
                    if st.button('Generate pre-exam prediction', type='primary'):
                        try:
                            result = predict_pre_exam(
                                s, session_id=sess.id, student_id=student.id, subject_id=subject.id
                            )
                            prediction = create_prediction(
                                s, session_id=sess.id, student_id=student.id, subject_id=subject.id,
                                predicted_score=result['predicted_total'], risk_level=result['risk_level'],
                                below_credit_probability=result['below_credit_probability'],
                                rationale=result['rationale'], model_version=result['model_version'],
                                feature_snapshot=json.dumps(result['features'], default=str)
                            )
                            a, b, c, d = st.columns(4)
                            a.metric('Estimated overall total', f"{result['predicted_total']:.1f}/100")
                            b.metric('Predicted examination', f"{result['predicted_exam_score']:.1f}/60")
                            c.metric('Below-credit probability', f"{result['below_credit_probability']:.1%}")
                            d.metric('Early warning', 'YES' if result['early_warning'] else 'No')
                            st.subheader(f"Risk level: {result['risk_level']}")
                            st.write(result['rationale'])
                            st.caption(f"Model: {result['model_version']} · Prediction ID: {prediction.id} · Advisory only; pilot not authorised.")
                        except Exception as exc:
                            st.error(f'Phase 5 prediction is blocked: {exc}')
                    with st.expander('School-wide pre-exam prediction run'):
                        st.caption('Uses only active student-subject enrollments with complete current CA and attendance. Invalid/incomplete records are reported as not generated.')
                        if st.button('Run batch predictions', key='batch_predict', type='secondary'):
                            rows = []
                            generated = 0
                            failed = 0
                            for student_row in students:
                                student_enrollments = s.query(StudentSubjectEnrollment).filter_by(
                                    session_id=sess.id, student_id=student_row.id, active=True
                                ).all()
                                for enrollment in student_enrollments:
                                    subject_row = s.get(Subject, enrollment.subject_id)
                                    if subject_row is None:
                                        failed += 1
                                        continue
                                    try:
                                        result = predict_pre_exam(
                                            s, session_id=sess.id, student_id=student_row.id,
                                            subject_id=enrollment.subject_id
                                        )
                                        create_prediction(
                                            s, session_id=sess.id, student_id=student_row.id,
                                            subject_id=enrollment.subject_id,
                                            predicted_score=result['predicted_total'], risk_level=result['risk_level'],
                                            below_credit_probability=result['below_credit_probability'],
                                            rationale=result['rationale'], model_version=result['model_version'],
                                            feature_snapshot=json.dumps(result['features'], default=str)
                                        )
                                        rows.append({
                                            'Student ID': student_row.student_code,
                                            'Subject': subject_row.name,
                                            'Estimated total /100': round(result['predicted_total'], 1),
                                            'Exam /60': round(result['predicted_exam_score'], 1),
                                            'Below-credit probability': f"{result['below_credit_probability']:.1%}",
                                            'Early warning': 'YES' if result['early_warning'] else 'No',
                                            'Risk': result['risk_level'],
                                        })
                                        generated += 1
                                    except Exception as exc:
                                        failed += 1
                                        rows.append({
                                            'Student ID': student_row.student_code,
                                            'Subject': subject_row.name,
                                            'Estimated total /100': '—',
                                            'Exam /60': '—',
                                            'Below-credit probability': '—',
                                            'Early warning': '—',
                                            'Risk': f'Not generated: {exc}',
                                        })
                            if generated:
                                st.success(f'Generated {generated} prediction(s).')
                            if failed:
                                st.warning(f'{failed} prediction(s) were not generated; see the rows below for reasons.')
                            if rows:
                                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
                    history = list_predictions(s, sess.id, student.id, subject.id)
                    if history:
                        st.subheader('Prediction history')
                        st.dataframe(pd.DataFrame([{
                            'Date': item.created_at,
                            'Estimated total /100': item.predicted_score,
                            'Risk': item.risk_level,
                            'Model': item.model_version,
                        } for item in history]), width='stretch', hide_index=True)

elif page=='Interventions':
    st.header('Interventions — CRUD')
    with SessionLocal() as s:
        ps=periods(s,school.id)
        if ps:
            sess=select_period(s,ps,'iv_period')
            if not require_period(sess, 'Interventions'):
                st.stop()
            students=students_for_period(s,sess); student=st.selectbox('Student',students,format_func=lambda x:f'{x.student_code} — {x.first_name} {x.last_name}',key='iv_student'); preds=list_predictions(s,sess.id,student.id); pred=st.selectbox('Prediction (optional)',[None]+preds,format_func=lambda x:'No prediction' if x is None else f'#{x.id} — {x.risk_level} — {x.predicted_score:.1f}%',key='iv_pred'); typ=st.text_input('Intervention type','Academic support'); desc=st.text_area('Description'); assigned=st.text_input('Assigned to');
            if st.button('Create intervention',type='primary'):
                if desc.strip(): create_intervention(s,session_id=sess.id,student_id=student.id,subject_id=pred.subject_id if pred else None,prediction_id=pred.id if pred else None,intervention_type=typ,description=desc,assigned_to=assigned or None); st.success('Intervention created.'); st.rerun()
            items=list_interventions(s,sess.id,student.id)
            if items:
                intervention_rows=pd.DataFrame([
                    {
                        'ID':x.id,
                        'Type':x.intervention_type,
                        'Status':x.status,
                        'Assigned to':x.assigned_to or '—',
                        'Review date':x.review_date or '—'
                    }
                    for x in items
                ])
                st.dataframe(intervention_rows,width='stretch',hide_index=True)
            else:
                st.info('No interventions yet.')
            if items:
                iv=st.selectbox('Update intervention',items,format_func=lambda x:f'#{x.id} — {x.intervention_type} — {x.status}',key='iv_edit'); status=st.selectbox('Status',['OPEN','IN PROGRESS','REVIEWED','COMPLETED','ARCHIVED'],index=['OPEN','IN PROGRESS','REVIEWED','COMPLETED','ARCHIVED'].index(iv.status) if iv.status in ['OPEN','IN PROGRESS','REVIEWED','COMPLETED','ARCHIVED'] else 0)
                if st.button('Update intervention'): update_intervention(s,iv.id,status=status); st.success('Intervention updated.'); st.rerun()
        else: st.warning('Create an academic period first.')

elif page=='Final Examination':
    st.header('Final Examination — POST-EXAM CRUD')
    st.caption('This page is the formal boundary where the actual examination score enters the system.')
    with SessionLocal() as s:
        ps=periods(s,school.id)
        if ps:
            sess=select_period(s,ps,'exam_period')
            if not require_period(sess, 'Final Examination'):
                st.stop()
            students=students_for_period(s,sess); student=st.selectbox('Student',students,format_func=lambda x:f'{x.student_code} — {x.first_name} {x.last_name}',key='exam_student'); enrolled=[x for x in s.query(Subject).filter(Subject.id.in_([e.subject_id for e in s.query(StudentSubjectEnrollment).filter_by(session_id=sess.id,student_id=student.id,active=True).all()])).order_by(Subject.name).all()]
            if enrolled:
                subject=st.selectbox('Subject',enrolled,format_func=lambda x:f'{x.code} — {x.name}',key='exam_subject'); existing=get_examination(s,sess.id,student.id,subject.id); score=st.number_input('Actual examination score',0.,100.,float(existing.exam_score) if existing else 0.,key='actual_exam'); exam_date=st.date_input('Examination date',existing.exam_date if existing and existing.exam_date else date.today()); comment=st.text_area('Exam comment',existing.comment if existing else '')
                if st.button('Save / update actual examination',type='primary'):
                    try: upsert_examination(s,session_id=sess.id,student_id=student.id,subject_id=subject.id,exam_score=score,exam_date=exam_date,comment=comment); st.success('Actual examination saved.'); st.rerun()
                    except Exception as exc: st.error(str(exc))
            else: st.info('Enroll the student in subjects first.')
        else: st.warning('Create an academic period first.')

elif page=='Final Results':
    st.header('Final Results — derived from CA + actual examination')
    with SessionLocal() as s:
        ps=periods(s,school.id)
        if ps:
            sess=select_period(s, ps, 'fr_period'); students=students_for_period(s,sess); student=st.selectbox('Student',students,format_func=lambda x:f'{x.student_code} — {x.first_name} {x.last_name}',key='fr_student'); enrolled=[x for x in s.query(Subject).filter(Subject.id.in_([e.subject_id for e in s.query(StudentSubjectEnrollment).filter_by(session_id=sess.id,student_id=student.id,active=True).all()])).order_by(Subject.name).all()]
            rows=[]
            for subject in enrolled:
                ca=get_assessment(s,sess.id,student.id,subject.id); ex=get_examination(s,sess.id,student.id,subject.id); result=s.query(Result).filter_by(session_id=sess.id,student_id=student.id,subject_id=subject.id).first();
                if ca and ex:
                    vals={'ca1':ca.ca1 or 0,'ca2':ca.ca2 or 0,'assignment':ca.assignment or 0,'project':ca.project or 0,'exam':ex.exam_score}; total=sum(float(vals[k])*getattr(policy,k+'_weight')/100 for k in vals); grade,gp=grade_from_score(total); rows.append({'Subject':subject.name,'CA':round(total-float(ex.exam_score)*policy.exam_weight/100,2),'Exam':ex.exam_score,'Final Score':round(total,2),'Grade':grade,'Status':'Complete'})
                else: rows.append({'Subject':subject.name,'CA':'—','Exam':'—','Final Score':'—','Grade':'—','Status':'Awaiting data'})
            st.dataframe(pd.DataFrame(rows),width='stretch',hide_index=True)
            if st.button('Finalize available results',type='primary'):
                count=0
                for subject in enrolled:
                    ca=get_assessment(s,sess.id,student.id,subject.id); ex=get_examination(s,sess.id,student.id,subject.id)
                    if ca and ex:
                        upsert_result(s,session_id=sess.id,student_id=student.id,subject_id=subject.id,policy=policy,ca1=ca.ca1 or 0,ca2=ca.ca2 or 0,assignment=ca.assignment or 0,project=ca.project or 0,exam=ex.exam_score,reason='Final result generated from separated CA and actual examination records.'); count+=1
                st.success(f'{count} final result(s) finalized.')
        else: st.warning('Create an academic period first.')

elif page=='Prediction Evaluation':
    st.header('Prediction Evaluation — predicted vs actual')
    with SessionLocal() as s:
        ps=periods(s,school.id)
        if ps:
            sess=select_period(s, ps, 'eval_period'); items=evaluate_available(s,sess.id); rows=[]
            for x in items:
                p=s.get(Prediction,x.prediction_id); stx=s.get(Student,x.student_id); sub=s.get(Subject,x.subject_id); rows.append({'Student':stx.student_code if stx else x.student_id,'Subject':sub.name if sub else x.subject_id,'Predicted':round(p.predicted_score,2),'Actual':round(x.actual_exam_score,2),'Error':round(x.error,2),'Absolute error':round(x.absolute_error,2),'Risk':p.risk_level})
            if rows:
                st.dataframe(pd.DataFrame(rows),width='stretch',hide_index=True); st.metric('MAE',f"{pd.DataFrame(rows)['Absolute error'].mean():.2f}")
            else: st.info('No prediction has an actual examination match yet.')
        else: st.warning('Create an academic period first.')

elif page=='Reports & Analytics':
    st.header('Reports & Analytics')
    with SessionLocal() as s:
        ps=periods(s,school.id)
        if ps:
            sess=select_period(s,ps,'report_period')
            tabs=st.tabs(['School / Class','Student Report','Subject Analysis'])
            with tabs[0]:
                groups=s.query(ClassGroup).filter_by(session_id=sess.id,active=True).order_by(ClassGroup.class_level,ClassGroup.arm).all()
                group=st.selectbox('Class / stream', [None]+groups, format_func=lambda x:'Whole school' if x is None else x.display_name, key='report_group')
                level=group.class_level if group else None; arm=group.arm if group else None
                rep=class_report(s,school.id,sess.id,level,arm)
                a,b,c,d=st.columns(4); a.metric('Students',rep['students']); b.metric('Subjects',rep['subjects']); c.metric('Average',f"{rep['average']:.1f}%" if rep['average'] is not None else '—'); d.metric('Completion',f"{rep['completion_rate']:.1f}%")
                if not rep['student_table'].empty:
                    st.dataframe(rep['student_table'],width='stretch',hide_index=True)
                    st.download_button('Export class report CSV',rep['student_table'].to_csv(index=False).encode(),'class_report.csv','text/csv')
                if not rep['subject_table'].empty: st.dataframe(rep['subject_table'],width='stretch',hide_index=True)
            with tabs[1]:
                students=s.query(Student).filter_by(school_id=school.id,active=True).order_by(Student.student_code).all()
                stx=st.selectbox('Student',students,format_func=lambda x:f'{x.student_code} — {x.first_name} {x.last_name}',key='report_student')
                rep=student_report(s,school.id,stx.id,sess.id)
                for k,v in [('Subjects enrolled',rep['subjects_enrolled']),('Results recorded',rep['results_recorded']),('Completion',f"{rep['completion_rate']:.1f}%"),('Average',rep['average'] if rep['average'] is not None else '—'),('Attendance',f"{rep['attendance_pct']:.1f}%" if rep['attendance_pct'] is not None else '—')]:
                    st.metric(k,v)
                st.write('Missing subjects:', ', '.join(rep['missing_subjects']) if rep['missing_subjects'] else 'None')
                if rep['weak_subjects']: st.dataframe(pd.DataFrame(rep['weak_subjects']),width='stretch',hide_index=True)
            with tabs[2]:
                level=st.selectbox('Class level (optional)',[None]+CLASS_LEVELS,format_func=lambda x:'All classes' if x is None else x,key='subject_level')
                arms=s.query(ClassGroup).filter_by(session_id=sess.id,active=True).all()
                arm=st.selectbox('Arm / stream (optional)',[None]+sorted(set((g.arm or '') for g in arms)),format_func=lambda x:'All arms' if x is None else (x or 'General'),key='subject_arm')
                sdf=subject_report(s,school.id,sess.id,level,arm)
                if not sdf.empty:
                    st.dataframe(sdf,width='stretch',hide_index=True)
                    st.download_button('Export subject analysis CSV',sdf.to_csv(index=False).encode(),'subject_analysis.csv','text/csv')
        else: st.warning('Create an academic period first.')

elif page=='Data Import':
    st.header('Data Import — CRUD ingestion')
    with SessionLocal() as s:
        kind=st.selectbox('Data type',['Students','Results','Attendance','Subject Enrollment'])
        uploaded=st.file_uploader('Upload CSV',type=['csv'])
        if uploaded:
            try:
                df=pd.read_csv(uploaded, sep=None, engine='python')
            except Exception as exc:
                st.error(f'CSV could not be read: {exc}')
                st.stop()
            st.dataframe(df.head(20),width='stretch',hide_index=True)
            if kind=='Students':
                missing={'student_code'}-set(df.columns)
                outcome_cols=sorted(set(df.columns) & {'G1','G2','G3','target_score_pct','grade','credit_pass','exam','exam_score'})
                if missing:
                    st.error(f'Missing required field(s): {", ".join(sorted(missing))}')
                elif outcome_cols:
                    st.error('Student profile import blocked: outcome/examination fields are not allowed here: ' + ', '.join(outcome_cols))
                else:
                    st.success(f'Valid pre-exam student profile: {len(df)} row(s). No examination/outcome fields detected.')
                    st.caption('Extra profile fields are preserved separately from Student records. Missing names/classes are left blank/unassigned; no academic results are invented.')
                    if st.button('Import Student Profiles',type='primary'):
                        try:
                            n=upsert_students(df,s,school.id)
                            st.success(f'Imported/updated {n} new student record(s). Profile data preserved.')
                        except Exception as exc:
                            s.rollback(); st.error(f'Import failed: {exc}')
            elif st.button('Import CSV',type='primary'):
                try:
                    if kind=='Results': n=import_results(df,s,school.id,policy)
                    elif kind=='Attendance': n=import_attendance(df,s,school.id)
                    else:
                        n=0
                        for r in df.to_dict('records'):
                            stx=s.query(Student).filter_by(school_id=school.id,student_code=str(r['student_code'])).first(); sess=ensure_session(s,school.id,str(r['session']),str(r['term'])); sub=s.query(Subject).filter_by(code=str(r['subject_code']).upper()).first()
                            if stx and sub: enroll_subject(s,school.id,sess.id,stx.id,sub.id); n+=1
                    st.success(f'Processed {n} record(s).')
                except Exception as exc:
                    s.rollback(); st.error(f'Import failed: {exc}')

elif page=='Settings':
    st.header('Settings')
    with SessionLocal() as s:
        p=s.query(AssessmentPolicy).first(); c1,c2,c3=st.columns(3); name=c1.text_input('School name',school.name); state=c2.text_input('State',school.state); lga=c3.text_input('LGA',school.lga or '')
        if st.button('Save school profile'): school_obj=s.get(School,school.id); school_obj.name=name.strip() or school_obj.name; school_obj.state=state.strip() or school_obj.state; school_obj.lga=lga.strip() or None; s.commit(); st.success('School profile updated.')
        a,b,c,d,e=st.columns(5); ca1=a.number_input('CA1 %',0.,100.,p.ca1_weight); ca2=b.number_input('CA2 %',0.,100.,p.ca2_weight); ass=c.number_input('Assignment %',0.,100.,p.assignment_weight); proj=d.number_input('Project %',0.,100.,p.project_weight); exam=e.number_input('Exam %',0.,100.,p.exam_weight); credit=st.number_input('Credit threshold %',0.,100.,p.credit_threshold)
        if st.button('Save assessment policy',type='primary'):
            if abs(ca1+ca2+ass+proj+exam-100)>0.001: st.error('Assessment weights must add up to 100%.')
            else: p.ca1_weight,p.ca2_weight,p.assignment_weight,p.project_weight,p.exam_weight,p.credit_threshold=ca1,ca2,ass,proj,exam,credit; s.commit(); st.success('Assessment policy saved.')