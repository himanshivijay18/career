import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import logging


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fallback_secret_key_for_development")


school_data = pd.read_csv('attached_assets/school.csv')
college_data = pd.read_csv('attached_assets/college.csv')


def preprocess_school_data():
    df = school_data.copy()
    
    df = df.iloc[1:].reset_index(drop=True)
    
    encoders = {}
    X_encoded = df.copy()
    
    for column in df.columns:
        if column not in ['Student Name', 'Recommended Path']:
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(df[column].astype(str))
            encoders[column] = le
    
    X = X_encoded.drop(['Student Name', 'Recommended Path'], axis=1)
    y = df['Recommended Path']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, encoders

def preprocess_college_data():
    df = college_data.copy()
    
    df = df.iloc[1:].reset_index(drop=True)
    
    encoders = {}
    X_encoded = df.copy()
    
    for column in df.columns:
        if column not in ['Student Name', 'Recommended Path']:
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(df[column].astype(str))
            encoders[column] = le
    
    X = X_encoded.drop(['Student Name', 'Recommended Path'], axis=1)
    y = df['Recommended Path']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, encoders

school_model, school_encoders = preprocess_school_data()
college_model, college_encoders = preprocess_college_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/choice')
def choice():
    return render_template('choice.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/school_form')
def school_form():
    subject_options = school_data['Subject Enjoyed'].unique()[1:]  
    topic_options = school_data['Topic Interest'].unique()[1:]
    activity_options = school_data['Whole Day Activity'].unique()[1:]
    talent_options = school_data['Natural Talent'].unique()[1:]
    task_options = school_data['Task Preference'].unique()[1:]
    free_time_options = school_data['Free Time Activity'].unique()[1:]
    extracurricular_options = school_data['Extracurricular'].unique()[1:]
    profession_options = school_data['Exciting Profession'].unique()[1:]
    admired_options = school_data['Admired Person'].unique()[1:]
    learning_options = school_data['Learning Style'].unique()[1:]
    work_options = school_data['Work Preference'].unique()[1:]
    tried_options = school_data['Tried Activities'].unique()[1:]
    inspired_options = school_data['Inspired Place'].unique()[1:]
    
    return render_template('school_form.html', 
                           subject_options=subject_options,
                           topic_options=topic_options,
                           activity_options=activity_options,
                           talent_options=talent_options,
                           task_options=task_options,
                           free_time_options=free_time_options,
                           extracurricular_options=extracurricular_options,
                           profession_options=profession_options,
                           admired_options=admired_options,
                           learning_options=learning_options,
                           work_options=work_options,
                           tried_options=tried_options,
                           inspired_options=inspired_options)

@app.route('/college_form')
def college_form():
    activity_options = college_data['Preferred Activity'].unique()[1:]  # Skip header
    skill_options = college_data['Skill to Develop'].unique()[1:]
    value_options = college_data['Career Value'].unique()[1:]
    environment_options = college_data['Work Environment Preference'].unique()[1:]
    description_options = college_data['Self Description'].unique()[1:]
    working_options = college_data['Working Preference'].unique()[1:]
    startup_options = college_data['Startup Interest'].unique()[1:]
    industry_options = college_data['Curious Industry'].unique()[1:]
    vision_options = college_data['5-Year Vision'].unique()[1:]
    
    return render_template('college_form.html',
                           activity_options=activity_options,
                           skill_options=skill_options,
                           value_options=value_options,
                           environment_options=environment_options,
                           description_options=description_options,
                           working_options=working_options,
                           startup_options=startup_options,
                           industry_options=industry_options,
                           vision_options=vision_options)

@app.route('/predict_school', methods=['POST'])
def predict_school():
    if request.method == 'POST':
        try:
            form_data = {
                'Student Name': request.form.get('name'),
                'Subject Enjoyed': request.form.get('subject'),
                'Topic Interest': request.form.get('topic'),
                'Whole Day Activity': request.form.get('activity'),
                'Natural Talent': request.form.get('talent'),
                'Task Preference': request.form.get('task'),
                'Free Time Activity': request.form.get('free_time'),
                'Extracurricular': request.form.get('extracurricular'),
                'Exciting Profession': request.form.get('profession'),
                'Admired Person': request.form.get('admired'),
                'Learning Style': request.form.get('learning'),
                'Work Preference': request.form.get('work'),
                'Tried Activities': request.form.get('tried'),
                'Inspired Place': request.form.get('inspired')
            }
            
            encoded_data = {}
            for column, value in form_data.items():
                if column != 'Student Name':
                    encoder = school_encoders.get(column)
                    if encoder:
                        try:
                            encoded_data[column] = encoder.transform([str(value)])[0]
                        except:
                            encoded_data[column] = encoder.transform([str(encoder.classes_[0])])[0]
            
            features = pd.DataFrame([encoded_data])
            
            prediction = school_model.predict(features)[0]
            
            importances = school_model.feature_importances_
            feature_importance_dict = dict(zip(school_model.feature_names_in_, importances))
            sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_importance[:3]
            
            session['name'] = form_data['Student Name']
            session['prediction'] = prediction
            session['student_type'] = 'school'
            session['top_features'] = [f[0] for f in top_features]
            
            return redirect(url_for('result'))
        
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            flash("An error occurred during prediction. Please try again.", "error")
            return redirect(url_for('school_form'))

@app.route('/predict_college', methods=['POST'])
def predict_college():
    if request.method == 'POST':
        try:
            form_data = {
                'Student Name': request.form.get('name'),
                'Preferred Activity': request.form.get('activity'),
                'Skill to Develop': request.form.get('skill'),
                'Career Value': request.form.get('value'),
                'Work Environment Preference': request.form.get('environment'),
                'Self Description': request.form.get('description'),
                'Working Preference': request.form.get('working'),
                'Startup Interest': request.form.get('startup'),
                'Curious Industry': request.form.get('industry'),
                '5-Year Vision': request.form.get('vision')
            }
            
            encoded_data = {}
            for column, value in form_data.items():
                if column != 'Student Name':
                    encoder = college_encoders.get(column)
                    if encoder:
                        try:
                            encoded_data[column] = encoder.transform([str(value)])[0]
                        except:
                            encoded_data[column] = encoder.transform([str(encoder.classes_[0])])[0]
            
            features = pd.DataFrame([encoded_data])
            
            prediction = college_model.predict(features)[0]
            
            importances = college_model.feature_importances_
            feature_importance_dict = dict(zip(college_model.feature_names_in_, importances))
            sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_importance[:3]
            
            session['name'] = form_data['Student Name']
            session['prediction'] = prediction
            session['student_type'] = 'college'
            session['top_features'] = [f[0] for f in top_features]
            
            return redirect(url_for('result'))
        
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            flash("An error occurred during prediction. Please try again.", "error")
            return redirect(url_for('college_form'))

@app.route('/result')
def result():
    name = session.get('name', 'Student')
    prediction = session.get('prediction', 'No prediction available')
    student_type = session.get('student_type', 'student')
    top_features = session.get('top_features', [])
    
    school_paths = {
        'Humanities': "Arts, Literature, History, Psychology, Sociology, Languages",
        'Humanities / Communication of law': "Journalism, Media Studies, Law, Public Relations",
        'Science (Non Medical)': "Physics, Chemistry, Mathematics, Computer Science, Engineering",
        'Medicine or Research': "Biology, Chemistry, Medicine, Pharmacy, Biotech",
        'Civil Services': "Political Science, Public Administration, Economics, History",
        'Design or Fine Arts': "Visual Arts, Fine Arts, Design, Architecture, Animation"
    }
    
    college_paths = {
        'IT Support / QA': "Information Technology, Quality Assurance, Technical Support",
        'Research / Lab Technician': "Scientific Research, Laboratory Work, Data Analysis",
        'Banking / Insurance': "Finance, Risk Assessment, Customer Service, Accounts",
        'General Career Counseling': "Education Guidance, Career Development, Student Support",
        'Technical Consultant': "IT Consulting, Business Analysis, System Architecture",
        'Software Development': "Programming, Application Design, Software Engineering",
        'Healthcare / Environment': "Health Services, Environmental Science, Public Health",
        'Accountancy / Finance': "Financial Analysis, Auditing, Tax Consulting",
        'Web Development': "Website Creation, Front-end/Back-end Development, UX/UI",
        'Public Relations / Teaching': "Communication, Education, Community Engagement",
        'Marketing / Sales': "Brand Management, Market Research, Sales Strategy",
        'Content Writing / Journalism': "Content Creation, Reporting, Media Production",
        'Digital Marketing': "Social Media Management, SEO, Online Advertising",
        'Operations / Sales': "Business Operations, Sales Management, Supply Chain",
        'Data Science': "Data Analysis, Machine Learning, Statistical Modeling"
    }
    
    path_explanation = ""
    if student_type == 'school':
        path_explanation = school_paths.get(prediction, "")
    else:
        path_explanation = college_paths.get(prediction, "")
        
    return render_template('result.html', 
                           name=name, 
                           prediction=prediction, 
                           student_type=student_type,
                           path_explanation=path_explanation,
                           top_features=top_features)
