import numpy as np
import pandas as pd

def process_categorical_columns(df):
    """
    Preprocesses the data so that categorical variables are interpreted as such for modeling.
    """
    for col in ['Extracurricular_Activities', 'Internet_Access', 'School_Type', 'Learning_Disabilities', 'Gender']:
        col_cat = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df[col_cat.columns[0]] = col_cat
        df.drop(columns = col, inplace=True)

    for col in ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality']:
        df[col] = pd.Categorical(df[col], categories=["Low", "Medium", "High"], ordered = True).codes

    df['Peer_Influence'] = pd.Categorical(df['Peer_Influence'], categories=["Negative", "Neutral", "Positive"], ordered = True).codes
    df['Parental_Education_Level'] = pd.Categorical(df['Parental_Education_Level'], categories=["High School", "College", "Postgraduate"], ordered = True).codes
    
    df = df[['Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities_Yes', 'Sleep_Hours',
    'Previous_Scores', 'Motivation_Level', 'Internet_Access_Yes', 'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality',
    'School_Type_Public', 'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities_Yes', 'Parental_Education_Level', 'Gender_Male', 'Exam_Score']]
    
    return df