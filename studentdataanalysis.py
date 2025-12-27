import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl

df=pd.read_csv("sample.txt")
print(df.head())
print(df.info())
df['StudyHours']=df['StudyHours'].fillna(df['StudyHours'].mean())
df['Attendance']=df['Attendance'].fillna(df['Attendance'].median())
df['Marks']=df['Marks'].fillna(df['Marks'].mean())
df['Gender_encoded']=df['Gender'].map({"Male":0, "Female":1})
def performance_level(Marks):
    if Marks>=70:
        return 'Good'
    elif Marks>=50:
        return 'Average'
    else :
        return 'Poor'
df['Performance']=df['Marks'].apply(performance_level)
def attendance_category(att):
    if att>=80:
        return 'High'
    elif att>=60:
        return 'Medium'
    else:
        return 'Low'
df['AttendanceCategory']=df['Attendance'].apply(attendance_category)
print("Mean Marks: ",df['Marks'].mean())
print("Median Marks: ",df['Marks'].median())
print("Standard Deviation: ",df['Marks'].std())
print(df[['StudyHours','Attendance','Marks']].corr())
def prediction(row):
    score=0
    if row['StudyHours']>=4:
        score+=1
    if row['Attendance']>=75:
        score+=1
    if row['Marks']>=50:
        score+=2
    if score>=3:
        return 'High chances pass'
    else :
        return 'High chances fail'

df['Prediction_result']=df.apply(prediction, axis=1)
mpl.scatter(df['StudyHours'],df['Marks'])
mpl.xlabel('StudyHours')
mpl.ylabel('Marks')
mpl.title("StudyHours vs Marks")
mpl.grid(True)
mpl.show()
df['Performance'].value_counts().plot(kind='bar')
mpl.xlabel('Performance')
mpl.ylabel('Number of students')
mpl.title("Performance vs Number of students")
mpl.grid(True)
mpl.show()
df.groupby('Gender')['Marks'].mean().plot(kind='bar')
mpl.xlabel('Gender')
mpl.ylabel('Avg Marks')
mpl.title("Gender vs Average Marks")
mpl.grid(True)
mpl.show()
print("Final data",df)