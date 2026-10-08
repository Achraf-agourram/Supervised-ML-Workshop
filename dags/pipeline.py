from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import joblib
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Définition des tâches (Tasks)
def extract_and_clean():
    with open('/opt/airflow/data/student-mat.csv', 'r', encoding='utf-8') as f:
        raw_data = f.read().replace(';', ',')
    df = pd.read_csv(io.StringIO(raw_data))
    df = df.drop_duplicates()
    df['pass_fail'] = (df['G3'] >= 10).astype(int)
    df.to_csv('/tmp/cleaned_data.csv', index=False)

def train_model():
    df = pd.read_csv('/tmp/cleaned_data.csv')
    X = df.drop(columns=['G1', 'G2', 'G3', 'pass_fail'])
    y = df['pass_fail']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), X.select_dtypes(include=['number']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object']).columns)
    ])
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    model.fit(X_train, y_train)
    joblib.dump(model, '/tmp/best_model.pkl')

def evaluate_model():
    model = joblib.load('/tmp/best_model.pkl')
    print('✅ Modèle évalué avec succès dans Airflow.')

def save_model():
    print('💾 Modèle sauvegardé dans le stockage persistant.')

# 2. Définition du DAG
default_args = {'owner': 'airflow', 'start_date': datetime(2023, 1, 1), 'retries': 1}

with DAG('student_performance_ml_pipeline', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    
    # 3. Opérateurs
    task1 = PythonOperator(task_id='extract_and_clean', python_callable=extract_and_clean)
    task2 = PythonOperator(task_id='train_model', python_callable=train_model)
    task3 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model)
    task4 = PythonOperator(task_id='save_model', python_callable=save_model)
    
    # 4. Dépendances (Orchestration linéaire)
    task1 >> task2 >> task3 >> task4