import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabeteshtml')
def diabeteshtml():
    return render_template('diabetes.html')

@app.route('/hearthtml')
def hearthtml():
    return render_template('heart.html')

@app.route('/dpfhtml')
def dpfhtml():
    return render_template('dpf.html')

# Data cleaning function
def clean_blood_pressure(df, column_name='Blood Pressure'):
    df[column_name] = df[column_name].str.split('/').str[0]
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

# diabetes dataset
df = pd.read_csv('diabetes.csv')
x = df[['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2)

# heart dataset
df1 = pd.read_csv('heart_attack_dataset.csv')
df1 = clean_blood_pressure(df1)  # Clean the Blood Pressure column
x1 = df1[['Age', 'Sex', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Diabetes', 'Smoking', 'Obesity']]
y1 = df1['Heart Attack Risk']
train_x, test_x, train_y, test_y = tts(x1, y1, test_size=0.2, random_state=42)

# diabetes train
knn = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
knn.fit(x_train, y_train)

# heart train
knn1 = KNeighborsClassifier(n_neighbors=13)
knn1.fit(train_x, train_y)

# recommendation
recommendations = {
    'low': [
        "Maintain a healthy lifestyle. Continue regular exercise and a balanced diet.",
        "Stay hydrated by drinking plenty of water throughout the day.",
        "Get regular sleep and maintain a consistent sleep schedule for optimal health.",
        "Consider incorporating stress-reduction techniques such as meditation or yoga into your daily routine."
    ],
    'moderate': [
        "Increase physical activity by incorporating more aerobic exercise and strength training into your routine.",
        "Monitor your diet closely and aim for a balanced intake of fruits, vegetables, lean proteins, and whole grains.",
        "Schedule regular check-ups with your doctor to monitor your health status and address any concerns.",
        "Reduce intake of processed foods, sugary beverages, and foods high in saturated fats to improve overall health."
    ],
    'high': [
        "Consult a healthcare provider for personalized advice and guidance on managing your health.",
        "Consider working with a nutritionist to develop a customized meal plan tailored to your dietary needs and goals.",
        "Increase exercise intensity and frequency to promote weight loss and improve cardiovascular health.",
        "Monitor blood sugar levels regularly and track food intake to identify patterns and make necessary adjustments."
    ],
    'very_high': [
        "Seek immediate medical advice and treatment from a healthcare professional.",
        "Follow all medical recommendations closely, including medication regimens and lifestyle modifications.",
        "Consider joining a support group or seeking counseling to help cope with the emotional aspects of managing diabetes.",
        "Educate yourself about diabetes management and complications to make informed decisions about your health."
    ]
}

def get_risk_category(prediction_percentage):
    if prediction_percentage < 20 :
        return 'low'
    elif 20 <= prediction_percentage < 50 :
        return 'moderate'
    elif 50 <= prediction_percentage < 80 :
        return 'high'
    else:
        return 'very_high'

def get_recommendation(prediction_percentage):
    category = get_risk_category(prediction_percentage)
    return recommendations[category]

# diabetes prediction
@app.route('/diabetespredict', methods=['POST'])
def diabetespredict():
    if request.method == 'POST':
        user_input = [
            int(request.form['Glucose']),
            int(request.form['BloodPressure']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            int(request.form['Age'])
        ]
        y_pred = knn.predict([user_input])
        y_prob = knn.predict_proba([user_input])[0][1] * 100
        y_prob = round(y_prob, 2)
        recommendation = get_recommendation(y_prob)
        return render_template('predict.html', prediction=recommendation, prediction1=f"The probability of getting diabetes is {y_prob}%")
    else:
        return render_template('predict.html', prediction="Error predicting CHD. Please check your input.")

# heart prediction
@app.route('/heartpredict', methods=['POST'])
def heartpredict():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cholesterol = int(request.form['cholesterol'])
            blood_pressure = int(request.form['blood_pressure'])  # Ensure this is converted to int
            heart_rate = int(request.form['heart_rate'])
            diabetes = int(request.form['diabetes'])
            Smoking = int(request.form['Smoking'])
            obesity = int(request.form['obesity'])

            ans = [[age, sex, cholesterol, blood_pressure, heart_rate, diabetes, Smoking, obesity]]

            y_prob1 = knn1.predict_proba(ans)[0][1] * 100
            y_prob1 = round(y_prob1, 2)
            return render_template('result.html', probability=f"{y_prob1}%")
        except ValueError as e:
            return render_template('result.html', probability="Error: Please ensure all input values are numeric.")
    else:
        return render_template('result.html', probability="Error: Request method not supported.")

# dpf prediction
@app.route('/dpf_predict', methods=['POST'])
def dpf_predict():
    if request.method == 'POST':
        first_degree_relatives = int(request.form['first_degree_relatives'])
        age_of_diagnosis = request.form['age_of_diagnosis']
        second_degree_relatives = int(request.form['second_degree_relatives'])

        age_of_diagnosis = list(map(int, age_of_diagnosis.split(',')))

        dpf = 0.2

        if first_degree_relatives > 0:
            if first_degree_relatives == 1:
                dpf += 0.3
            elif first_degree_relatives >= 2:
                dpf += 0.6

            for age in age_of_diagnosis:
                if age < 50:
                    dpf += 0.2
                else:
                    dpf += 0.1

        dpf += second_degree_relatives * 0.1

        return render_template('dpf.html', dpf_prediction=f"Diabetes Pedigree Function: {dpf}")
    else:
        return render_template('dpf.html', dpf_prediction="Error")

if __name__ == '__main__':
    app.run(debug=True)
