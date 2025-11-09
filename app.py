from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
app = Flask(__name__)

# -----------------------------
# Load dataset from GitHub
# -----------------------------
url = 'https://raw.githubusercontent.com/PankajMishra99/OOPS-QUESTION/main/new_insurance_data.csv'
df = pd.read_csv(url)

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Label encoding
def encoder(new_df):
    cate_col = ['sex', 'smoker', 'region']
    le = LabelEncoder()
    for col in cate_col:
        new_df[col] = le.fit_transform(new_df[col])
    return new_df

df = encoder(df)

# Outlier removal
outlier_col = ['bmi', 'Anual_Salary', 'Hospital_expenditure', 'past_consultations']
def remove_outlier(new_df):
    for col in outlier_col:
        q1 = new_df[col].quantile(0.25)
        q3 = new_df[col].quantile(0.75)
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        new_df = new_df[(new_df[col] > lb) & (new_df[col] < ub)]
    return new_df

df = remove_outlier(df)

# Prepare model
x = df.drop('charges', axis=1)
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

# -----------------------------
# Flask Routes
# -----------------------------
# model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        Claim_Amount = float(request.form['Claim_Amount'])
        past_consultations = int(request.form['past_consultations'])
        num_of_steps = float(request.form['num_of_steps'])
        Hospital_expenditure = float(request.form['Hospital_expenditure'])
        NUmber_of_past_hospitalizations = int(request.form['NUmber_of_past_hospitalizations'])
        Anual_Salary = float(request.form['Anual_Salary'])
        region = int(request.form['region'])

        # Arrange all 12 features in same order as training data
        features = np.array([[age, sex, bmi, children, smoker, Claim_Amount,
                              past_consultations, num_of_steps, Hospital_expenditure,
                              NUmber_of_past_hospitalizations, Anual_Salary, region]])

        prediction = model.predict(features)[0]

        return render_template('index.html',
                               prediction_text=f'Predicted Insurance Charge: ₹{prediction:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
