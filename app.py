# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier

# app = Flask(__name__)

# df = pd.read_csv("heart.csv")

# categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
# label_encoders = {}

# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le

# X = df.drop('HeartDisease', axis=1)
# y = df['HeartDisease']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, random_state=42
# )

# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # ========================

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None

#     if request.method == "POST":
#         try:
#             input_data = {
#                 "Age": int(request.form["Age"]),
#                 "Sex": request.form["Sex"],
#                 "ChestPainType": request.form["ChestPainType"],
#                 "RestingBP": int(request.form["RestingBP"]),
#                 "Cholesterol": int(request.form["Cholesterol"]),
#                 "FastingBS": int(request.form["FastingBS"]),
#                 "RestingECG": request.form["RestingECG"],
#                 "MaxHR": int(request.form["MaxHR"]),
#                 "ExerciseAngina": request.form["ExerciseAngina"],
#                 "Oldpeak": float(request.form["Oldpeak"]),
#                 "ST_Slope": request.form["ST_Slope"]
#             }

#             user_df = pd.DataFrame([input_data])

#             for col in categorical_cols:
#                 user_df[col] = label_encoders[col].transform(user_df[col])

#             result = model.predict(user_df)[0]
#             prediction = "⚠️ Berisiko Gagal Jantung" if result == 1 else "✅ Tidak Berisiko Gagal Jantung"

#         except Exception as e:
#             prediction = f"Error: {str(e)}"

#     return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)

# ============================================================

from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder="../templates")

# Load & train model (sederhana)
df = pd.read_csv("heart.csv")

categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        data = {k: request.form[k] for k in request.form}
        data = pd.DataFrame([data])

        for col in categorical_cols:
            data[col] = encoders[col].transform(data[col])

        result = model.predict(data)[0]
        prediction = "⚠️ Berisiko Gagal Jantung" if result == 1 else "✅ Tidak Berisiko Gagal Jantung"

    return render_template("index.html", prediction=prediction)

