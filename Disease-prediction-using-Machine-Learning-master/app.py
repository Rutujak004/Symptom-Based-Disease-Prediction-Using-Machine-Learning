import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# ---------------------------------------------
# SYMPTOMS & DISEASE LIST
# ---------------------------------------------
l1 = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease = ['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria',
'Chicken pox','Dengue','Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D',
'Hepatitis E','Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia',
'Dimorphic hemmorhoids(piles)','Heart attack','Varicose veins','Hypothyroidism','Hyperthyroidism',
'Hypoglycemia','Osteoarthristis','Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne',
'Urinary tract infection','Psoriasis','Impetigo']

replace = {
    'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,
    'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,
    'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,'Common Cold':26,
    'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40
}

# ---------------------------------------------
# TRAINING & TEST DATA
# ---------------------------------------------
df = pd.read_csv("Training.csv")
df.replace(replace, inplace=True)
df["prognosis"] = df["prognosis"].astype(str).str.strip().replace(replace)

X = df[l1]
y = df["prognosis"].astype(int)

tr = pd.read_csv("Testing.csv")
tr.replace(replace, inplace=True)
tr["prognosis"] = tr["prognosis"].astype(str).str.strip().replace(replace)

X_test = tr[l1]
y_test = tr["prognosis"].astype(int)

# ---------------------------------------------
# TRAIN MODELS
# ---------------------------------------------
dt = DecisionTreeClassifier().fit(X, y)
rf = RandomForestClassifier().fit(X, y)
nb = GaussianNB().fit(X, y)

# ---------------------------------------------
# STREAMLIT UI
# ---------------------------------------------
st.title("🩺 AI Disease Predictor")
st.markdown("### Advanced Machine Learning Diagnosis System")

# Patient info
patient_name = st.text_input("👤 Enter Patient Name:")

# Symptom selection
st.subheader("Select Symptoms")
symptom1 = st.selectbox("Symptom 1", [""] + sorted(l1))
symptom2 = st.selectbox("Symptom 2", [""] + sorted(l1))
symptom3 = st.selectbox("Symptom 3", [""] + sorted(l1))
symptom4 = st.selectbox("Symptom 4", [""] + sorted(l1))
symptom5 = st.selectbox("Symptom 5", [""] + sorted(l1))

# Convert to input vector
selected_symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]
input_vector = [0] * len(l1)
for s in selected_symptoms:
    if s in l1:
        input_vector[l1.index(s)] = 1

# Prediction
if st.button("🔮 Predict Disease"):
    if sum(input_vector) == 0:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        pred_dt = dt.predict([input_vector])[0]
        pred_rf = rf.predict([input_vector])[0]
        pred_nb = nb.predict([input_vector])[0]

        st.subheader("Results")
        st.write(f"🌳 **Decision Tree**: {disease[pred_dt]}")
        st.write(f"🌲 **Random Forest**: {disease[pred_rf]}")
        st.write(f"📊 **Naive Bayes**: {disease[pred_nb]}")

        if patient_name:
            st.success(f"Prediction completed for patient: **{patient_name}**")

# Accuracy
if st.checkbox("Show Model Accuracy"):
    st.write("✅ Decision Tree Accuracy:", round(accuracy_score(y_test, dt.predict(X_test)) * 100, 2), "%")
    st.write("✅ Random Forest Accuracy:", round(accuracy_score(y_test, rf.predict(X_test)) * 100, 2), "%")
    st.write("✅ Naive Bayes Accuracy:", round(accuracy_score(y_test, nb.predict(X_test)) * 100, 2), "%")

st.markdown("---")
st.info("⚠️ Note: This is an AI-assisted prediction tool. Please consult a doctor for an accurate medical diagnosis.")
