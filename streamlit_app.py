from tkinter import *
import numpy as np
import pandas as pd
from tkinter import ttk
from sklearn.metrics import accuracy_score

# ---------------------------------------------
# SYMPTOMS & DISEASE LIST
# ---------------------------------------------
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
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

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria',
'Chicken pox','Dengue','Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D',
'Hepatitis E','Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia',
'Dimorphic hemmorhoids(piles)','Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism',
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


l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# ---------------------------------------------
# TRAINING & TEST DATA
# ---------------------------------------------
df = pd.read_csv("Training.csv")
df.replace(replace, inplace=True)
df["prognosis"] = df["prognosis"].astype(str).str.strip()   # remove spaces
df["prognosis"] = df["prognosis"].replace(replace)
X = df[l1]
y = df["prognosis"].astype(int)   


tr = pd.read_csv("Testing.csv")
tr.replace(replace, inplace=True)
tr["prognosis"] = tr["prognosis"].astype(str).str.strip()
tr["prognosis"] = tr["prognosis"].replace(replace)
X_test = tr[l1]
y_test = tr["prognosis"].astype(int)   #force labels to int


# ---------------------------------------------
# PREDICTION FUNCTIONS
# ---------------------------------------------
def DecisionTree():
    from sklearn import tree
    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X,y)

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    test_input = [0]*len(l1)
    for z in psymptoms:
        if z in l1:
            test_input[l1.index(z)] = 1

    predict = clf3.predict([test_input])
    predicted=predict[0]

    if predicted < len(disease):
        result = disease[predicted]
        print(" Decision Tree Prediction:", result)   # terminal output
        t1.delete("1.0", END)
        t1.insert(END, result)
        t1.config(bg="#d4edda", fg="#155724")
    else:
        print("Decision Tree Prediction: Not Found")
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")
        t1.config(bg="#f8d7da", fg="#721c24")

def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    test_input = [0]*len(l1)
    for z in psymptoms:
        if z in l1:
            test_input[l1.index(z)] = 1

    predict = clf4.predict([test_input])
    predicted=predict[0]

    if predicted < len(disease):
        result = disease[predicted]
        print("🌲 Random Forest Prediction:", result)   # terminal output
        t2.delete("1.0", END)
        t2.insert(END, result)
        t2.config(bg="#d4edda", fg="#155724")
    else:
        print("❌ Random Forest Prediction: Not Found")
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")
        t2.config(bg="#f8d7da", fg="#721c24")

def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    test_input = [0]*len(l1)
    for z in psymptoms:
        if z in l1:
            test_input[l1.index(z)] = 1

    predict = gnb.predict([test_input])
    predicted=predict[0]

    if predicted < len(disease):
        result = disease[predicted]
        print("📊 Naive Bayes Prediction:", result)   # terminal output
        t3.delete("1.0", END)
        t3.insert(END, result)
        t3.config(bg="#d4edda", fg="#155724")
    else:
        print("❌ Naive Bayes Prediction: Not Found")
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")
        t3.config(bg="#f8d7da", fg="#721c24")

def predict_all():
    DecisionTree()
    randomforest()
    NaiveBayes()

def clear_all():
    Symptom1.set('')
    Symptom2.set('')
    Symptom3.set('')
    Symptom4.set('')
    Symptom5.set('')
    Name.set('')
    t1.delete("1.0", END)
    t2.delete("1.0", END)
    t3.delete("1.0", END)
    t1.config(bg="white", fg="black")
    t2.config(bg="white", fg="black")
    t3.config(bg="white", fg="black")

# ---------------------------------------------
# ACCURACY FUNCTION
# ---------------------------------------------
def calculate_accuracy():
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB

    clf_dt = tree.DecisionTreeClassifier()
    clf_dt.fit(X, y)
    y_pred_dt = clf_dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt) * 100

    clf_rf = RandomForestClassifier()
    clf_rf.fit(X, np.ravel(y))
    y_pred_rf = clf_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf) * 100

    clf_nb = GaussianNB()
    clf_nb.fit(X, np.ravel(y))
    y_pred_nb = clf_nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb) * 100

    print("\n Model Accuracy on Testing Data:")
    print(f"   Decision Tree : {acc_dt:.2f} %")
    print(f"   Random Forest : {acc_rf:.2f} %")
    print(f"   Naive Bayes   : {acc_nb:.2f} %\n")

# ---------------------------------------------
# MODERN UI DESIGN
# ---------------------------------------------
root = Tk()
root.title("AI Disease Predictor")
root.configure(background='#f0f8ff')
root.geometry("1000x700")

Name = StringVar()
Symptom1 = StringVar()
Symptom2 = StringVar()
Symptom3 = StringVar()
Symptom4 = StringVar()
Symptom5 = StringVar()

style = ttk.Style()
style.configure('TButton', font=('Arial', 12), padding=10)
style.configure('TLabel', font=('Arial', 11), background='#f0f8ff')

header_frame = Frame(root, bg='#2c3e50', height=120)
header_frame.pack(fill="x", pady=(0, 20))

title_label = Label(header_frame, text="AI Disease Predictor", 
                   font=('Arial', 28, 'bold'), 
                   fg='white', bg='#2c3e50')
title_label.pack(pady=30)

subtitle_label = Label(header_frame, text="Advanced Machine Learning Diagnosis System", 
                      font=('Arial', 14), 
                      fg='#ecf0f1', bg='#2c3e50')
subtitle_label.pack()

main_frame = Frame(root, bg='#f0f8ff')
main_frame.pack(fill=BOTH, expand=True, padx=30)

left_frame = Frame(main_frame, bg='#f0f8ff')
left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 20))

patient_section = LabelFrame(left_frame, text=" Patient Information ", 
                            font=('Arial', 12, 'bold'), 
                            bg='#f0f8ff', fg='#2c3e50', bd=2)
patient_section.pack(fill="x", pady=(0, 15))

Label(patient_section, text="Patient Name:", font=('Arial', 11), 
      bg='#f0f8ff', fg='#2c3e50').grid(row=0, column=0, sticky=W, padx=10, pady=10)

NameEn = Entry(patient_section, textvariable=Name, font=('Arial', 11), 
               width=30, bd=2, relief=GROOVE)
NameEn.grid(row=0, column=1, padx=10, pady=10, sticky=EW)

symptoms_section = LabelFrame(left_frame, text=" Select Symptoms ", 
                             font=('Arial', 12, 'bold'), 
                             bg='#f0f8ff', fg='#2c3e50', bd=2)
symptoms_section.pack(fill=BOTH, expand=True)

OPTIONS = sorted(l1)
symptoms = [
    ("Symptom 1", Symptom1),
    ("Symptom 2", Symptom2),
    ("Symptom 3", Symptom3),
    ("Symptom 4", Symptom4),
    ("Symptom 5", Symptom5)
]

for i, (text, var) in enumerate(symptoms):
    Label(symptoms_section, text=text + ":", font=('Arial', 11), 
          bg='#f0f8ff', fg='#2c3e50').grid(row=i, column=0, sticky=W, padx=10, pady=8)
    option = OptionMenu(symptoms_section, var, *OPTIONS)
    option.config(font=('Arial', 10), width=25, bd=2, relief=GROOVE)
    option.grid(row=i, column=1, padx=10, pady=8, sticky=EW)

button_frame = Frame(left_frame, bg='#f0f8ff')
button_frame.pack(fill="x", pady=20)

predict_btn = Button(button_frame, text="Predict All Diseases", 
                    command=predict_all, font=('Arial', 12, 'bold'),
                    bg='#27ae60', fg='white', bd=0, padx=20, pady=10,
                    cursor='hand2')
predict_btn.pack(side=LEFT, padx=(0, 10))

clear_btn = Button(button_frame, text="Clear All", 
                  command=clear_all, font=('Arial', 12),
                  bg='#e74c3c', fg='white', bd=0, padx=20, pady=10,
                  cursor='hand2')
clear_btn.pack(side=LEFT, padx=(0, 10))

accuracy_btn = Button(button_frame, text="Check Accuracy", 
                     command=calculate_accuracy, font=('Arial', 12),
                     bg='#3498db', fg='white', bd=0, padx=20, pady=10,
                     cursor='hand2')
accuracy_btn.pack(side=LEFT)

right_frame = Frame(main_frame, bg='#f0f8ff')
right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

results_section = LabelFrame(right_frame, text=" Prediction Results ", 
                           font=('Arial', 12, 'bold'), 
                           bg='#f0f8ff', fg='#2c3e50', bd=2)
results_section.pack(fill=BOTH, expand=True)



t1 = Text(results_section, height=2, width=40, font=('Arial', 11))
t1.pack(pady=5)
t2 = Text(results_section, height=2, width=40, font=('Arial', 11))
t2.pack(pady=5)
t3 = Text(results_section, height=2, width=40, font=('Arial', 11))
t3.pack(pady=5)

footer_frame = Frame(root, bg='#2c3e50', height=40)
footer_frame.pack(fill="x", side=BOTTOM)

footer_label = Label(footer_frame, text="Note: This is an AI-assisted prediction tool. Please consult a healthcare professional for accurate diagnosis.",
                    font=('Arial', 10), fg='#bdc3c7', bg='#2c3e50')
footer_label.pack(pady=10)

patient_section.columnconfigure(1, weight=1)
symptoms_section.columnconfigure(1, weight=1)
button_frame.columnconfigure(0, weight=1)

root.mainloop()
