import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

df = pd.read_csv("./fixed_datased.csv")
X = df.iloc[:,0:8]
Y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

AL = AdaBoostClassifier()
AL.fit(X_train, Y_train)

# Simpan model
pickle.dump(AL, open('model.pkl', 'wb'))

# Memuat model
model = pickle.load(open('model.pkl', 'rb'))

# Melakukan prediksi dengan format yang benar
print(model.predict([[3953, 0, 0, 1, 0, 1, 2, 7.544067522789540]]))
