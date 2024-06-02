from flask import Flask, request, render_template
import pickle
import numpy as np
model = pickle.load(open('model.pkl', 'rb'))

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict(): 
    return render_template ("predict.html")

@app.route("/instruction")
def instruction(): 
    return render_template ("instruction.html")

@app.route("/prediction",methods=["POST"])
def analyze():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    return render_template('return.html', data=prediction)

@app.route("/aboutus")
def aboutus():
    return render_template ("aboutus.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)

