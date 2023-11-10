from flask import Flask,jsonify,request,render_template,url_for
import pickle

app=Flask(__name__)

@app.route("/")
def index():
    return "default API" 

@app.route("/welcome")
def index1():
    return "Welcome to Flask"

@app.route("/predict")
def predict_flower():

    with open("model.pkl","rb") as model:
        ml_model=pickle.load(model)

    SepalLengthCm = 6
    SepalWidthCm = 4.8
    PetalLengthCm = 5.2
    PetalWidthCm = 3.75

    result = ml_model.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    if result[0]==2:
        iris_flower="iris_virginica"
    if result[0]==0:
        iris_flower="iris_setosa"
    if result[0]==1:
        iris_flower="iris_versicolor"

    return jsonify({"result":f"Flower is {iris_flower}"})


if __name__ == "__main__":
    app.run(debug=True)
