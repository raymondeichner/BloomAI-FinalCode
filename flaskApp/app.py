from flask import Flask, render_template, request
from predict2 import classify

app = Flask(__name__)

# Base app page
@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def ask():
    user_input = request.form["user_input"]

    response = classify(user_input)

    return render_template("index.html", response=response, question=user_input)

if __name__ == "__main__":
    app.run(debug=True)