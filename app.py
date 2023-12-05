from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained personality prediction model
model = pickle.load(open('personality_model.pkl', 'rb'))


@app.route('/')
def personality_home():
    return render_template('personality_home.html')


@app.route('/taketest')
def taketest():
    return render_template('taketest.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect answers to all 10 questions
    answers = [request.form[f'question{i}'] for i in range(1, 11)]

    # Concatenate answers into a single string
    data = ' '.join(answers)

    # Use the model to predict personality
    arr = np.array([data])
    pred = model.predict(arr)

    # Render the result template
    return render_template('personality_result.html', data=pred[0])

@app.route('/predict2', methods=['POST'])
def predict2():
    # Collect answers to all 10 questions
    answers = [request.form[f'question{i}'] for i in range(11, 21)]

    # Concatenate answers into a single string
    data = ' '.join(answers)

    # Use the model to predict personality
    arr = np.array([data])
    pred = model.predict(arr)

    # Render the result template
    return render_template('personality_result.html', data=pred[0])

@app.route('/predict3', methods=['POST'])
def predict3():
    # Collect answers to all 10 questions
    answers = [request.form[f'question{i}'] for i in range(21, 31)]

    # Concatenate answers into a single string
    data = ' '.join(answers)

    # Use the model to predict personality
    arr = np.array([data])
    pred = model.predict(arr)
    # Render the result template
    return render_template('personality_result.html', data=pred[0])

@app.route('/predict4', methods=['POST'])
def predict4():
    data = request.form['selfIntroduction']
    arr = np.array([data])
    pred = model.predict(arr)
    return render_template('personality_result.html', data=pred[0])

# Add other routes as before

if __name__ == "__main__":
    app.run(debug=True)
