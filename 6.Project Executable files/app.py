from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('srujan vec/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        number_of_customers = float(request.form['number_of_customers'])
        menu_price = float(request.form['menu_price'])
        marketing_spend = float(request.form['marketing_spend'])
        cuisine_type = request.form['cuisine_type']
        average_customer_spending = float(request.form['average_customer_spending'])
        promotions = float(request.form['promotions'])
        reviews = float(request.form['reviews'])

        # Prepare the input data for prediction
        input_data = np.array([[number_of_customers, menu_price, marketing_spend, average_customer_spending, promotions, reviews]])

        # Dummy example: map cuisine type to a numerical value (e.g., 0, 1, 2)
        cuisine_mapping = {'Italian': 0, 'Chinese': 1, 'Indian': 2}  # Example mapping
        if cuisine_type in cuisine_mapping:
            input_data = np.append(input_data, cuisine_mapping[cuisine_type])
        else:
            input_data = np.append(input_data, -1)  # Unknown cuisine type

        input_data = input_data.reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Render the output template with the prediction
        return render_template('output.html', prediction=prediction)

    return render_template('input.html')

if __name__ == '__main__':
    app.run(debug=True)
