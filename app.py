import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from flask import Flask, request, render_template, jsonify
from keras.models import model_from_json

app = Flask(__name__)

con_train = pd.read_csv('con_train.csv')

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('model_weights.h5')

scaler = MinMaxScaler()
con_train['sales'] = scaler.fit_transform(con_train['sales'].values.reshape(-1, 1))

target_variable = 'sales'
sequence_length = 7

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input tweet text from the request
        store = int(request.form.get('storeNumber'))
        day = int(request.form.get('day'))
        month = int(request.form.get('month'))
        year = int(request.form.get('year'))
        print(store)

        # Preprocess the input
        date_str = str(year) + '-' + str(month) + '-' + str(day)
        date = datetime.strptime(date_str, '%Y-%m-%d')
        day_of_week = date.strftime('%A')
        input_data = pd.DataFrame({'store_nbr': [store], 'day_name': day_of_week})
        print(input_data)

        historical_data = con_train[(con_train[f'store_nbr_{store}'] == 1) & (con_train['day_name_' + day_of_week] == 1)]
        if len(historical_data) >= sequence_length:
            historical_data = historical_data[-sequence_length:]
        input_data = historical_data.drop(columns=[target_variable])
        input_data = np.array(input_data)

        # Make prediction
        predicted_sales = model.predict(input_data.reshape(1, sequence_length, input_data.shape[1]))
        predicted_sales = scaler.inverse_transform(predicted_sales)
        print(f"Predicted sales for {date_str} at store {store} : {round(predicted_sales[0][0])}")

        # Convert the prediction to a human-readable label
        result = str(round(predicted_sales[0][0]))

        # Debugging: Print the prediction
        print("Prediction:", result)

        return result  # Return prediction as JSON
    except Exception as e:
        return jsonify({"error": str(e)})  # Return error as JSON

if __name__ == '__main__':
    app.run(debug=True,port=5004)
