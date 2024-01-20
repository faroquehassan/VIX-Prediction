import pickle

from flask import Flask
from flask import request
from flask import jsonify

app = Flask('predict')
@app.route('/predict', methods=['GET'])
def predict():
    with open('model.bin', 'rb') as f_in:
        model = pickle.load(f_in)

    with open('X_test.pkl', 'rb') as file:
        X_test = pickle.load(file)

    y_pred = model.predict(X_test)
    X_test.reset_index(inplace=True)

    result = {
        'Date': X_test['Date'][0],'VIX next close: ': float(y_pred)
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)