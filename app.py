from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'loan_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Verify that the model is loaded correctly
print(f"Model type: {type(model)}")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Loan APPROVED' if prediction[0] == 1 else 'Loan NOT APPROVED'
    except Exception as e:
        output = f"Error during prediction: {str(e)}"

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
