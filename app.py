from flask import Flask, request, jsonify, render_template
from thyroid_detection.pipline.prediction_pipeline import ThyroidClassifier, ThyroidData
from thyroid_detection.exception import ThyroidException

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting input values from the form
        data = request.form.to_dict()
        
        # Creating an instance of ThyroidData with the extracted form data
        thyroid_data = ThyroidData(
            sex=data.get('sex'),
            on_thyroxine=data.get('on_thyroxine'),
            query_on_thyroxine=data.get('query_on_thyroxine'),
            on_antithyroid_medication=data.get('on_antithyroid_medication'),
            sick=data.get('sick'),
            pregnant=data.get('pregnant'),
            I131_treatment=data.get('I131_treatment'),
            tumor=data.get('tumor'),
            hypopituitary=data.get('hypopituitary'),
            psych=data.get('psych'),
            age=int(data.get('age')),
            TSH=float(data.get('TSH')),
            T3=float(data.get('T3')),
            TT4=float(data.get('TT4')),
            T4U=float(data.get('T4U')),
            FTI=float(data.get('FTI'))
        )

        # Converting input data to DataFrame
        input_dataframe = thyroid_data.get_thyroid_input_data_frame()

        # Creating an instance of the ThyroidClassifier
        classifier = ThyroidClassifier()

        # Making the prediction
        prediction = classifier.predict(input_dataframe)

        # Preparing the response
        response = {
            'prediction': prediction,  # Mapped prediction result
            'input_data': data
        }
        return jsonify(response)

    except ThyroidException as e:
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
