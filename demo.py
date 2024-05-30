#app.py
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




index.html

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thyroid Detection</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            background-color: #f8f9fa;
        }

        .container {
            margin-top: 50px;
        }

        .form-section {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
        }

        .form-group label {
            font-weight: bold;
        }

        .btn-primary {
            background-color: #007bff;
            border: none;
        }

        .btn-primary:hover {
            background-color: #0056b3;
        }

        .result-section {
            margin-top: 30px;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
        }

        .result-section h3 {
            font-weight: bold;
        }

        .footer {
            margin-top: 50px;
            text-align: center;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1 class="text-center">Thyroid Detection Prediction</h1>
        <div class="form-section">
            <form action="/predict" method="post">
                <div class="form-row">
                    <div class="form-group col-md-6">
                        <label for="age">Age</label>
                        <input type="number" class="form-control" id="age" name="age" required>
                    </div>
                    <div class="form-group col-md-6">
                        <label for="sex">Sex</label>
                        <select class="form-control" id="sex" name="sex" required>
                            <option value="M">Male</option>
                            <option value="F">Female</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group col-md-6">
                        <label for="on_thyroxine">On Thyroxine</label>
                        <select class="form-control" id="on_thyroxine" name="on_thyroxine" required>
                            <option value="f">False</option>
                            <option value="t">True</option>
                        </select>
                    </div>
                    <div class="form-group col-md-6">
                        <label for="query_on_thyroxine">Query on Thyroxine</label>
                        <select class="form-control" id="query_on_thyroxine" name="query_on_thyroxine" required>
                            <option value="f">False</option>
                            <option value="t">True</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group col-md-6">
                        <label for="on_antithyroid_medication">On Antithyroid Medication</label>
                        <select class="form-control" id="on_antithyroid_medication" name="on_antithyroid_medication" required>
                            <option value="f">False</option>
                            <option value="t">True</option>
                        </select>
                    </div>
                    <div class="form-group col-md-6">
                        <label for="sick">Sick</label>
                        <select class="form-control" id="sick" name="sick" required>
                            <option value="f">False</option>
                            <option value="t">True</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group col-md-6">
                        <label for="pregnant">Pregnant</label>
                        <select class="form-control" id="pregnant" name="pregnant" required>
                            <option value="f">False</option>
                            <option value="t">True</option>
                        </select>
                    </div>
                    <div class="form-group col-md-6">
                        <label for="I131_treatment">I131 Treatment</label>
                        <select class="form-control" id="I131_treatment" name="I131_treatment" required>
                            <option value="f">False</option>
                            <option value="t">True</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group col-md-6">
                        <label for="tumor">Tumor</label>
                        <select class="form-control" id="tumor" name="tumor" required>
                            <option value="f">False</option>
                            <option value="t">True</option>
                        </select>
                    </div>
                    <div class="form-group col-md-6">
                        <label for="hypopituitary">Hypopituitary</label>
                        <select class="form-control" id="hypopituitary" name="hypopituitary" required>
                            <option value="f">False</option>
                            <option value="t">True</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group col-md-6">
                        <label for="psych">Psych</label>
                        <select class="form-control" id="psych" name="psych" required>
                            <option value="f">False</option>
                            <option value="t">True</option>
                        </select>
                    </div>
                    <div class="form-group col-md-6">
                        <label for="TSH">TSH</label>
                        <input type="number" step="any" class="form-control" id="TSH" name="TSH" required>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group col-md-6">
                        <label for="T3">T3</label>
                        <input type="number" step="any" class="form-control" id="T3" name="T3" required>
                    </div>
                    <div class="form-group col-md-6">
                        <label for="TT4">TT4</label>
                        <input type="number" step="any" class="form-control" id="TT4" name="TT4" required>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group col-md-6">
                        <label for="T4U">T4U</label>
                        <input type="number" step="any" class="form-control" id="T4U" name="T4U" required>
                    </div>
                    <div class="form-group col-md-6">
                        <label for="FTI">FTI</label>
                        <input type="number" step="any" class="form-control" id="FTI" name="FTI" required>
                    </div>
                </div>
                <button type="submit" class="btn btn-primary btn-block">Predict</button>
            </form>
        </div>
        <div id="result" class="result-section">
            {% if prediction %}
            <h3 class="text-center">Prediction: {{ prediction }}</h3>
            {% endif %}
        </div>
        <div class="footer">
            <p>&copy; 2024 Thyroid Detection. All rights reserved.</p>
        </div>
    </div>
</body>

</html>


