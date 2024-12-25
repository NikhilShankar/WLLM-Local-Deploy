# app.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from AWSS3CosinePredictionHelper import CosinePredictionHelper

app = Flask(__name__)

# Configuration
MODEL_MAP = {
    "ModelA": "TrainedModels/WLLM-Model-0001",
    "ModelB": "TrainedModels/WLLM-Model-0002",
    "ModelC": "TrainedModels/WLLM-Model-0003"
}

predictor = CosinePredictionHelper(
    models=MODEL_MAP,
    N=3,
    image_dataset_path="dataset",
    s3_bucket="wllm-public",
    s3_output_prefix="predictions"
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            file.save(tmp.name)
            
            # Run prediction pipeline
            try:
                top_avg, top_score, plot_url, topN = predictor.run_pipeline(tmp.name, shouldCreateANewImageWithPredictions=False)
                print("20")
                response = {
                    'top_avg_personalities': top_avg,
                    'top_score_personalities': top_score,
                    'plot_url': plot_url,
                    'topN': topN
                }
                print("21")
                return jsonify(response), 200
            
            finally:
                # Clean up temporary file
                print("22")
                #os.unlink(tmp.name)
                print("23")
                
    except Exception as e:
        print("24")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)