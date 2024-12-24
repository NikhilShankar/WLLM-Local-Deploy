# cosine_prediction_helper.py
import os
import time
import boto3
import uuid
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import random
import tempfile
from WLLMSimilarityCalculatorAdvancedCorrected import SimilarityCalculatorAdvancedCorrected2
from WLLMModelLoader import WLLMModelLoader
from botocore.config import Config


class CosinePredictionHelper:
    def __init__(self, models: Dict[str, str], N: int, image_dataset_path: str,
                 s3_bucket: str, s3_output_prefix: str = "predictions"):
        """
        Initialize the predictor
        """
        self.models = models
        self.N = N
        self.image_dataset_path = image_dataset_path
        self.s3_bucket = s3_bucket
        self.s3_output_prefix = s3_output_prefix
        #session = boto3.Session(region_name="us-east-2")
        #self.s3_client = session.client('s3')

        config = Config(signature_version="s3v4", region_name="us-east-2")
        self.s3_client = boto3.client("s3", config=config)
        
        # Load models
        self.loaded_models = {}
        for key, value in models.items():
            keras_file = os.path.join(value, "best_model.keras")
            self.loaded_models[key] = WLLMModelLoader(keras_file).embedding_model

    def _upload_to_s3(self, local_path: str, filename: str) -> str:
        """Upload file to S3 and return public URL"""
        s3_key = f"{self.s3_output_prefix}/{filename}"
        self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
        
        # Make the object public
        #self.s3_client.put_object_acl(ACL='public-read', Bucket=self.s3_bucket, Key=s3_key)
        url = self.s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': self.s3_bucket,
                'Key': s3_key,
            },
            ExpiresIn=3600  # URL expires in 1 hour
        )
        path = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
        print(f"*******************************************")
        print(f"Upload s3 : {path} : {url}")
        return url

    def run_pipeline(self, test_image_path: str) -> Tuple[Dict[str, float], Dict[str, float], str]:
        """
        Run the prediction pipeline
        
        Returns:
            Tuple of (top_avg_personalities, top_score_personalities, plot_url)
        """
        # Calculate predictions
        predictions, times, total_time = self.calculate_predictions(test_image_path)

        # Create dataframe
        df = self.create_dataframe(predictions)

        # Create and save plot to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            top_avg_personalities, top_score_personalities = self.create_top_n_plot(df, self.N)
            self.save_personality_images(
                self.image_dataset_path,
                test_image_path,
                top_avg_personalities,
                N=self.N,
                save_path=tmp.name
            )
            
            # Upload to S3
            plot_filename = f"prediction_plot_{uuid.uuid4()}.png"
            plot_url = self._upload_to_s3(tmp.name, plot_filename)
            
            # Clean up temporary file
            print("17")
            #os.unlink(tmp.name)
        print("16")
        return (
            top_avg_personalities.to_dict(),
            top_score_personalities.to_dict(),
            plot_url
        )

    def save_personality_images(self, dataset_path: str, test_image_path: str, 
                              top_avg, N: int, save_path: str):
        """Save the personality images plot to a file"""
        print("1")
        personality_names = top_avg.index
        fig, axes = plt.subplots(1, N+1, figsize=(N * 3, 3))
        print("2")
        img = plt.imread(test_image_path)
        print("3")
        axes[0].axis("off")
        print("4")
        axes[0].imshow(img)
        print("5")
        axes[0].set_title("TEST IMAGE")
        print("6")
        for i in range(N):
            if i >= len(personality_names):
                break

            personality = personality_names[i]
            folder_path = os.path.join(dataset_path, personality)
            print(f"7 : {i}")
            image_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                print(f"No images found for personality: {personality}")
                continue
            print("8")
            image_path = os.path.join(folder_path, random.choice(image_files))
            img = plt.imread(image_path)
            print("9")
            axes[i+1].imshow(img)
            print("10")
            axes[i+1].axis("off")
            print("11")
            axes[i+1].set_title(f"{personality} - Rank - {i+1}")
            print("12")
        plt.title("Rankings based on Average")
        plt.tight_layout()
        print("13")
        plt.savefig(save_path, dpi=40)
        print("14")
        plt.close()
        print("15")

    def calculate_predictions(self, test_image_path: str):
        predictions = {}
        times = {}

        for model_name, model_path in self.models.items():
            start_time = time.time()
            embedding_folder = os.path.join(model_path, "embeddings")
            similarity_calculator = SimilarityCalculatorAdvancedCorrected2(embedding_folder)
            pred_by_avg_N, pred_by_total_N = similarity_calculator.calculate_similarity(
                test_image_path, 
                self.loaded_models[model_name], 
                100
            )
            predictions[model_name] = {
                "pred_by_avg_N": pred_by_avg_N,
                "pred_by_total_N": pred_by_total_N
            }
            times[model_name] = time.time() - start_time

        total_time = sum(times.values())
        return predictions, times, total_time

    def create_dataframe(self, predictions):
        data = defaultdict(list)
        for model_name, preds in predictions.items():
            for index, personality in enumerate(preds["pred_by_avg_N"]):
                data["personality"].append(personality[0])
                data["model_name"].append(model_name)
                data["pred_avg"].append(personality[1])
                data["pred_score"].append(preds["pred_by_total_N"][index][1])

        df = pd.DataFrame(data)
        df = df.pivot(index="model_name", columns="personality", values=["pred_avg", "pred_score"])
        df = df.sort_index(axis=1, level=1)
        return df

    def create_top_n_plot(self, df, top_n: int):
        top_avg = df["pred_avg"].mean(axis=0).nlargest(top_n)
        top_score = df["pred_score"].mean(axis=0).nlargest(top_n)
        return top_avg, top_score