import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from WLLMSimilarityCalculatorAdvancedCorrected import SimilarityCalculatorAdvancedCorrected2
from WLLMModelLoader import WLLMModelLoader
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random


class CosinePredictionHelper:
    def __init__(self, models: Dict[str, str], N: int, image_dataset_path: str):
        self.models = models
        self.N = N
        self.image_dataset_path = image_dataset_path
        self.loaded_models = {}
        for key, value in models.items():
            keras_file = os.path.join(value, "best_model.keras")
            self.loaded_models[key] = WLLMModelLoader(keras_file).embedding_model



    def run_pipeline(self, test_image_path: str, save_path: str):
        # Step 1: Calculate predictions
        predictions, times, total_time = self.calculate_predictions(test_image_path)

        # Step 2: Create dataframe
        df = self.create_dataframe(predictions)
        # Step 6: Create and save top N plot
        top_n_plot_path = os.path.join(save_path, "top_n_plot.png")
        top_avg_personalities, top_score_personalities = self.create_top_n_plot(df, self.N, top_n_plot_path)
        self.plot_personality_images(self.image_dataset_path, test_image_path, top_avg_personalities, N=self.N)
        #self.plot_personality_images(self.image_dataset_path, test_image_path, top_score_personalities, N=self.N)
        return top_avg_personalities, top_score_personalities

    def calculate_predictions(self, test_image_path: str):
        predictions = {}
        times = {}

        for model_name, model_path in self.models.items():
            start_time = time.time()

            
            embedding_folder = os.path.join(model_path, "embeddings")

            # Assuming `SimilarityCalculatorAdvancedCorrected2` is defined elsewhere
            similarity_calculator = SimilarityCalculatorAdvancedCorrected2(embedding_folder)

            pred_by_avg_N, pred_by_total_N = similarity_calculator.calculate_similarity(test_image_path, self.loaded_models[model_name], 100)

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


    def create_top_n_plot(self, df, top_n: int, save_path: str):
        top_avg = df["pred_avg"].mean(axis=0).nlargest(top_n)
        top_score = df["pred_score"].mean(axis=0).nlargest(top_n)
        return top_avg, top_score


    def plot_personality_images(self, dataset_path: str, test_image_path, top_avg, N: int):
        personality_names = top_avg.index
        fig, axes = plt.subplots(1, N+1, figsize=(N * 3, 3))
        img = plt.imread(test_image_path)
        axes[0].axis("off")
        axes[0].imshow(img)
        axes[0].set_title("TEST IMAGE")
        for i in range(N):
            if i >= len(personality_names):
                break

            personality = personality_names[i]
            folder_path = os.path.join(dataset_path, personality)
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                print(f"No images found for personality: {personality}")
                continue

            # Load the first image
            image_path = os.path.join(folder_path, random.choice(image_files))
            img = plt.imread(image_path)

            # Plot image
            axes[i+1].imshow(img)
            axes[i+1].axis("off")
            axes[i+1].set_title(f"{personality} - Rank - {i+1} ")
        

        plt.title("Rankings based on Average")
        plt.tight_layout()
        plt.show()