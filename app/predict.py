from ultralytics import YOLO
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Predictor:
    """
    Predictor class that detects ingredients from an image and generates a recipe.
    """

    def __init__(self, yolo_model_path="app\models\YOLO\last.pt",
                 t5_model_path=f"app\models\T5"):
        """
        Initializes the predictor with model paths.

        Args:
            yolo_model_path (str, optional): Path to the trained YOLO model. Defaults to ".\models\YOLO\last.pt".
            t5_model_path (str, optional): Path to the trained T5 model. Defaults to ".\models\T5".
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load YOLO model
        self.yolo_model = YOLO(yolo_model_path).to(self.device)

        # Load T5 model and tokenizer
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_path)

    def predict(self, image_path):
        """
        Detects ingredients from an image and generates a recipe.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Dictionary containing the detected ingredients and generated recipe.
        """

        # Perform inference
        results = self.yolo_model(image_path)
        image_name = os.path.basename(image_path)
        
        # Extract ingredients and remove item counts
        ingredients = []
        for result in results:
            # Save the image with bounding boxes
            result.save(filename=f"app/results/{image_name}")  # Save the image to the results directory
            
            for *xyxy, conf, cls in result.boxes.data:  # iterate over detections
                ingredient_name = self.yolo_model.names[int(cls)]  # get ingredient name from class ID
                # Remove any digits and leading/trailing spaces from the ingredient name
                ingredient_name = ''.join([i for i in ingredient_name if not i.isdigit()]).strip()
                # Only add ingredient if it's not already in the list
                if ingredient_name not in ingredients:
                    ingredients.append(ingredient_name)  # add to ingredients list

        # Join ingredients into a string
        ingredients_str = ', '.join(ingredients)
        ingredients_str = ingredients_str.replace('_', ' ')

        # Generate recipe
        prompt = f"Generate recipe from these ingredients: {ingredients_str}"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        outputs = self.t5_model.generate(**inputs, max_new_tokens=512, num_beams=2, no_repeat_ngram_size=2, repetition_penalty=1.2)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "detected_ingredients": ingredients_str,
            "generated_recipe": generated_text,
            "result_img_path": f"http://localhost:8000/app/results/{image_name}"
        }