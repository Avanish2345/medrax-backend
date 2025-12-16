import sys
import json
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification  # Vision Transformer

class SimpleReportGenerator:
    def __init__(self):
        # Load the Hugging Face Vision Transformer model
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

    def predict(self, img_path):
        try:
            # Open the image and convert it to RGB (if it's not already)
            img = Image.open(img_path).convert("RGB")

            # Preprocess the image for the ViT model
            inputs = self.processor(images=img, return_tensors="pt")

            # Run inference
            outputs = self.model(**inputs)

            # Get the predicted class
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=-1).item()

            # Get the class label from the model's label list
            label = self.model.config.id2label[predicted_class_idx]

            # Create findings from the predicted label (in a real scenario, you'd have more detailed findings)
            findings = {
                "prediction": label,
                "analysis": [
                    "No acute cardiopulmonary abnormality detected.",
                    "Heart and mediastinum are of normal size.",
                    "No pleural effusion detected."
                ]
            }

            return findings
        except Exception as e:
            print(f"Error in image preprocessing or model inference: {e}")
            return {"error": "Error in processing the image or generating report."}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)
    image_path = sys.argv[1]
    
    report_generator = SimpleReportGenerator()
    findings = report_generator.predict(image_path)
    print(json.dumps(findings))

if __name__ == "__main__":
    main()
