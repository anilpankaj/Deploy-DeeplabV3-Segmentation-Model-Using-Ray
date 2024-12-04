import ray
from ray import serve
from fastapi import FastAPI, Request
from fastapi.responses import Response
from PIL import Image
import torch
from torchvision import transforms
import io
import requests
import numpy as np

# Initialize Ray Serve
ray.init(address="auto", namespace="serve")
serve.start() #it will deploy for you a controller and a HTTP Proxy on rear cluster

# FastAPI app
app = FastAPI()

@serve.deployment # it's going to instantiate and deploy a model onto a task, running on your cluster
class DeepLabv3Model:
    def __init__(self):
        # Load the DeepLabV3 model from PyTorch Hub
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Preprocessing pipeline for input images
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    async def __call__(self, request: Request):
        content_type = request.headers.get("content-type")
        
        if content_type and "multipart/form-data" in content_type:
            form = await request.form()
            file = form["file"]
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        else:
            data = await request.json()
            image_url = data.get("image_url")
            if not image_url:
                return {"error": "No image URL provided"}
            
            # Fetch image, bypassing SSL certificate verification
            response = requests.get(image_url, verify=False)
            if response.status_code != 200:
                return {"error": "Failed to fetch image from the URL"}
            image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Preprocess the image
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            output = output.argmax(0).byte().cpu().numpy()  # Segmentation mask

        # Create a color map for visualization (21 classes in DeepLabV3 COCO model)
        colormap = np.zeros((21, 3), dtype=np.uint8)
        colormap[1] = [128, 0, 0]    # Example class color (adjust as needed)
        colormap[2] = [0, 128, 0]    # Example class color (adjust as needed)
        colormap[3] = [128, 128, 0]  # Example class color (adjust as needed)
        # Add more colors if needed for other classes

        # Convert the segmentation output to a color image
        colored_output = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
        for class_id in np.unique(output):
            colored_output[output == class_id] = colormap[class_id]

        segmented_image = Image.fromarray(colored_output)

        # Convert the segmented image to bytes
        output_bytes = io.BytesIO()
        segmented_image.save(output_bytes, format="PNG")
        output_bytes.seek(0)

        return Response(content=output_bytes.read(), media_type="image/png")

# Deploy the model with Ray Serve
serve.run(DeepLabv3Model.bind(), route_prefix="/segment")