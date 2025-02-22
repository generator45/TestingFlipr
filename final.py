import requests
import os
from dotenv import load_dotenv
import io
from PIL import Image

# Load environment variables from .env file
load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
hf_token = os.getenv('hf_token')
if not hf_token:
    raise ValueError("Hugging Face token not found. Please set it in the .env file.")
headers = {"Authorization": f"Bearer {hf_token}"}
# print(hf_token)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for bad status codes
    return response.content

image_bytes = query({
    "inputs": "horse",
})

image = Image.open(io.BytesIO(image_bytes))

# Save the image
image.save("generated_image.png")
print("Image saved as generated_image.png")