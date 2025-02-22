import requests
import os
from dotenv import load_dotenv
import io
from PIL import Image

# Load environment variables from .env file
load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/ZB-Tech/Text-to-Image"
headers = {"Authorization": f"Bearer {os.getenv('hf_token')}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for bad status codes
    return response.content

image_bytes = query({
    "inputs": "dark knight rises", 
})

# You can access the image with PIL.Image for example
image = Image.open(io.BytesIO(image_bytes))

# Save the image
image.save("generated_image.png")

print("Image saved as generated_image.png")