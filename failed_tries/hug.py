from transformers import pipeline

pipe = pipeline("text-to-image", model="runwayml/stable-diffusion-v1-5")

image = pipe("A fantasy landscape with castles and dragons")  
image[0].save("output.png")
