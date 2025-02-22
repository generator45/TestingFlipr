from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os
load_dotenv()

model_name = "meta-llama/Llama-2-7b-hf"
hf_token =  os.getenv("hf_token")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=hf_token)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt text to generate from
prompt_text = "What is artificial intelligence?"

# Generate text
result = generator(prompt_text, max_length=50, num_return_sequences=1, truncation=True)

# Print generated text
print(result[0]['generated_text'])