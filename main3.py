from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "deepseek-ai/deepseek-1.3b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

prompt_text = "Explain the importance of space exploration."

result = generator(prompt_text, num_return_sequences=1)

print(result[0]['generated_text'])
