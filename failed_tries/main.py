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
text = '''The last time India and Pakistan clashed in a major ICC 50-over contest was in 2023, at the highly anticipated World Cup league match in Ahmedabad.

As a contest, it turned out to be a bit of an anti-climax - India, in dominant form, comfortably chased down a subpar Pakistan target, securing a resounding victory.

And as Pakistani fans didn't get visas to travel to India, aside from the cricket team, the country's only notable presence was in the media centre.

Sunday's ICC Champions Trophy clash between the arch-rivals at Dubai International Stadium promises a vastly different atmosphere.

The International Cricket Council (ICC) reported that tickets sold out within minutes - and with the UAE hosting more than 3.7 million Indians and nearly 1.7 million Pakistanis, a vibrant and well-represented crowd from both nations is all but guaranteed.

But can a sea of green flags in the stands inspire Pakistan's Mohammad Rizwan's men to defy the odds in this must-win clash against Rohit Sharma's India?

Pakistan can take comfort in their strong head-to-head record in UAE - 19 wins in 28 ODIs, plus a lone T20I victory in the 2021 World Cup in Dubai.'''
prompt_text = f'summarize this text:{text}'

# Generate text
result = generator(prompt_text, num_return_sequences=1)

# Print generated text
print(result[0]['generated_text'])