from PIL import Image
import requests
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login

import os
login(token=os.environ.get("HF_TOKEN"))

model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it",device_map="auto")
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it").to(model.device)

prompt = "Where is the cat standing?"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text=prompt,  return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs,)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
