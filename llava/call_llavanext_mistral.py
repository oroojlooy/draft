import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# Load the model in half-precision
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
#model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map="auto")
model = LlavaNextForConditionalGeneration.from_pretrained("./checkpoints/llava-v1.5-13b", torch_dtype=torch.float16, device_map="auto")

# Get three different images
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image_stop = Image.open(requests.get(url, stream=True).raw)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_cats = Image.open(requests.get(url, stream=True).raw)

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image_snowman = Image.open(requests.get(url, stream=True).raw)

# Prepare a batched prompt, where the first one is a multi-turn conversation and the second is not
prompt = [
    "[INST] <image>\nWhat is shown in this image? [/INST] There is a red stop sign in the image. [INST] <image>\nWhat about this image? How many cats do you see [/INST]",
    "[INST] <image>\nWhat is shown in this image? [/INST]"
]

# We can simply feed images in the order they have to be used in the text prompt
# Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
inputs = processor(text=prompt, images=[image_stop, image_cats, image_snowman], padding=True, return_tensors="pt").to(model.device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=300)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(processor.decode(generate_ids[0], skip_special_tokens=True))
