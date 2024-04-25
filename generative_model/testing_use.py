import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" # the device to load the model onto MAC GPU, use 'cuda' for nvidia or 'cpu' for not gpu

model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "Give me a recipe for chicken and rice"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
response = tokenizer.batch_decode(generated_ids)[0]
print('Input prompt:', prompt)
print('Output:      ', response[len(prompt)+2:])
print('Have a nice day!')