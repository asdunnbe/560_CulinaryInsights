from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Function to load the model and tokenizer
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

# Load your fine-tuned model and tokenizer
model_path = './finetuned_model'  # Adjust path as necessary
model, tokenizer = load_model(model_path)

# Setup the pipeline for text generation
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1)  # device=0 for GPU; use device=-1 for CPU

# Function to generate text based on a prompt
def generate_text(prompt, max_length=500):
    generated_texts = text_generator(prompt, max_length=max_length, num_return_sequences=8)
    return generated_texts[0]['generated_text']

# Example usage
user_prompt = "I want to make something with flour, sugar, eggs, and pears, what should I make?"
generated_recipe = generate_text(user_prompt)
print(generated_recipe)
