import torch
import json
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Function to load dataset from JSON file
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data_json = json.load(file)
    # Convert list of dicts into dict of lists
    data = {key: [dic[key] for dic in data_json] for key in data_json[0]}
    return Dataset.from_dict(data)

# Tokenize function adapted for handling 'map' correctly
def tokenize_function(examples):
    model_inputs = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples['output_text'], padding="max_length", truncation=True, max_length=512)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load and process the dataset
dataset = load_dataset('recipe_data/encoded_recipes_full.json')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the eos_token as the pad_token
tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Prepare the data for training by creating train and validation splits
train_dataset = tokenized_datasets.shuffle(seed=42).select(range(800))  # Adjust the range based on your data size
eval_dataset = tokenized_datasets.shuffle(seed=42).select(range(800, 1000))  # Adjust the range based on your data size

# Setup training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Load model and move to GPU if available
device =  "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model_path = "./finetuned_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

