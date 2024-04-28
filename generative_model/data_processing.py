import pandas as pd
import pickle
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import json

def rand_ing_full() :
    return random.choice(list(unique_ingredients))


# takes in a list of ingredients and outputs an array of 6 with variability 
def input_variability(ingredients: list, var: int) :
    input_possiblities = []
    input_possiblities.append(ingredients)

    ## subtractive
    less_ing = ingredients[:]
    for _ in range(var) :
        less_ing.remove(random.choice(less_ing))
        if (len(less_ing) == 0) : break
        input_possiblities.append(less_ing[:])

    ##additive 
    more_ing = ingredients[:]
    for _ in range(var) :
        more_ing.append(rand_ing_full())
        input_possiblities.append(more_ing[:])

    return input_possiblities


def generate_input(ingredients):
    """ Generate a natural language query asking what to make with a list of ingredients. """
    # Format the ingredient list into a natural language string
    if len(ingredients) > 3:
        # List the first three and add 'and other ingredients' for longer lists
        formatted_ingredients = ', '.join(ingredients) + ', and other ingredients'

    elif len(ingredients) == 1:
        formatted_ingredients = ingredients[0]
    else:
        # Join all with 'and' for the last item
        formatted_ingredients = ', '.join(ingredients[:-1]) + ' and ' + ingredients[-1] 

    # Template for the query
    query = f"I want to make something with {formatted_ingredients}, what should I make?"

    return query


def generate_output(name, ingredients, steps):
    # Sentence Components
    titles = [
        f"Today, we're making {name}. ", 
        f"Get ready to cook {name}. ", 
    ]
    ingredient_intros = [
        "The ingredients you'll need are: ", 
        "You will need the following items: ",
        "Make sure you have all these ingredients: "
    ]
    step_intros = [
        "Here are the steps.", 
        "The cooking steps are as follows."
    ]
    # adjectives = ["delicious", "wonderful", "scrumptious", "tasty"]
    # verbs = ["enjoy", "savor", "appreciate", "relish"]
    conclusions = [
        f"Yum!",
        "Enjoy your meal!",
        "That's all it takes to make this fabulous dish!"
    ]

    # Select random elements
    title = random.choice(titles)
    ing_intro = random.choice(ingredient_intros)
    step_intro = random.choice(step_intros)
    conclusion = random.choice(conclusions)

    # Generate ingredients list
    ingredients_text = ing_intro + ", ".join(ingredients) + "."

    # Generate steps text
    steps_text = step_intro + " " + " ".join(f"Step {i+1}: {step}" for i, step in enumerate(steps))

    # Assemble the full text
    full_text = f"{title}{ingredients_text}{steps_text}{conclusion}"
    
    return full_text


def generate_output(name, ingredients, steps):
    # Sentence Components
    titles = [
        f"Today, we're making {name}. ", 
        f"Get ready to cook {name}. ", 
    ]
    ingredient_intros = [
        "The ingredients you'll need are: ", 
        "You will need the following items: ",
        "Make sure you have all these ingredients: "
    ]
    conclusions = [
        f"Yum!",
        "Enjoy your meal!",
        "That's all it takes to make this fabulous dish!"
    ]

    # Select random elements
    title = random.choice(titles)
    ing_intro = random.choice(ingredient_intros)
    conclusion = random.choice(conclusions)

    # Generate ingredients list
    ingredients_text = ing_intro + ", ".join(ingredients) + "."

    # Assemble the full text
    full_text = f"{title}{ingredients_text}{conclusion}"
    
    return full_text


def combine_input_output(name, ingredients, steps):
    input = generate_input(ingredients)
    output = generate_output(name, ingredients, steps)
    return input, output


def entries_per_row(row, count):
    name = row['name']
    ingredients = row['ingredients_list']
    steps = eval(row['steps'])

    input_possiblities = input_variability(ingredients, 3)

    list_entries = []
    for _ in range(min(count, len(input_possiblities))):
        input_ing = random.choice(input_possiblities)
        input_possiblities.remove(input_ing)
        input, output = combine_input_output(name, input_ing, steps)
        list_entries.append({'input_text': input, 
                             'output_text': output,
                             'input_ids': encode_texts(input), 
                             'output_ids': encode_texts(output)
                            })
        
    return list_entries

# Function to encode texts
def encode_texts(text):
    return tokenizer.encode(text, truncation=True, max_length=1024)

# Load the recipe data
pp_recipes_path = 'recipe_data/RAW_recipes.csv'
df = pd.read_csv(pp_recipes_path)

# Load pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

df['ingredients_list'] = df['ingredients'].apply(lambda x: eval(x) if isinstance(x, str) else x)
df['name'] = df['name'].astype(str).str.strip()  # Convert to string and strip any whitespace
df = df.drop(columns=['id', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition','n_steps','description','n_ingredients'])

start_time = time.time()

unique_ingredients = set(ingredient for sublist in df['ingredients_list'] for ingredient in sublist)

new_list = []
count = 6
for _, row in df.iterrows():
    entries = entries_per_row(row, count)
    for e in entries: new_list.append(e)

new_df = pd.DataFrame(new_list)


# Load pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


new_df.to_csv('recipe_data/output_full_simplied.csv', index=False)

# Write to a JSON file
with open('/recipe_data/encoded_recipes_full_simplified.json', 'w') as file:
    json.dump(new_list, file, indent=4)

end_time = time.time()


duration = end_time - start_time
print(f"Execution time: {duration} seconds")