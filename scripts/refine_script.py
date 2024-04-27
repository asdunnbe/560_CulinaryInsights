import pandas as pd
import pickle
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time


start_time = time.time()

# Load the recipe data
pp_recipes_path = 'data/RAW_recipes.csv'
df = pd.read_csv(pp_recipes_path)

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# for each row in csv. 

df['ingredients_list'] = df['ingredients'].apply(lambda x: eval(x) if isinstance(x, str) else x)
df['name'] = df['name'].astype(str).str.strip()  # Convert to string and strip any whitespace


unique_ingredients = set(ingredient for sublist in df['ingredients_list'] for ingredient in sublist)



def rand_ing_full() :

    return random.choice(list(unique_ingredients))




inputs = []
outputs = []


# takes in a list of ingredients and outputs an array of 6 with variability 
def input_variability(ingredients: list, var: int) :
    holder = []
    holder.append(ingredients)
    length = len(ingredients)
    less_ing = ingredients[:]
    more_ing = ingredients[:]
    i = 0
    ## subtractive

    for _ in range(var) :
        
        print("lessing:", less_ing)
        less_ing.remove(random.choice(less_ing))
        
        holder.append(less_ing[:])

        if (len(less_ing) == 0) :
            print("ingredient list runs out")
            break



    ##additive 

    for _ in range(var) :
        more_ing.append(rand_ing_full())
        holder.append(more_ing[:])


    return holder



##print(input_variability(['cheese','nuts','ice'], 2))



def generate_input(ingredients):
    """ Generate a natural language query asking what to make with a list of ingredients. """
    # Format the ingredient list into a natural language string
    if len(ingredients) > 3:
        # List the first three and add 'and other ingredients' for longer lists
        formatted_ingredients = ', '.join(ingredients[:3]) + ', and other ingredients'
    else:
        # Join all with 'and' for the last item
        formatted_ingredients = ', '.join(ingredients[:-1]) + ' and ' + ingredients[-1] if len(ingredients) > 1 else ingredients[0]

    # Template for the query
    query = f"I want to make something with {formatted_ingredients}, what should I make?"

    return query



def generate_output(name, ingredients, steps):
    # Sentence Components
    titles = [f"Today, we're making {name}", f"Get ready to cook {name}", f"Let's prepare a delicious {name}"]
    ingredient_intros = [
        "The ingredients you'll need are:", 
        "Gather these ingredients:",
        "You will need the following items:",
        "Make sure you have all these ingredients:"
    ]
    step_intros = [
        "Here are the steps:", 
        "Follow these instructions:",
        "Let's get cooking:",
        "The cooking steps are as follows:"
    ]
    adjectives = ["delicious", "wonderful", "scrumptious", "tasty"]
    verbs = ["enjoy", "savor", "appreciate", "relish"]
    conclusions = [
        f"Hope you {random.choice(verbs)} this {random.choice(adjectives)} dish!",
        "Enjoy your meal!",
        "That's all it takes to make this fabulous dish!"
    ]

    # Select random elements
    title = random.choice(titles)
    ing_intro = random.choice(ingredient_intros)
    step_intro = random.choice(step_intros)
    conclusion = random.choice(conclusions)

    # Generate ingredients list
    ingredients_text = ing_intro + " " + ", ".join(ingredients) + "."

    # Generate steps text
    steps_text = step_intro + " " + " ".join(f"Step {i+1}: {step}" for i, step in enumerate(steps))

    # Assemble the full text
    full_text = f"{title}\n\n{ingredients_text}\n\n{steps_text}\n\n{conclusion}"
    
    return full_text

# Example usage
#name = "Vegetable Stir Fry"
#ingredients = ["1 cup of sliced carrots", "2 cups of broccoli", "1 bell pepper", "2 tablespoons of soy sauce"]
#steps = ["Slice all your vegetables.", "Heat oil in a pan over medium heat.", "Add vegetables and stir fry for 10 minutes.", "Pour soy sauce and stir well."]



#input data into new csv as rows. 
#eval_rows is the number of rows from the original data to expand. var = (factor - 1)/2 AKA the number to replicate inputs in each direction (+ and -)

def generate_data(eval_rows, var) : 

    for i in range(eval_rows) :

        input_ingredients = eval(df.iloc[i]['ingredients']) if isinstance(df.iloc[i]['ingredients'], str) else df.iloc[i]['ingredients']

        input_name = df.iloc[i]['name']
        input_steps = eval(df.iloc[i]['steps']) if isinstance(df.iloc[i]['steps'], str) else df.iloc[i]['steps']


        input_variability(input_ingredients, var)

        for j in range(var*2+1) :
            output = generate_output(input_name, input_ingredients, input_steps) 
            outputs.append(output)






generate_data(1, 2)




new_df = pd.DataFrame({
    'input' : inputs,
    'output' : outputs
})

new_df.to_csv('data/output.csv', index=False)



end_time = time.time()


duration = end_time - start_time
print(f"Execution time: {duration} seconds")
