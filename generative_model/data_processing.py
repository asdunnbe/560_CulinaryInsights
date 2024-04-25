import pandas as pd
import pickle

# Load the recipe data
pp_recipes_path = '../recipe_data/PP_recipes.csv'
recipes_df = pd.read_csv(pp_recipes_path)

# Load the ingredients mapping
with open('../recipe_data/ingr_map.pkl', 'rb') as f:
    ingr_map = pickle.load(f)

# Display the first few rows of each dataframe and the ingredient map
# print(recipes_df.head())
# print(ingr_map.head())

# Check the columns in ingr_map
print("Columns in ingr_map:", ingr_map.columns)
# print(ingr_map.head)


# Function to convert ingredient IDs back to names using ingr_map
def id_to_name(id_list, ingr_map):
    names = []
    for i in id_list:
        # Assuming the mapping is in a column 'ingredient_name'
        name = ingr_map[ingr_map['id'] == i]['raw_words'].values[0]
        names.append(name)
    return names

# Adding a column with ingredient names to the recipes dataframe
recipes_df['ingredient_names'] = recipes_df['ingredient_ids'].apply(lambda x: id_to_name(eval(x), ingr_map))

# Preparing the training data
recipes_df['input_output'] = recipes_df.apply(lambda row: (row['ingredient_names'], row['steps']), axis=1)

# Display a sample input and output
print(recipes_df['input_output'].head())

# Example function to minimize additional ingredients
def minimize_ingredients(input_ingredients, available_ingredients):
    # Filter out the ingredients already available
    needed_ingredients = [ing for ing in input_ingredients if ing not in available_ingredients]
    return needed_ingredients

# Assuming 'user_ingredients' is a list of available ingredients
user_ingredients = ['salt', 'pepper', 'chicken']  # Example user ingredients
recipes_df['minimized_ingredients'] = recipes_df['ingredient_names'].apply(lambda x: minimize_ingredients(x, user_ingredients))

# Display the updated dataframe
print(recipes_df[['minimized_ingredients', 'steps']].head())
