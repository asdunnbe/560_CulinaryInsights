''' The current version of this takes a random sample from the RAW recipe dataset, then converts the way 
    'ingredients' and 'steps' are stored so as to give you a single string for each, making it easier to 
    use the ingredients in a prompt, and making it easier to use the steps as part of a ROUGE score evaluation'''

import pandas as pd
from ast import literal_eval
#pip install rouge if needed
from rouge import Rouge

rouge = Rouge()

#CHANGE TO YOUR FILEPATH
recipes_path = '../RAW_recipes.csv'
df = pd.read_csv(recipes_path)

#Taking a sample of only 3 recipes
df = df.sample(n=3, random_state=22)

testing_recs = df.loc[:,['id', 'name', 'ingredients', 'steps']]

#Below code converts the list of strings representation of 'ingredients' and 'steps' to a single string each
testing_recs.ingredients = testing_recs.ingredients.apply(literal_eval)
testing_recs.steps = testing_recs.steps.apply(literal_eval)

for index, rec in testing_recs.iterrows():
    ingString = ', '.join(rec.ingredients)
    stepsString = ' '.join(rec.steps)

    testing_recs.loc[index,['steps']] = stepsString
    testing_recs.loc[index, ['ingredients']] = ingString

print(testing_recs)

''' 
    
    From here you might want to use the ingredient list ( stored in testing_recs.iloc[INDEX].ingredients ) as part of a prompt to generate recipe steps.

    Then, you will want to compare the generated output steps to the steps string for the pre-existing recipe ( stored in testing_recs.iloc[INDEX].steps )
    
    The way this comparison is done is by using the rouge.get_scores() function below. The inputs are:

                        rouge.get_score(GENERATED_RECIPE_STEPS, PREEXISTING_RECIPE_STEPS)

    There is also a way to compute the ROUGE scores for multiple outputs, and you can even have it output an average if setting avg=True in the function call.
    Either way, please reference the documentation here: https://github.com/pltrdy/rouge
    
    The below code is a basic framework for getting a score (regardless of if fed lists of steps or a single string of steps)

'''

generated_steps = 'Model Recipe Steps output for given input'
preexisting_steps = 'Ground-truth Recipe Steps for the input given to the model'

def rouge_score(generated_steps, preexisting_steps):
    if (isinstance(generated_steps, list) and isinstance(preexisting_steps, list)):
        return rouge.get_score(generated_steps, preexisting_steps, avg=True)
    else:
        return rouge.get_score(generated_steps, preexisting_steps)

#scores = rouge_score(GENERATED, PREEXISTING)