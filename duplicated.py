import json
import pandas as pd
from collections import defaultdict

def export_duplicates_to_excel(recipe_data, output_filename="duplicate_recipes.xlsx"):
    # Step 1: Group all recipes by their cleaned title
    grouped_recipes = defaultdict(list)
    
    for recipe in recipe_data:
        title = recipe.get("title", "").strip()
        grouped_recipes[title].append(recipe)
        
    # Step 2: Extract only the groups that have more than 1 entry
    duplicates_extracted = []
    
    for title, group in grouped_recipes.items():
        if len(group) > 1:
            for recipe in group:
                # Join lists into strings with newlines so they format cleanly in Excel cells
                categories_str = ", ".join(recipe.get("categories", []))
                ingredients_str = "\n".join(recipe.get("ingredients", []))
                directions_str = "\n".join(recipe.get("directions", []))
                
                duplicates_extracted.append({
                    "Title": title,
                    "Categories": categories_str,
                    "Ingredients": ingredients_str,
                    "Directions": directions_str
                })
                
    # Step 3: Convert to a Pandas DataFrame and save to Excel
    if not duplicates_extracted:
        print("No duplicates found.")
        return

    df = pd.DataFrame(duplicates_extracted)
    
    # Write to Excel
    df.to_excel(output_filename, index=False, engine='openpyxl')
    print(f"Successfully saved {len(duplicates_extracted)} duplicate entries to {output_filename}")

# --- Example Usage ---

# 1. Load your JSON data
with open('full_format_recipes.json', 'r', encoding='utf-8') as file:
    raw_recipes = json.load(file)

# 2. Run the function
export_duplicates_to_excel(raw_recipes, "duplicate_recipes.xlsx")