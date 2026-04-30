import json
import os

notebook_path = "C:/Users/91891/.gemini/antigravity/scratch/sagan/sagan_capabilities_showcase.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update code cells
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            # We could change 'formula' to 'composite_formula' here, 
            # but since we added the alias, it's technically not needed.
            # However, the user asked to "make those changes", which usually implies
            # updating the code to match the new preferred way.
            # Let's update it to use 'composite_formula' where appropriate.
            updated_line = line.replace("results_turbo['formula']", "results_turbo['composite_formula']")
            updated_line = updated_line.replace("results_turbo[\"formula\"]", "results_turbo[\"composite_formula\"]")
            new_source.append(updated_line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated sagan_capabilities_showcase.ipynb successfully.")
