import json
import os

notebook_path = "C:/Users/91891/.gemini/antigravity/scratch/sagan/sagan_capabilities_showcase.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Restore the broken cell
for cell in nb.get('cells', []):
    if cell.get('id') == 'ba412d4f':
        cell['source'] = [
            "# Turbo Profile: Deep Mathematical Discovery\n",
            "reg_turbo = SymbolicRegressor([ticker], signals=signals, profile=\"turbo\")\n",
            "results_turbo = reg_turbo.train()\n",
            "\n",
            "print(f\"Discovered Formula (Turbo): {results_turbo['composite_formula']}\")\n",
            "print(f\"Mean R2 Score: {np.mean(list(results_turbo['r2_stats'].values())):.4f}\")"
        ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Restored and updated sagan_capabilities_showcase.ipynb successfully.")
