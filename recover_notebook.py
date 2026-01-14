import json

# Load the corrupted notebook
with open("broken.json", "r", encoding="utf-8") as f:
    outer = json.load(f)

# Extract the raw cell content
raw_cell = outer["cells"][0]
inner_text = "".join(raw_cell["source"])

# Parse the inner notebook JSON
inner_nb = json.loads(inner_text)

# Save recovered notebook
with open("recovered.ipynb", "w", encoding="utf-8") as f:
    json.dump(inner_nb, f, indent=2)

print("Recovered notebook saved as recovered.ipynb")
