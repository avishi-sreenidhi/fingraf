import json
import re

# Load raw data
with open("data/kaggle/financial_advice.json", "r") as f:
    advice_data = json.load(f)

def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces/newlines
    return text

processed_data = []
for item in advice_data:
    processed_item = {
        "about_me": clean_text(item["about_me"]),
        "context": clean_text(item["context"]),
        "response": clean_text(item["response"])
    }
    processed_data.append(processed_item)

# Save processed data
with open("data/processed/cleaned_advice.json", "w") as f:
    json.dump(processed_data, f, indent=4)

print(f"Processed {len(processed_data)} advice entries and saved cleaned data.")
