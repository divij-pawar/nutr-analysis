import torch
from transformers import BertForSequenceClassification, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer

# Load models
bert_model = BertForSequenceClassification.from_pretrained("./models/bert_lora")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Prediction function
def classify_ingredients(ingredient_list):
    inputs = bert_tokenizer(ingredient_list, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    logits = outputs.logits
    predicted_labels = torch.sigmoid(logits).round().tolist()[0]
    
    label_map = ["Contains Dairy", "High Sugar", "Vegan", "Unhealthy"]
    predictions = {label: bool(pred) for label, pred in zip(label_map, predicted_labels)}
    return predictions

def generate_health_summary(ingredients, classifications):
    input_text = f"Ingredients: {ingredients} Labels: {classifications}"
    input_ids = t5_tokenizer(input_text, return_tensors="pt").input_ids
    output = t5_model.generate(input_ids)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
ingredients = "Milk, Sugar, Cocoa, Palm Oil"
classifications = classify_ingredients(ingredients)
summary = generate_health_summary(ingredients, classifications)

print("Classifications:", classifications)
print("Health Summary:", summary)
