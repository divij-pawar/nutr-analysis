import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dataset import load_data

# Load dataset
df = load_data()

# Prepare input text in the format required for T5
df["input_text"] = df.apply(lambda row: f"Ingredients: {row['ingredients']} Labels: Contains Dairy: {row['contains_dairy']}, High Sugar: {row['high_sugar']}, Vegan: {row['vegan']}, Unhealthy: {row['unhealthy']}", axis=1)

# Initialize tokenizer & model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize inputs
def encode_data(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Train loop
def train_t5(df):
    inputs = encode_data(df["input_text"].tolist())
    labels = encode_data(df["health_description"].tolist())["input_ids"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(3):  # Number of epochs
        optimizer.zero_grad()
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

train_t5(df)
