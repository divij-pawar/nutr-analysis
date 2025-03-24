import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Load dataset
def load_data(file_path="data/en.openfoodfacts.org.products.csv"):
    df = pd.read_csv(file_path,on_bad_lines='skip')
    df.dropna(inplace=True)
    return df

# Preprocess text
def preprocess_data(df, tokenizer_name="bert-base-uncased", max_length=128):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def tokenize_text(text):
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    df["tokens"] = df["ingredients"].apply(tokenize_text)
    return df

# Train-test split
def prepare_data(file_path="data/en.openfoodfacts.org.products.csv"):
    df = load_data(file_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return preprocess_data(train_df), preprocess_data(val_df)
