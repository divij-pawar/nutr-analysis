# Nutrition Chatbot using Open Food Facts and Zephyr-7B

This project implements an interactive chatbot that provides nutritional insights for food products using data from [Open Food Facts](https://world.openfoodfacts.org/) and a Hugging Face instruction-tuned model (`zephyr-7b-beta`). The system analyzes ingredients, nutrients, known allergens, and unhealthy additives using a combination of structured metadata and natural language prompts.

## Features

- Answer user questions about nutrients, ingredients, and labels for real-world food products.
- Supports keyword search with multiple matching product selection.
- Detects and highlights unhealthy ingredients using a JSON-based rule file.
- Flags common allergens based on ingredient text.
- Uses a Gradio web interface with:
  - Search field
  - Product dropdown
  - Custom questions
  - Structured, model-generated answers
- Optimized prompt handling for stability and speed on GPU.

## Requirements

- Python 3.8+
- GPU (e.g. NVIDIA T4, A100 for Zephyr-7B)
- Hugging Face Transformers
- Gradio
- Pandas
- Accelerate (for efficient model loading)

## Setup

Install dependencies:

```bash
pip install transformers accelerate pandas gradio
```

Download the dataset:

```bash
curl -L -o data.csv.gz "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
gunzip data.csv.gz
```

Prepare the unhealthy ingredient mapping:

- Store a file named `unhealthy_ingredients.json` with entries like:

```json
[
  {
    "ingredient": "High-Fructose Corn Syrup",
    "aliases": ["HFCS"],
    "reason": "Linked to obesity, insulin resistance, and liver fat buildup."
  },
  {
    "ingredient": "Aspartame",
    "aliases": [],
    "reason": "Controversial artificial sweetener; possible link to cancer."
  }
]
```

## Running the Application

To launch the chatbot:

```bash
python nutrition_chatbot.py
```

Or in a Jupyter/Colab notebook, call:

```python
demo.launch()
```

The application includes:

- A text field to enter product keywords (e.g., "chocolate", "soup")
- A dropdown to choose from matching products
- A text box to ask questions like:
  - "How much sugar is in this product?"
  - "Are there any unhealthy ingredients?"
  - "Is it suitable for someone with a peanut allergy?"

## Architecture Overview

- **Model**: `HuggingFaceH4/zephyr-7b-beta`, loaded with GPU acceleration.
- **Data**: Open Food Facts TSV product database.
- **UI**: Gradio Blocks-based interface with multi-step interaction.
- **Prompt**: Instruction format tailored for nutrition Q&A.
- **Logic**:
  - Ingredient text is scanned for known allergens and flagged additives.
  - Context is inserted into a structured prompt passed to the LLM.
  - Token count is capped to prevent overload.


## License

This project is open for educational and research purposes. Be sure to comply with Hugging Face model license terms and Open Food Facts usage guidelines.
