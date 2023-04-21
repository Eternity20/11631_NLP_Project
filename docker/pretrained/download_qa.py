from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Instantiate a model and tokenizer
model_name = 'deepset/roberta-large-squad2'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer
model.save_pretrained("qg_model")
tokenizer.save_pretrained("qg_tok")