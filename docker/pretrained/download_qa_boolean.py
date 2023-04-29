from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from test_model_size import print_size_of_model

# Instantiate a model and tokenizer
model_name = 'gsgoncalves/roberta-base-boolq'
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Save the model and tokenizer
model.save_pretrained("qa_model_yn")
tokenizer.save_pretrained("qa_tok_yn")

print("------------qa_model_boolean---------------")
print_size_of_model(model)
