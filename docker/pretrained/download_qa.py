from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
# Instantiate a model and tokenizer
model_name = 'deepset/roberta-large-squad2'
model = RobertaForQuestionAnswering.from_pretrained(model_name)
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

# Save the model and tokenizer
model.save_pretrained("qa_model")
tokenizer.save_pretrained("qa_tok")