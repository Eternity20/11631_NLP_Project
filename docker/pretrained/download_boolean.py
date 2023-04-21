from transformers import T5ForConditionalGeneration, T5Tokenizer

# Instantiate a model and tokenizer
model_name = 'ramsrigouthamg/t5_boolean_questions'
token_name = 't5-base'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(token_name)


# Save the model and tokenizer
model.save_pretrained("qg_model_boolean")
tokenizer.save_pretrained("qg_tok_boolean")