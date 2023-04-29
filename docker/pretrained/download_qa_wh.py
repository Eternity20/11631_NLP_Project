from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
from test_model_size import print_size_of_model

# Instantiate a model and tokenizer
model_name = 'deepset/roberta-large-squad2'
model = RobertaForQuestionAnswering.from_pretrained(model_name)
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

# Save the model and tokenizer
model.save_pretrained("qa_model_wh")
tokenizer.save_pretrained("qa_tok")

print("------------qa_model_wh---------------")
print_size_of_model(model)
