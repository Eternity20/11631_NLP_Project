from test_model_size import print_size_of_model
from transformers import RobertaForSequenceClassification

model = RobertaForSequenceClassification.from_pretrained("yn_qa_roberta_base",local_files_only = True)
print("-----------qa_model_boolean-------------")
print_size_of_model(model)
