from transformers import BertTokenizerFast,BertForSequenceClassification,T5TokenizerFast,T5ForConditionalGeneration
from test_model_size import print_size_of_model
# Instantiate a model and tokenizer

QAE_PRETRAINED = "iarfmoose/bert-base-cased-qa-evaluator"
qae_tokenizer = BertTokenizerFast.from_pretrained(QAE_PRETRAINED)
qae_model =  BertForSequenceClassification.from_pretrained(QAE_PRETRAINED)

qae_model.save_pretrained("qae_model_ad")
qae_tokenizer.save_pretrained("qae_tok_ad")

QG_PRETRAINED = "iarfmoose/t5-base-question-generator"
qg_tokenizer = T5TokenizerFast.from_pretrained(QG_PRETRAINED)
qg_model = T5ForConditionalGeneration.from_pretrained(QG_PRETRAINED)
qg_model.save_pretrained("qg_model_ad")
qg_tokenizer.save_pretrained("qg_tok_ad")

print("-------------qae_model--------------")
print_size_of_model(qae_model)
print("-------------qg_model_wh---------------")
print_size_of_model(qg_model)



