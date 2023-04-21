from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoModelForSeq2SeqLM

# Instantiate a model and tokenizer
QG_PRETRAINED = "iarfmoose/t5-base-question-generator"

qg_tokenizer = AutoTokenizer.from_pretrained(
            QG_PRETRAINED, use_fast=False)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)

QAE_PRETRAINED = "iarfmoose/bert-base-cased-qa-evaluator"
qae_tokenizer = AutoTokenizer.from_pretrained(QAE_PRETRAINED)
qae_model = AutoModelForSequenceClassification.from_pretrained(QAE_PRETRAINED)


# Save the model and tokenizer
qg_model.save_pretrained("qg_model_ad")
qg_tokenizer.save_pretrained("qg_tok_ad")

qae_model.save_pretrained("qae_model_ad")
qae_tokenizer.save_pretrained("qae_tok_ad")

