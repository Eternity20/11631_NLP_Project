from test_model_size import print_size_of_model
from transformers import AutoTokenizer,AutoModelForQuestionAnswering

config = {
    'model_checkpoint': "deepset/roberta-base-squad2",
    "max_length": 308,
    "truncation": "only_second",
    "padding": True,
    "return_overflowing_tokens": True,
    "return_offsets_mapping": True,
    "stride": 200,
    "n_best_size": 3,
    "max_answer_length": None,
    "batch_size": 64
}
tokenizer = AutoTokenizer.from_pretrained(config['model_checkpoint'])
qa_model = AutoModelForQuestionAnswering.from_pretrained(config['model_checkpoint'], return_dict = True)

qa_model.save_pretrained("qa_baseline_model")
print_size_of_model(qa_model)
