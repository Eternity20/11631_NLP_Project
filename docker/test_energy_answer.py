#!/usr/bin/python3 -W ignore::DeprecationWarning
# -*- coding:utf8 -*-
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(level=logging.CRITICAL)
from pyJoules.energy_meter import measure_energy

import sys
from typing import List, Tuple, Any
from torch.utils.data import DataLoader
from mydatasets.QAProjectDataset import QAProjectDataset
from mymodels.QAProjectModelWH import QAProjectModelWH
from mymodels.QAProjectModelYN import QAProjectModelYN
from mymodels.question_type_detector import QuestionDetector
from utils import set_device

QA_WH_MODEL = 'deepset/roberta-large-squad2'
QA_YN_MODEL = 'gsgoncalves/roberta-base-boolq'

@measure_energy
def answer_questions(wiki_doc, questions, loaded_conf_dict):
    device = set_device()
    # Let's first split the questions into yes/no and wh-questions
    question_type_detector = QuestionDetector()
    yn_questions = []
    wh_questions = []
    order_dict = {'yn': [], 'wh': []}
    # for i, q in enumerate(questions):
    #     _type = question_type_detector.detect(q)
    #     if _type == question_type_detector.WH_FLAG:
    #         wh_questions.append(q)
    #         order_dict['wh'].append(i)
    #     else:
    #         yn_questions.append(q)
    #         order_dict['yn'].append(i)
    for i, q in enumerate(questions):
        wh_questions.append(q)
        order_dict['wh'].append(i)

    predicted_answers = [''] * len(questions)
    # Let's first answer the yes/no questions
    # qa_model = QAProjectModelYN(QA_YN_MODEL, device)
    # qa_dataset = QAProjectDataset(wiki_doc, yn_questions, qa_model.tokenizer, max_length=512, stride=0, truncation="only_second", padding="max_length", return_overflowing_tokens=False, return_offsets_mapping=False)
    # qa_dataloader = DataLoader(qa_dataset, batch_size=qa_model.batch_size, num_workers=2, pin_memory=True)
    # pred_answers = qa_model.qa_inference(qa_dataloader)
    # for i in range(len(pred_answers)):
    #     predicted_answers[order_dict['yn'][i]] = pred_answers[i]

    # # finally the wh-questions
    qa_model = QAProjectModelWH.from_config_dict(QA_WH_MODEL, device, loaded_conf_dict)
    qa_dataset = QAProjectDataset.from_config_dict(wiki_doc, wh_questions, qa_model.tokenizer, loaded_conf_dict)
    qa_dataloader = DataLoader(qa_dataset, batch_size=qa_model.batch_size, num_workers=2, pin_memory=True)

    start_logits, end_logits = qa_model.qa_inference(qa_dataloader)
    pred_answers = qa_model.post_processing(wh_questions, qa_dataset.wiki_doc_str, qa_dataset.tokenized_data, start_logits,
                                            end_logits)
    for i in range(len(pred_answers)):
        predicted_answers[order_dict['wh'][i]] = pred_answers[i]
    return predicted_answers


def read_files(wiki_file_path: str, questions_file_path: str) -> Tuple[List[str], List[str], Any]:
    config_dict = {"max_length": 512, "truncation": "only_second", "padding": "max_length",
                   "return_overflowing_tokens": True, "return_offsets_mapping": True, "stride": 256, "n_best_size": 50,
                   "max_answer_length": 100, "batch_size": 4}
    # with open(config_file_path, 'r') as f:
    #  config_dict = json.load(f)

    wiki_doc = []
    with open(wiki_file_path, 'r') as f:
        for line in f:
            wiki_doc.append(line.strip())

    questions = []
    with open(questions_file_path, 'r') as f:
        for line in f:
            questions.append(line.strip())
    return wiki_doc, questions, config_dict


if __name__ == '__main__':
    wiki_file_path = sys.argv[1]
    questions_file_path = sys.argv[2]

    wiki_doc, questions, config_dict = \
        read_files(wiki_file_path, questions_file_path)

    for i in range(5):
        predicted_answers = answer_questions(wiki_doc, questions, config_dict)
        for answer in predicted_answers:
            print(f"{answer['prediction_text']}")


    sys.exit(0)