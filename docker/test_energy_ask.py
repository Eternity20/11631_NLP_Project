#!/usr/bin/python3 -W ignore::DeprecationWarning
# -*- coding:utf8 -*-
import cProfile
import pstats
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(level=logging.CRITICAL)
import sys
from Asking.t5_boolean import YesNoQuestionGenerator
from Asking.advance_qg import QuestionGenerator
from Asking.advance_qg import print_qa
from pyJoules.energy_meter import measure_energy
@measure_energy
def QG1(N,article):
    qg = QuestionGenerator()
    qa_list = qg.generate(
        article,
        use_evaluator=True,
        num_questions=N,
    )
    # print qa
    print_qa(qa_list, show_answers=False)

@measure_energy
def yes_no(N,article):
    qg_yesNo = YesNoQuestionGenerator(article, N)
    yesNo_questions = qg_yesNo.generate()
    # print yes or no dataset
    for question in yesNo_questions:
        print(f'{question}')

if __name__ == '__main__':
    input_file = sys.argv[1]
    N = int(sys.argv[2])
    model = int(sys.argv[3])
    with open(input_file, encoding="UTF-8") as a:
        article = a.read()

    for i in range(5):
        if model == 1:
            QG1(N, article)
        else:
            yes_no(N, article)

    sys.exit(0)