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

if __name__ == '__main__':
    input_file = sys.argv[1]
    N = int(sys.argv[2])
    model = int(sys.argv[3])
    with open(input_file, encoding="UTF-8") as a:
        article = a.read()

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(5):
        if model == 1:
            qg = QuestionGenerator()
            qa_list = qg.generate(
                article,
                use_evaluator=True,
                num_questions=N,
            )
            # print qa
            print_qa(qa_list, show_answers=False)
        else:
            qg_yesNo = YesNoQuestionGenerator(article, N)
            yesNo_questions = qg_yesNo.generate()
            # print yes or no dataset
            for question in yesNo_questions:
                print(f'{question}')
    profiler.disable()
    stats = pstats.Stats(profiler)

    print(f"Average CPU utilization: {stats.total_tt/5} seconds")

    sys.exit(0)