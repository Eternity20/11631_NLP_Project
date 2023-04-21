import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import re
import torch
import transformers
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast
)
from typing import Any, List, Mapping, Tuple
from .QAEvaluator import QAEvaluator
from .QGProjectDataset import QGProjectDataset
#from tqdm import tqdm


class QuestionGenerator:
    """A transformer-based NLP system for generating reading comprehension-style questions from
    texts. It can generate full sentence questions, multiple choice questions, or a mix of the
    two styles.

    To filter out low quality questions, questions are assigned a score and ranked once they have
    been generated. Only the top k questions will be returned. This behaviour can be turned off
    by setting use_evaluator=False.
    """

    def __init__(self) -> None:

        QG_PRETRAINED = "iarfmoose/t5-base-question-generator"
        self.answer_token = "<answer>"
        self.context_token = "<context>"
        self.max_seq_length = 512
        self.stride = 128
        self.truncation=True
        self.padding = "max_length"
        self.return_overflowing_tokens = True
        self.return_offsets_mapping = True

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.qg_tokenizer = T5TokenizerFast.from_pretrained(QG_PRETRAINED)
        self.qg_model = T5ForConditionalGeneration.from_pretrained(QG_PRETRAINED)
        #self.qg_tokenizer = AutoTokenizer.from_pretrained(
        #    "pretrained/qg_tok_ad", use_fast=False)
        #self.qg_model = AutoModelForSeq2SeqLM.from_pretrained('pretrained/qg_model_ad')
        self.qg_model.to(self.device)
        self.qg_model.eval()

        self.qa_evaluator = QAEvaluator()

    def generate(
            self,
            article: str,
            use_evaluator: bool = True,
            num_questions: int = None,
    ) -> List:
        # print("Generating questions...\n")

        dataset = QGProjectDataset(article, self.qg_tokenizer, self.answer_token, self.context_token,
                                   self.max_seq_length, self.stride, self.truncation, self.padding,
                                   self.return_overflowing_tokens, self.return_offsets_mapping)
        #qg_inputs, qg_answers = self.generate_qg_inputs(article, answer_style)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2,
                                                 collate_fn=dataset.filter_empty_paragraphs_collate_fn)

        generated_questions = []
        qg_answers = []
        #for i, (batch_inputs, batch_answers) in tqdm(enumerate(dataloader), desc="Generating questions", total=len(dataloader)):
        #for i, batch_list in tqdm(enumerate(dataloader), desc="Generating questions", total=len(dataloader)):
        for i, batch_list in enumerate(dataloader):
            for (batch_inputs, batch_answers) in batch_list:
                outputs = self.qg_model.generate(input_ids=batch_inputs["input_ids"].to(self.device))
                questions = self.qg_tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
                generated_questions.extend(questions)
                qg_answers.extend(batch_answers)

        message = "{} questions doesn't match {} answers".format(
            len(generated_questions), len(qg_answers)
        )
        assert len(generated_questions) == len(qg_answers), message

        if use_evaluator:
            #print("Evaluating QA pairs...\n")
            encoded_qa_pairs = self.qa_evaluator.encode_qa_pairs(generated_questions, qg_answers)
            scores = self.qa_evaluator.get_scores(encoded_qa_pairs)

            if num_questions:
                qa_list = self._get_ranked_qa_pairs(generated_questions, qg_answers, scores, num_questions)
            else:
                qa_list = self._get_ranked_qa_pairs(generated_questions, qg_answers, scores)
        else:
            #print("Skipping evaluation step.\n")
            qa_list = self._get_all_qa_pairs(generated_questions, qg_answers)
        return qa_list

    def generate_qg_inputs(self, text: str) -> Tuple[List[str], List[str]]:
        """Given a text, returns a list of model inputs and a list of corresponding answers.
        Model inputs take the form "answer_token <answer text> context_token <context text>" where
        the answer is a string extracted from the text, and the context is the wider text surrounding
        the context.
        """
        inputs = []
        answers = []

        segments = self._split_into_segments(text)

        for segment in segments:
            sentences = self._split_text(segment)
            prepped_inputs, prepped_answers = self._prepare_qg_inputs(
                sentences, segment
            )
            inputs.extend(prepped_inputs)
            answers.extend(prepped_answers)

        return inputs, answers

    def generate_questions_from_inputs(self, qg_inputs: List) -> List[str]:
        """Given a list of concatenated answers and contexts, with the form:
        "answer_token <answer text> context_token <context text>", generates a list of
        questions.
        """
        generated_questions = []

        for qg_input in qg_inputs:
            question = self._generate_question(qg_input)
            generated_questions.append(question)

        return generated_questions

    def _split_text(self, text: str) -> List[str]:
        """Splits the text into sentences, and attempts to split or truncate long sentences."""
        MAX_SENTENCE_LEN = 128
        sentences = re.findall(".*?[.!\?]", text)
        cut_sentences = []

        for sentence in sentences:
            if len(sentence) > MAX_SENTENCE_LEN:
                cut_sentences.extend(re.split("[,;:)]", sentence))

        # remove useless post-quote sentence fragments
        cut_sentences = [s for s in sentences if len(s.split(" ")) > 5]
        sentences = sentences + cut_sentences

        return list(set([s.strip(" ") for s in sentences]))

    def _prepare_qg_inputs(
            self,
            sentences: List[str],
            text: str
    ) -> Tuple[List[str], List[str]]:
        """Uses sentences as answers and the text as context. Returns a tuple of (model inputs, answers).
        Model inputs are "answer_token <answer text> context_token <context text>"
        """
        inputs = []
        answers = []

        for sentence in sentences:
            qg_input = f"{self.answer_token} {sentence} {self.context_token} {text}"
            inputs.append(qg_input)
            answers.append(sentence)

        return inputs, answers

    @torch.no_grad()
    def _generate_question(self, qg_input: str) -> str:
        """Takes qg_input which is the concatenated answer and context, and uses it to generate
        a question sentence. The generated question is decoded and then returned.
        """
        encoded_input = self._encode_qg_input(qg_input)
        output = self.qg_model.generate(input_ids=encoded_input["input_ids"])
        question = self.qg_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        return question

    def _encode_qg_input(self, qg_input: str) -> torch.tensor:
        """Tokenizes a string and returns a tensor of input ids corresponding to indices of tokens in
        the vocab.
        """
        return self.qg_tokenizer(
            qg_input,
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _get_ranked_qa_pairs(
            self, generated_questions: List[str], qg_answers: List[str], scores, num_questions: int = 10
    ) -> List[Mapping[str, str]]:
        """Ranks generated questions according to scores, and returns the top num_questions examples.
        """
        if num_questions > len(scores):
            num_questions = len(scores)
            print((
                f"\nWas only able to generate {num_questions} questions.",
                "For more questions, please input a longer text.")
            )

        qa_list = [{"question": generated_questions[index].split("?")[0] + "?", "answer": qg_answers[index]} for index in scores[:num_questions]]

#        for i in range(num_questions):
#            index = scores[i]
#            qa = {
#                "question": generated_questions[index].split("?")[0] + "?",
#                "answer": qg_answers[index]
#            }
#            qa_list.append(qa)

        return qa_list

    def _get_all_qa_pairs(self, generated_questions: List[str], qg_answers: List[str]):
        """Formats question and answer pairs without ranking or filtering."""
        qa_list = [{"question": question.split("?")[0] + "?", "answer": answer} for question, answer in zip(generated_questions, qg_answers)]
        return qa_list


def print_qa(qa_list: List[Mapping[str, str]], show_answers: bool = True) -> None:
    """Formats and prints a list of generated questions and answers."""

    for i in range(len(qa_list)):
        # wider space for 2 digit q nums
        space = " " * int(np.where(i < 9, 3, 4))

        print(f"{qa_list[i]['question']}")

        answer = qa_list[i]["answer"]

        # print a list of multiple choice answers
        if type(answer) is list:

            if show_answers:
                print(
                    f"{space}A: 1. {answer[0]['answer']} "
                    f"{np.where(answer[0]['correct'], '(correct)', '')}"
                )
                for j in range(1, len(answer)):
                    print(
                        f"{space + '   '}{j + 1}. {answer[j]['answer']} "
                        f"{np.where(answer[j]['correct'] == True, '(correct)', '')}"
                    )

            else:
                print(f"{space}A: 1. {answer[0]['answer']}")
                for j in range(1, len(answer)):
                    print(f"{space + '   '}{j + 1}. {answer[j]['answer']}")

            print("")

        # print full sentence answers
        else:
            if show_answers:
                print(f"{space}A: {answer}\n")


if __name__ == '__main__':
    input_file = sys.argv[1]
    N = int(sys.argv[2])
    with open(input_file, encoding="UTF-8") as a:
        article = a.read()
    qg = QuestionGenerator()
    qa_list = qg.generate(
        article,
        num_questions=N,
    )
    print_qa(qa_list, show_answers=False)
    sys.exit(0)