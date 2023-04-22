import warnings
warnings.filterwarnings('ignore')
import torch
import transformers
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
from typing import List
from transformers import BertTokenizerFast, BertForSequenceClassification

class QAEvaluator:
    """Generate score for a QA pair that used to rank and filter.
    """

    def __init__(self) -> None:

        #QAE_PRETRAINED = "iarfmoose/bert-base-cased-qa-evaluator"
        self.SEQ_LENGTH = 256

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.qae_tokenizer = BertTokenizerFast.from_pretrained("pretrained/qae_tok_ad",local_files_only = True)
        self.qae_model = BertForSequenceClassification.from_pretrained(
            "pretrained/qae_model_ad",local_files_only = True
        )
        self.qae_model.to(self.device)
        self.qae_model.eval()

    def encode_qa_pairs(self, questions: List[str], answers: List[str]) -> List[torch.tensor]:
        """
        Input: questions and answers 
        Output: encoded tensors."""
        encoded_pairs = []

        for question, answer in zip(questions, answers):
            encoded_qa = self._encode_qa(question, answer)
            encoded_pairs.append(encoded_qa.to(self.device))

        return encoded_pairs

    def get_scores(self, encoded_qa_pairs: List[torch.tensor]) -> List[float]:
        """Generates scores for the encoded QA pairs."""
        scores = {}

        for i in range(len(encoded_qa_pairs)):
            scores[i] = self._evaluate_qa(encoded_qa_pairs[i])

        return [
            k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def _encode_qa(self, question: str, answer: str) -> torch.tensor:
        """
        Input: question and answer
        Return: a tensor of input ids
        """
        if type(answer) is list:
            for a in answer:
                if a["correct"]:
                    correct_answer = a["answer"]
        else:
            correct_answer = answer

        return self.qae_tokenizer(
            text=question,
            text_pair=correct_answer,
            padding="max_length",
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def _evaluate_qa(self, encoded_qa_pair: torch.tensor) -> float:
        """
        Input: encoded QA pair 
        Return: score."""
        output = self.qae_model(**encoded_qa_pair)
        return output[0][0][1]
