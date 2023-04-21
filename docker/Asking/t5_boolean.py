import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
from transformers import T5ForConditionalGeneration,T5Tokenizer,T5TokenizerFast
import torch
import sys
class YesNoQuestionGenerator:
  def __init__(self,passage, nquestion):
    self.truefalse = "yes"
    self.text ="truefalse: %s passage: %s </s>" % (passage, self.truefalse)
    #self.model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
    #self.tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    self.model = T5ForConditionalGeneration.from_pretrained("pretrained/qg_model_boolean",local_files_only = True)
    self.tokenizer = T5Tokenizer.from_pretrained('pretrained/qg_tok_boolean',local_files_only = True)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.encoding = self.tokenizer.encode_plus(self.text, max_length=512, return_tensors="pt", return_overflowing_tokens=True, return_attention_mask=True, pad_to_max_length=True, truncation=True)
    self.input_ids, self.attention_masks = self.encoding["input_ids"].to(self.device), self.encoding["attention_mask"].to(self.device)
    self.nquestion = nquestion

  def generate(self):
      topkp_output = self.model.generate(input_ids=self.input_ids,
                                         attention_mask=self.attention_masks,
                                         max_length=512,
                                         do_sample=True,
                                         top_k=40,
                                         top_p=0.80,
                                         num_return_sequences=1,
                                         no_repeat_ngram_size=2,
                                         early_stopping=True
                                        )
      #Questions = [self.tokenizer.decode(out, skip_special_tokens=True,clean_up_tokenization_spaces=True) for out in topkp_output]
      topk_decoded_questions = self.tokenizer.batch_decode(topkp_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
      return [question.strip().capitalize() for nq, question in enumerate(topk_decoded_questions) if nq < self.nquestion]

if __name__ == '__main__':
     input_file = sys.argv[1]
     N = int(sys.argv[2])

     with open(input_file, encoding="UTF-8") as a:
         article = a.read()

     qg = YesNoQuestionGenerator(article, N)
     output = qg.generate()
     for out in output:
         print(out)


