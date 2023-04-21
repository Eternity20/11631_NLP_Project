import spacy

SPACY_MODEL = "en_core_web_md"
WH_FLAG = 'WH'
YN_FLAG = 'YN'

class QuestionDetector():
	def __init__(self, spacy_model=SPACY_MODEL):
		self.wh_lst = ['which', 'what', 'whose', 'who', 'whom', 'where', 'whither', 'whence', 'when', 'how', 'why', 'whether']
		self.nlp = spacy.load(spacy_model)

	def detect(self, question):
		doc = self.nlp(question)
		whflag = False
		questionflag = False
		for tok in doc:
			if doc.text[tok.sent.end_char - 1] == '?':
				questionflag = True
				if tok.lemma_.lower() in self.wh_lst:
					whflag = True
		if (questionflag and whflag):
			return WH_FLAG
		elif (questionflag):
			return YN_FLAG
		else:
			return WH_FLAG
