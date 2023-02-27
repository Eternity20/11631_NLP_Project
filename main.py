import sys
import argparse
import os
import logging
logging.basicConfig(level=logging.INFO)

def generate_questions(text, model_name):
	raise NotImplementedError # TODO implement based on the HW1 example

def get_answers(questions, model_name):
	raise NotImplementedError # TODO implement based on the HW1 example

if __name__ == '__main__':
	logger = logging.getLogger(__name__)
	parser = argparse.ArgumentParser()
	parser.add_argument('--qa_model', type=str, default='bert-base-uncased')
	parser.add_argument('--qg_model', type=str, default='bert-base-uncased')
	parser.add_argument('--data_path', type=str, default='data/wikipedia_text')
	parser.add_argument('--output_path', type=str, default='results/dummy.txt', help='path to dir where save results')
	args = parser.parse_args()

	# create directories if they don't exist
	os.makedirs(args.output_path, exist_ok=True)
	if not os.path.exists(args.data_path):
		logger.error("Data path does not exist. Exiting.")
		sys.exit(-1)
	if os.path.isdir(args.data_path):
		# TODO: add support for multiple files
		raise NotImplementedError
	else:
		with open(args.data_path, 'r') as f:
			text = f.readlines() # This loads text to memory
			generated_questions = generate_questions(text, args.qg_model)
			answers = get_answers(generated_questions, args.qa_model)
			with open(args.output_path, 'w') as f:
				for question, answer in zip(generated_questions, answers):
					f.write(f'{question} ###---### {answer}\n')
	sys.exit(0)
