import json
import sys
import argparse
import os
import logging
import torch
logging.basicConfig(level=logging.INFO)
from pathlib import Path
from qa import QAModel

QA_MODEL = 'deepset/tinyroberta-squad2'
QG_MODEL = 'allenai/t5-small-squad2-question-generation'

# This fn validates the args, and resturns and experiment_id
def run_arg_sanity_checks(_args) -> str:
	valid = True
	experiment_id = ''
	if _args.do_qa and _args.do_qg:
		valid = False
		logger.error('Cannot run QA and QG models at the same time.'
		             'Please specify *only one* of the flags qa_model, qg_model flags')
	if not _args.do_qa and not _args.do_qg:
		valid = False
		logger.error('Please specify *only one* of the flags qa_model, qg_model flags')

	if not os.path.exists(_args.data_dir_path) and not os.path.exists(_args.wiki_file_path):
		valid = False
		logger.error("Data path does not exist. Check the data_dir_path or wiki_file_path Exiting.")

	if not valid:
		sys.exit(-1)
	else:
		if len(_args.experiment_id) > 0:
			experiment_id = _args.experiment_id
		else:
			nr_results = sum(1 for element in Path(args.output_dir_path).iterdir()) #if element.is_file())
			experiment_id += f'QA_{nr_results}' if args.do_qa else f'QG_{nr_results}'
		return experiment_id

def generate_questions(text, model_name):
	raise NotImplementedError # TODO implement based on the HW1 example

def get_answers(qa_model_path, device, config_dict_path, wiki_file_path=None, wiki_dir_path=None, paragraph=True):
	if config_dict_path:
		with open(config_dict_path, 'r') as conf_f:
			config = json.load(conf_f)
			model = QAModel.from_config_dict(qa_model_path, device, config)
	else:
		model = QAModel(qa_model_path, device)
	# process our dataset (wikipedia files)
	eval_dataset, eval_dataloader = model.load_dataset_and_dataloaders(wiki_file_path, wiki_dir_path, paragraph=paragraph)
	# Get answers using pretrained model
	start_logits, end_logits = model.qa_inference(eval_dataloader)
	pred_answers = model.post_processing(raw_dataset=eval_dataset.data,
                                         tokenized_dataset=eval_dataset.tokenized_data,
                                         start_logits=start_logits, end_logits=end_logits)
	return pred_answers

if __name__ == '__main__':
	logger = logging.getLogger(__name__)
	parser = argparse.ArgumentParser()
	parser.add_argument('--do_qa', action='store_true')
	parser.add_argument('--do_qg', action='store_true')
	parser.add_argument('--do_dir', action='store_true')
	parser.add_argument('--cpu', action='store_true')
	parser.add_argument('--document', action='store_true')
	parser.add_argument('--qa_model', type=str, default=QA_MODEL)
	parser.add_argument('--qg_model', type=str, default=QG_MODEL)
	parser.add_argument('--data_dir_path', type=str, default='data/wikipedia_text')
	#parser.add_argument('--wiki_file_path', type=str, default='data/wikipedia_text/chinese_dynasties/Chen_dynasty.txt')
	parser.add_argument('--wiki_file_path', type=str, default=None)
	parser.add_argument('--experiment_id', type=str, default='dummy.txt')
	parser.add_argument('--output_dir_path', type=str, default='results/', help='path to dir where save results')
	parser.add_argument('--output_file_path', type=str, default='results/{}', help='path to file to save results')
	parser.add_argument('--config_dict_path', type=str, default=None)
	args = parser.parse_args()

	# create directories if they don't exist
	os.makedirs(args.output_dir_path, exist_ok=True)
	# Do sanity check on arguments
	experiment_id = run_arg_sanity_checks(args)

	cpu = args.cpu
	if torch.cuda.is_available() and not cpu:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	logger.info(f'Using Device: {device}')

	# Give preference to running single file experiments than for processing the whole dir
	paragraph = not args.document
	if args.do_dir:
		# TODO: add support for multiple files
		raise NotImplementedError
	else:
		with open(args.wiki_file_path, 'r') as f:
			text = f.readlines() # This loads text to memory
			if args.do_qa:
				predicted_answers = get_answers(args.qa_model_path, device, args.config_dict_path,
				                                wiki_file_path=args.wiki_file_path, wiki_dir_path=args.wiki_dir_path,
				                                paragraph=paragraph)
				outputs = predicted_answers
			if args.do_qg:
				generated_questions = generate_questions(text, args.qg_model)

			# write results to disk (file)
			out_file_path = args.output_file_path.format(experiment_id)
			with open(out_file_path, 'w') as f:
				json.dump(outputs, f)
	sys.exit(0)
