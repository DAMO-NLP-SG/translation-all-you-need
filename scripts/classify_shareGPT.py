# v2: add support for mkqa and xlsum
from utils import *
from utils_langs import *
from utils_template import *
from util_mkqa import Rouge_Scorer
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm

def print_info(args):
    # print(f'task: {args.task}')
    # print(f'lang: {args.lang}')
    # print(f'model: {args.model}')
    print(f'prompt_type: {args.prompt_type}')
    print(f'num_samples: {args.num_samples}')
    print(f'overwrite: {args.overwrite}')
    print(f'results_folder: {args.results_folder}')
    print(f'verbose: {args.verbose}')
    print(f'do_inference: {args.do_inference}')
    print(f'tensor_parallel_size: {args.tensor_parallel_size}')
    print(f'task_list: {args.task_list}')
    print(f'prompt_type_list: {args.prompt_type_list}')
    print(f'lang_list: {args.lang_list}')

def get_output_folder(args):
    output_folder = f'{args.results_folder}/{args.task}/category/{args.lang}'
    create_folder_if_not_exist(output_folder)
    return output_folder

def get_agent(args):
    model_loaded, tokenizer = prepara_model(args.model)
    agent_base = Agent("baseline",'', model=args.model, temperature=0, max_tokens=1024,
                       model_loaded=model_loaded, tokenizer=tokenizer) 
    return agent_base

def write_log(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message+'\n')

template_classification = """Given the following user request, evaluate whether answering this request necessitates local cultural knowledge in that language. Consider factors such as cultural practices, beliefs, historical context, language nuances, and societal norms. Begin your evaluation by providing a short explanation (up to 50 words). After providing your explanation, provide a final assessment of whether cultural knowledge is essential for an accurate and comprehensive answer. Give you final answer in the format "Answer: [Yes/No]".

User Request: 
{question}
"""

def extract_ans(res):
    # result = res.split('Answer: ')[1].strip()
    list_result = res.split('Answer: ')
    if len(list_result) > 1:
        result = list_result[1].strip()
    else:
        result = list_result[0].strip()
    if 'yes' in result.lower():
        return 1
    else:
        return 0

def classify_questions(args):
    ds = get_data(args)
    output_folder = get_output_folder(args)
    num_samples = min(args.num_samples, len(ds))

    log_file = f'{output_folder}/log.txt'
    write_log(log_file, f'start time: {datetime.now()}')
    write_log(log_file, f'args: {args}')

    list_idx = list(range(num_samples))
    agent_base = get_agent(args)

    list_items = []
    num_preds_yes = 0
    for idx in tqdm(list_idx):
        agent_base.reset()
        
        file_path = os.path.join(output_folder,f'{idx}.json')
        if os.path.exists(file_path) and not args.overwrite:
            print(f'file {file_path} exists, skipping...')
            # read the item 
            with open(file_path, 'r') as f:
                item = json.load(f)
                list_items.append(item)
                num_preds_yes += item['pred']
            continue
            
        item = ds[idx]
        # prompt = gen_prompt(args, item)
        question = item['question']
        prompt = template_classification.format(question=question)

        item['prompt'] = prompt

        res1 = agent_base.respond(prompt)
        # item['pred'] = clean_ans(args,res1)
        item['pred'] = extract_ans(res1)
        num_preds_yes += item['pred']
        item['model'] = args.model
        item['agent_base'] = agent_base.messages
        list_items.append(item)

        # print(f'pred: {res1}')

        with open(file_path, 'w') as f:
            json.dump(item, f, indent=2, ensure_ascii=False)

    write_log(log_file, f'end time: {datetime.now()}')

    with open(f'{output_folder}/results.json', 'w') as f:
        json.dump(list_items, f, indent=2, ensure_ascii=False)

    print(f'results saved to {output_folder}/results.jsonl')
    print(f'num_preds_yes: {num_preds_yes}/{num_samples}')
    # save num_preds_yes to a file
    with open(f'{output_folder}/num_preds_yes.txt', 'w') as f:
        f.write(f'{num_preds_yes}/{num_samples}')

    # save num_preds_yes to a csv file
    with open(f'results/shareGPT/category/num_preds_yes.csv', 'a') as f:
        f.write(f'{args.lang},{num_preds_yes},{num_samples},{num_preds_yes/num_samples}\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='shareGPT', choices=['mgsm', 'xcopa', 'xnli', 'paws-x', 'xlsum', 'mkqa','shareGPT'], help='the name of the task')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', help='model name')
    parser.add_argument('--model_judge', type=str, default='gpt-4-0125-preview', help='judge model name')
    parser.add_argument('--prompt_type', type=str, default='direct', choices=["direct","en_cot","google","nllb"], help='prompt type')
    parser.add_argument('--lang', type=str, default='zh', help='language code')
    parser.add_argument('-o','--overwrite', type=int, default=0, help='overwrite existing files')
    parser.add_argument('-oj','--overwrite_judge', type=int, default=0, help='overwrite existing judge files')
    parser.add_argument('--num_samples', type=int, default='500', help='list of indices to run')
    parser.add_argument('--results_folder', type=str, default='results', help='list of indices to run')
    parser.add_argument('--verbose', type=int, default=0, help='debug mode')
    parser.add_argument('--do_inference', type=int, default=1, help='do inference')
    parser.add_argument('--post_process', type=int, default=1, help='post_process')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='tensor parallel size')

    parser.add_argument('--task_list', type=str, default='shareGPT', help='list of tasks')
    parser.add_argument('--prompt_type_list', type=str, default='all', help='list of prompt types')
    parser.add_argument('--lang_list', type=str, default='all', help='list of languages')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    tasks = ['shareGPT'] if args.task_list == "all" else args.task_list.split(',')
    prompt_types = ['direct','google'] if args.prompt_type_list == "all" else args.prompt_type_list.split(',')
    print_info(args)

    langs = dic_list_langs[args.task] if args.lang_list == "all" else args.lang_list.split(',')
    for lang in langs:
        args.lang = lang
        print(f'================{args.lang}===============')
        classify_questions(args)

if __name__ == '__main__':
    main()
