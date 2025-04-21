from utils import *
from utils_langs import *
from utils_template_v4 import *
from util_mkqa import Rouge_Scorer
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm

template = """Translate the following question from {lang} to English: 
{text}
Don't answer the question, just translate it!"""

def remove_quotas(text):
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1].strip()
    else:
        return text.strip()
    
def get_output_folder(args):
    output_folder = f'{args.results_folder}/{args.task}/{args.model.split("/")[-1]}_{args.prompt_type}/{args.lang}'
    create_folder_if_not_exist(output_folder)
    return output_folder
    
def prepare_llm(args):
    llm, sampling_params = prepare_vllm(args.model,tensor_parallel_size=args.tensor_parallel_size)
    if args.model == "TheBloke/Llama-2-70B-Chat-AWQ":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    return llm, sampling_params, tokenizer
    
def self_translate(data, llm, sampling_params, tokenizer, args):
    if args.task == 'mgsm':
        questions = [d['Q'] for d in data]
        prompts = [template.format(lang=lang_codes[args.lang], text=q) for q in questions]
        print(prompts[0])
        list_message = [prompt_to_messages("user", prompt, []) for prompt in prompts]
        list_prompt = [messages_to_prompt(message, tokenizer) for message in list_message]
        responses = get_vllm_completion(llm, list_prompt, sampling_params)
        translations = [remove_quotas(translation) for translation in responses]
        for i, d in enumerate(data):
            d['Q'] = translations[i]
        return data
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mgsm', choices=['mgsm', 'xcopa', 'xnli', 'paws-x', 'xlsum', 'mkqa'], help='the name of the task')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', help='model name')
    parser.add_argument("--model_type", type=str, default="default", help="[default, openai, together, vllm]")
    parser.add_argument('--model_judge', type=str, default='gpt-3.5-turbo-1106', help='judge model name')
    parser.add_argument('--prompt_type', type=str, default='direct', choices=["direct","direct_native","native_cot","en_cot","google","nllb","xlt",'self_trans'], help='prompt type')
    parser.add_argument('--lang', type=str, default='zh', help='language code')
    parser.add_argument('-o','--overwrite', type=int, default=0, help='overwrite existing files')
    parser.add_argument('-oj','--overwrite_judge', type=int, default=0, help='overwrite existing judge files')
    parser.add_argument('--num_samples', type=int, default='500', help='list of indices to run')
    parser.add_argument('--results_folder', type=str, default='results', help='list of indices to run')
    parser.add_argument('--verbose', type=int, default=0, help='debug mode')
    parser.add_argument('--do_inference', type=int, default=1, help='do inference')
    parser.add_argument('--post_process', type=int, default=1, help='post_process')
    parser.add_argument('--re_evaluate', type=int, default=0, help='post_process')
    # parser.add_argument('--translate_back', type=int, default=0, help='translate back for xlsum and mkqa datasets')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='tensor parallel size')
    parser.add_argument('--task_list', type=str, default='all', help='list of tasks')
    parser.add_argument('--prompt_type_list', type=str, default='all', 
                        help="""list of prompt types, choices = ['direct',"direct_native","native_cot","xlt",'en_cot','google','nllb', "self_trans"] """)
    parser.add_argument('--lang_list', type=str, default='all', help='list of languages')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    llm, sampling_params, tokenizer = prepare_llm(args)

    langs = dic_list_langs[args.task] if args.lang_list == "all" else args.lang_list.split(',')
    for args.lang in langs:
        if args.lang == 'en':
            continue
        output_folder = get_output_folder(args)
        ds = get_data(args)
        num_samples = min(args.num_samples, len(ds))
        ds = ds[:num_samples]
        ds_translated = self_translate(ds, llm, sampling_params, tokenizer, args)
        # save to json file
        file_path = os.path.join(output_folder,f'data_translated.json')
        with open(file_path, 'w') as f:
            json.dump(ds_translated, f, indent=2, ensure_ascii=False)

def clean_translation():
    args = get_args()
    langs = dic_list_langs[args.task] if args.lang_list == "all" else args.lang_list.split(',')
    for args.lang in langs:
        if args.lang == 'en':
            continue
        output_folder = get_output_folder(args)
        # save to json file
        file_path = os.path.join(output_folder,f'data_translated.json')
        file_path_cleaned = os.path.join(output_folder,f'data_translated_cleaned.json')

        with open(file_path, 'r') as f:
            ds = json.load(f)

        for d in ds:
            d['Q'] = d['Q'].split("English:\n\n")[-1]
            # remove '"' at the beginning and end of the string
            d['Q'] = remove_quotas(d['Q'])

        with open(file_path_cleaned, 'w') as f:
            json.dump(ds, f, indent=2, ensure_ascii=False)

        print(f"Saved to {file_path_cleaned}")

if __name__ == "__main__":
    main()
    clean_translation()
