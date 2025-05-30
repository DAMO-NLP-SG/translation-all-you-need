from utils import *
from utils_langs import *
from utils_template import *
from util_mkqa import Rouge_Scorer
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mgsm', choices=['mgsm', 'xcopa', 'xnli', 'paws-x', 'xlsum', 'mkqa'], help='the name of the task')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', help='model name')
    parser.add_argument("--model_type", type=str, default="default", help="[default, openai, together, vllm]")
    parser.add_argument('--model_judge', type=str, default='gpt-3.5-turbo-1106', help='judge model name')
    parser.add_argument('--prompt_type', type=str, default='direct', choices=["direct","direct_native","native_cot","en_cot","google","nllb","xlt"], help='prompt type')
    parser.add_argument('--lang', type=str, default='zh', help='language code')
    parser.add_argument('-o','--overwrite', type=int, default=0, help='overwrite existing files')
    parser.add_argument('--num_samples', type=int, default='500', help='list of indices to run')
    parser.add_argument('--results_folder', type=str, default='results', help='list of indices to run')
    parser.add_argument('--verbose', type=int, default=0, help='debug mode')
    parser.add_argument('--do_inference', type=int, default=1, help='do inference')
    parser.add_argument('--post_process', type=int, default=1, help='post_process')
    parser.add_argument('--re_evaluate', type=int, default=0, help='post_process')
    parser.add_argument('--tensor_parallel_size', type=int, default=None, help='tensor parallel size')

    parser.add_argument('--task_list', type=str, default='all', help='list of tasks')
    parser.add_argument('--prompt_type_list', type=str, default='all', 
                        help="""list of prompt types, choices = ['direct',"direct_native","native_cot","xlt",'en_cot','google','nllb', "self_trans"] """)
    parser.add_argument('--lang_list', type=str, default='all', help='list of languages')
    args = parser.parse_args()
    return args

def print_info(args):
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


def get_agent(args):
    model_loaded, tokenizer = prepara_model(args.model)
    agent_base = Agent("baseline",'', model=args.model, temperature=0, max_tokens=1024,
                       model_loaded=model_loaded, tokenizer=tokenizer, model_type=args.model_type)
    return agent_base

def write_log(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message+'\n')

def inference(args):
    ds = get_data(args)
    # print(f'num_samples: {len(ds)}')
    prompt_for_ans = get_prompt_ans(args)
    output_folder = get_output_folder(args)
    log_file = f'{output_folder}/log.txt'
    # write the start time
    write_log(log_file, f'start time: {datetime.now()}')
    # write args information
    write_log(log_file, f'args: {args}')

    num_samples = min(args.num_samples, len(ds))
    list_idx = list(range(num_samples))
    agent_base = get_agent(args)

    for idx in list_idx:
        agent_base.reset()
        item = ds[idx]
        file_path = os.path.join(output_folder,f'{idx}.json')
        if os.path.exists(file_path) and not args.overwrite:
            print(f'file {file_path} exists, skipping...')
            continue

        # print(f'===============================')
        print(f'======={args.model}_{args.task}_{args.prompt_type}_{args.lang}_{idx}=======')
        prompt = gen_prompt(args, item)
        item['prompt'] = prompt

        res1 = agent_base.respond(prompt)
        if check_ans(args, res1):
            ans = clean_ans(args,res1)
        else:
            ans = agent_base.respond(prompt_for_ans)
            ans = clean_ans(args,ans)

        item['pred'] = ans
        item['check'] = evaluate_item(args, item)
        item['model'] = args.model
        item['message'] = agent_base.messages

        print(f'pred: {ans} {item["check"]}')

        with open(file_path, 'w') as f:
            json.dump(item, f, indent=2, ensure_ascii=False)

    write_log(log_file, f'end time: {datetime.now()}')

def prepare_llm(args):
    llm, sampling_params = prepare_vllm(args.model,tensor_parallel_size=args.tensor_parallel_size)
    if args.model == "TheBloke/Llama-2-70B-Chat-AWQ":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    return llm, sampling_params, tokenizer

def inference_vllm(args, llm, sampling_params, tokenizer):
    ds = get_data(args)
    prompt_for_ans = get_prompt_ans(args)
    output_folder = get_output_folder(args)
    num_samples = min(args.num_samples, len(ds))
    num_existing_json = len([f for f in os.listdir(output_folder) if f.endswith('.json')])
    if num_existing_json >= num_samples and not args.overwrite:
        print(f'folder {output_folder} exists, skipping...')
        return

    log_file = f'{output_folder}/log.txt'
    write_log(log_file, f'start time: {datetime.now()}')
    write_log(log_file, f'args: {args}')

    list_idx = list(range(num_samples))

    items = [ds[idx] for idx in list_idx]
    # round 1
    list_prompt = [gen_prompt(args, item) for item in items]
    list_message = [prompt_to_messages("user", prompt, []) for prompt in list_prompt]
    list_prompt = [messages_to_prompt(message, tokenizer) for message in list_message]
    responses = get_vllm_completion(llm, list_prompt, sampling_params)
    # round 2
    list_message = [prompt_to_messages("assistant", prompt, message) for prompt, message in zip(responses, list_message)]
    list_message = [prompt_to_messages("user", prompt_for_ans, message) for message in list_message]
    list_prompt = [messages_to_prompt(message, tokenizer) for message in list_message]
    responses = get_vllm_completion(llm, list_prompt, sampling_params)
    # final processing
    list_message = [prompt_to_messages("assistant", prompt, message) for prompt, message in zip(responses, list_message)]
    for i, item in tqdm(enumerate(items)):
        item['prompt'] = list_prompt[i]
        item['model'] = args.model
        item['message'] = list_message[i]
        # dump item to a json file
        file_path = os.path.join(output_folder,f'{i}.json')
        with open(file_path, 'w') as f:
            json.dump(item, f, indent=2, ensure_ascii=False)
    
    # write the end time
    write_log(log_file, f'end time: {datetime.now()}')

def post_process(args):
    output_folder = get_output_folder(args)
    num_files = len([f for f in os.listdir(output_folder) if (f.endswith('.json') and not f.startswith('d'))])
    if num_files == 0:
        print(f'folder {output_folder} is empty, skipping...')
        return
    num_samples = min(args.num_samples, num_files)
    list_idx = list(range(num_samples))
    list_Q = []
    list_A = []
    list_pred = []
    list_check = []
    for idx in tqdm(list_idx):
        file_path = os.path.join(output_folder,f'{idx}.json')
        with open(file_path, 'r') as f:
            item = json.load(f)
        if args.re_evaluate or 'pred' not in item:
            if "message" not in item:
                item['message'] = item['agent_base']
                del item['agent_base']
            response = item['message'][-1]['content']
            item['pred'] = clean_ans(args,response)
            item['check'] = evaluate_item(args, item)
            with open(file_path, 'w') as f:
                json.dump(item, f, indent=2, ensure_ascii=False)
        list_Q.append(item['prompt'])
        list_A.append(item['label'])
        list_pred.append(item['pred'])
        list_check.append(item['check'])

    df = pd.DataFrame({'Q':list_Q,'A':list_A,'pred':list_pred,'check':list_check})
    df.to_csv(f'{output_folder}/summary.csv',index=False)

    print('accuracy: ',np.mean(list_check))
    with open(f'{output_folder}/accuracy.txt','w') as f:
        f.write(str(np.mean(list_check)))

    with open(f'{args.results_folder}/accuracy_{args.task}.csv','a') as f:
        f.write(f'{args.model},{args.prompt_type},{args.lang},{num_samples},{np.mean(list_check)}\n')

def process_lang(args, llm, sampling_params, tokenizer):
    if args.lang == 'en': # and args.prompt_type in ['google','nllb','google_direct']:
        return

    if args.do_inference:
        # if "gpt-3.5" in args.model:
        if args.model_type != "vllm":
            inference(args)
        else:
            inference_vllm(args, llm, sampling_params, tokenizer)

    if args.post_process:
        post_process(args)

def main():
    args = get_args()
    if args.model_type == "default":
        if "gpt-3.5" in args.model or "gpt-4" in args.model:
            args.model_type = "openai"
        else:
            args.model_type = "vllm"

    args.tensor_parallel_size = torch.cuda.device_count() if args.tensor_parallel_size == 0 else args.tensor_parallel_size
    # llm, sampling_params, tokenizer = prepare_llm(args) if ("gpt-3.5" not in args.model and args.do_inference) else (None, None, None)
    llm, sampling_params, tokenizer = prepare_llm(args) if (args.model_type == "vllm" and args.do_inference) else (None, None, None)
    args.rouge_scorer = Rouge_Scorer(model_name_or_path = args.model, metrics=['rouge1', 'rougeL'])
    tasks = ['mgsm','xcopa','xnli','paws-x', 'mkqa', 'xlsum'] if args.task_list == "all" else args.task_list.split(',')
    prompt_types = ['direct',"direct_native","native_cot","xlt",'en_cot','google','nllb'] if args.prompt_type_list == "all" else args.prompt_type_list.split(',')
    print_info(args)

    for task in tasks:
        for prompt_type in prompt_types:
            args.task = task
            args.prompt_type = prompt_type
            langs = dic_list_langs[args.task] if args.lang_list == "all" else args.lang_list.split(',')
            for lang in langs:
                if args.task == 'mkqa' and lang == 'zh':
                    lang = 'zh_cn'
                args.lang = lang
                # assign translation pipeline
                if (task == 'mkqa' or task == 'xlsum') and prompt_type == 'nllb' and args.post_process:
                    args.translation_pipeline = prepare_pipeline_nllb(src_lang='en', tgt_lang=args.lang, max_length=400)
                print(f'================{args.model}_{args.task}_{args.prompt_type}_{args.lang}===============')
                process_lang(args, llm, sampling_params, tokenizer)

if __name__ == '__main__':
    main()
