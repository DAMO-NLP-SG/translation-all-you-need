from utils import *
from utils_langs import dic_list_langs
from utils_template import *
from util_mkqa import Rouge_Scorer
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mgsm', choices=['mgsm', 'xcopa', 'xnli', 'paws-x', 'xlsum', 'mkqa','shareGPT', 'shareGPT_filter'], help='the name of the task')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', help='model name')
    parser.add_argument('--model_judge', type=str, default='gpt-3.5-turbo-1106', help='judge model name')
    parser.add_argument('--prompt_type', type=str, default='direct', choices=["direct","en_cot","google","nllb"], help='prompt type')
    parser.add_argument('--lang', type=str, default='zh', help='language code')
    parser.add_argument('-o','--overwrite', type=int, default=0, help='overwrite existing files')
    parser.add_argument('-oj','--overwrite_judge', type=int, default=0, help='overwrite existing judge files')
    parser.add_argument('--num_samples', type=int, default='500', help='list of indices to run')
    parser.add_argument('--results_folder', type=str, default='results', help='list of indices to run')
    parser.add_argument('--verbose', type=int, default=0, help='debug mode')
    parser.add_argument('--do_inference', type=int, default=1, help='do inference')
    parser.add_argument('--post_process', type=int, default=1, help='post_process')
    parser.add_argument('--tensor_parallel_size', type=int, default=None, help='tensor parallel size')
    parser.add_argument('--task_list', type=str, default='all', help='list of tasks')
    parser.add_argument('--prompt_type_list', type=str, default='all', help='list of prompt types')
    parser.add_argument('--lang_list', type=str, default='all', help='list of languages')
    args = parser.parse_args()
    return args

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
    if args.model == "gpt-3.5-turbo-16k-0613":
        output_folder = f'{args.results_folder}/{args.task}/gpt-3.5-turbo-0613_{args.prompt_type}/{args.lang}'
    else:
        output_folder = f'{args.results_folder}/{args.task}/{args.model.split("/")[-1]}_{args.prompt_type}/{args.lang}'
    create_folder_if_not_exist(output_folder)
    return output_folder

template_judge = """[System] 
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, expected language and level of detail of the response. Begin your evaluation by providing a short explanation (up to 100 words). Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "Rating: <rating>", for example: "Rating: 5".

[Question]
{question}

[The Start of Assistant’s Answer]
{answer}
[The End of Assistant’s Answer]
"""

def clean_judge(judgement):
    pattern = 'Rating: (\d+)'
    pred = re.findall(pattern, judgement)
    if(len(pred) >= 1):
        pred = pred[-1]
    else: pred = -1
    return int(pred)

def gen_prompt_judge(question,answer):
    return template_judge.format(question=question, answer=answer)

def get_judge_folder(args):
    if args.model == "gpt-3.5-turbo-16k-0613":
        output_folder = f'{args.results_folder}/{args.task}/gpt-3.5-turbo-0613_{args.prompt_type}/{args.lang}'
    else:
        output_folder = f'{args.results_folder}/{args.task}_{args.model_judge.split("/")[-1]}/{args.model.split("/")[-1]}_{args.prompt_type}/{args.lang}'
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

def inference(args):
    ds = get_data(args)
    print(f'num_samples: {len(ds)}')
    output_folder = get_output_folder(args)
    log_file = f'{output_folder}/log.txt'
    write_log(log_file, f'start time: {datetime.now()}')
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

        print(prompt)
        print('-'*20)

        res1 = agent_base.respond(prompt)
        
        item['pred'] = clean_ans(args,res1)
        item['model'] = args.model
        item['agent_base'] = agent_base.messages

        print(f'pred: {res1}')

        with open(file_path, 'w') as f:
            json.dump(item, f, indent=2, ensure_ascii=False)

    # write the end time
    write_log(log_file, f'end time: {datetime.now()}')

def prepare_llm(args):
    llm, sampling_params = prepare_vllm(args.model,tensor_parallel_size=args.tensor_parallel_size)
    if args.model == "TheBloke/Llama-2-70B-Chat-AWQ":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    return llm, sampling_params, tokenizer

def inference_vllm(args, llm, sampling_params, tokenizer):
    print("preparing vllm inference....")
    ds = get_data(args)
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

    # prepare the data
    items = [ds[idx] for idx in list_idx]
    # round 1
    list_prompt = [gen_prompt(args, item) for item in items]
    list_message = [prompt_to_messages("user", prompt, []) for prompt in list_prompt]
    list_prompt = [messages_to_prompt(message, tokenizer) for message in list_message]
    responses = get_vllm_completion(llm, list_prompt, sampling_params)
    # round 2
    list_message = [prompt_to_messages("assistant", prompt, message) for prompt, message in zip(responses, list_message)]
    for i, item in tqdm(enumerate(items)):
        ans = clean_ans(args, responses[i])
        item['pred'] = ans
        # item['check'] = evaluate_item(args, item)
        item['prompt'] = list_prompt[i]
        item['model'] = args.model
        item['message'] = list_message[i]
        # dump item to a json file
        file_path = os.path.join(output_folder,f'{i}.json')
        if not os.path.exists(file_path) or args.overwrite:
            with open(file_path, 'w') as f:
                json.dump(item, f, indent=2, ensure_ascii=False)
    
    # write the end time
    write_log(log_file, f'end time: {datetime.now()}')

def post_process(args):
    output_folder = get_output_folder(args)
    judge_folder = get_judge_folder(args)
    num_files = len([f for f in os.listdir(output_folder) if f.endswith('.json')])
    num_samples = min(args.num_samples, num_files)
    list_idx = list(range(num_samples))
    list_Q = []
    list_pred = []
    list_check = []
    for idx in tqdm(list_idx):
        # agent_judge.reset()
        file_path = os.path.join(output_folder,f'{idx}.json')
        judge_path = os.path.join(judge_folder,f'{idx}.json')
        with open(file_path, 'r') as f:
            item = json.load(f)
        list_Q.append(item['prompt'])
        list_pred.append(item['pred'])
        if 'rating' not in item or args.overwrite_judge or not os.path.exists(judge_path):
            if args.prompt_type == 'google':
                prompt_judge = gen_prompt_judge(item['question_native'], item['pred'])
            else:
                prompt_judge = gen_prompt_judge(item['question'], item['pred'])
            message = prompt_to_messages("user", prompt_judge, [])
            res_judge = query_openrouter_model(message, args.model_judge,max_tokens=1024)
            item['model_judge'] = args.model_judge
            item['prompt_judge'] = prompt_judge
            item['judgement'] = res_judge
            item['rating'] = clean_judge(res_judge)
            # save the updated item
            with open(judge_path, 'w') as f:
                json.dump(item, f, indent=2, ensure_ascii=False)
        list_check.append(item['rating'])

    df = pd.DataFrame({'Q':list_Q,'pred':list_pred,'rating':list_check})
    df.to_csv(f'{judge_folder}/summary.csv',index=False)
    # check number of ratings that is equal to -1
    num_invalid = len([x for x in list_check if x == -1])
    # write the number of invalid ratings to log file
    log_file = f'{judge_folder}/log.txt'
    write_log(log_file, f'num_invalid: {num_invalid}')
    log_file2 = f'{args.results_folder}/log.txt'
    write_log(log_file2, f'output_folder: {judge_folder}, num_invalid: {num_invalid}')
    # update list_check
    list_check = [x for x in list_check if x != -1]

    print('rating: ',np.mean(list_check))
    with open(f'{judge_folder}/accuracy.txt','w') as f:
        f.write(str(np.mean(list_check)))

    with open(f'{args.results_folder}/accuracy_{args.task}_{args.model_judge.split("/")[-1]}.csv','a') as f:
        f.write(f'{args.model},{args.model_judge},{args.prompt_type},{args.lang},{num_samples},{np.mean(list_check)}\n')

def process_lang(args, llm, sampling_params, tokenizer):
    if (args.task == 'mkqa' or args.task == 'xlsum') and args.prompt_type == 'nllb':
        try:
            args.translation_pipeline = prepare_pipeline_nllb(src_lang='en', tgt_lang=args.lang, max_length=400) 
        except:
            return

    if args.lang == 'en' and args.prompt_type in ['google','nllb']:
        return
    if args.do_inference:
        if "gpt-" in args.model:
            inference(args)
        else:
            print("inference vllm....")
            inference_vllm(args, llm, sampling_params, tokenizer)
    if args.post_process:
        post_process(args)

def main():
    args = get_args()

    args.tensor_parallel_size = torch.cuda.device_count() if args.tensor_parallel_size == 0 else args.tensor_parallel_size
    llm, sampling_params, tokenizer = prepare_llm(args) if ("gpt-" not in args.model and args.do_inference) else (None, None, None)
    args.rouge_scorer = Rouge_Scorer(model_name_or_path = args.model, metrics=['rouge1', 'rougeL'])
    tasks = ['shareGPT'] if args.task_list == "all" else args.task_list.split(',')
    prompt_types = ['direct','google'] if args.prompt_type_list == "all" else args.prompt_type_list.split(',')
    print_info(args)

    for task in tasks:
        for prompt_type in prompt_types:
            args.task = task
            args.prompt_type = prompt_type
            langs = dic_list_langs[args.task] if args.lang_list == "all" else args.lang_list.split(',')
            for lang in langs:
                args.lang = lang
                print(f'================{args.model}_{args.task}_{args.prompt_type}_{args.lang}===============')
                process_lang(args, llm, sampling_params, tokenizer)

if __name__ == '__main__':
    main()
