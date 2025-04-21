import json
import os, torch
import jsonlines
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
) 
from transformers import AutoTokenizer #, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
import requests
from openai import OpenAI

client = OpenAI()

# These are the basic functions to call ChatGPT
def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")

@retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(6), before=before_retry_fn)
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)
    
def get_completion_messages(messages, model="gpt-3.5-turbo-1106",temperature=0,max_tokens=1024, 
                            model_loaded=None, tokenizer=None,
                            model_type = "openai"):

    if model_type == "openai":
        response = completion_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature, 
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    elif model_type == "together":
        response = query_together_model(messages, model=model, max_tokens=max_tokens, temperature=temperature)
        return response
    elif "mis" in model and model_loaded != None and tokenizer != None:
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to('cuda')
        generated_ids = model_loaded.generate(model_inputs, max_new_tokens=max_tokens, \
                                              pad_token_id=tokenizer.pad_token_id, \
                                              do_sample=True if temperature > 0 else False, temperature=temperature)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0].split('[/INST] ')[-1]
    else:
        raise NotImplementedError
    
def get_completion(prompt, model="gpt-3.5-turbo",temperature=0,max_tokens=1024, model_loaded=None, tokenizer=None):
    messages = [{"role": "user", "content": prompt}]
    response = get_completion_messages(messages, model=model,temperature=temperature,max_tokens=max_tokens, 
                                       model_loaded=model_loaded, tokenizer=tokenizer)
    return response

@retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(5), before=before_retry_fn)
def query_together_model(messages, model: str = "Qwen/Qwen1.5-72B-Chat", max_tokens: int = 128, temperature: float = 0):
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    api_key = os.getenv("TOGETHER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}",}
    try: 
        res = requests.post(endpoint, json={
            "model": model,
            "max_tokens": max_tokens,
            "request_type": "language-model-inference",
            "temperature": temperature,
            "messages": messages#[{"role": "user", "content": prompt}]
        }, headers=headers)
        output = res.json()['choices'][0]['message']['content']
    except Exception as e:
        res_json = res.json()
        if 'error' in res_json and res_json['error']['message'].startswith('Input validation error: `inputs`'):
            output = "the question is too long"
        else:
            raise e
    return output

@retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(3), before=before_retry_fn)
def query_openrouter_model(messages, model="openai/gpt-4o-mini", max_tokens=64, temperature=0):

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),)
    try:
        completions = client.chat.completions.create(
            extra_headers={},
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        output = completions.choices[0].message.content.strip()

    except Exception as e:
        print("[ERROR]", e)
        output = 'CAUTION: Problematic output!'

    return output

def parallel_query_chatgpt_model(args):
    return get_completion(*args)

class Agent: 
    def __init__(self, name, system_prompt = '', model="gpt-3.5-turbo-0125", temperature=0, max_tokens=1024, 
                 model_loaded=None, tokenizer=None, model_type="openai"):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.model_type = model_type
        self.model_loaded = model_loaded
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = [] if system_prompt == '' else [{"role": "system", "content": system_prompt}]
        
    def respond(self, prompt, model = None):
        self.messages.append({"role": "user", "content": prompt})
        response = get_completion_messages(self.messages, model=self.model if model == None else model, temperature=self.temperature, max_tokens=self.max_tokens,
                                           model_loaded=self.model_loaded, tokenizer=self.tokenizer, model_type=self.model_type)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def respond_messages(self, messages, model = None):
        self.messages = messages
        response = get_completion_messages(self.messages, model=self.model if model == None else model, temperature=self.temperature, max_tokens=self.max_tokens,
                                           model_loaded=self.model_loaded, tokenizer=self.tokenizer, model_type=self.model_type)
        self.messages.append({"role": "assistant", "content": response})
        return response
    
    def reset(self):
        self.messages = [] if self.system_prompt == '' else [{"role": "system", "content": self.system_prompt}]

# Helper functions
def read_line_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [l.strip() for l in lines]

# write a list of strings to a txt file
def write_list_to_txt(list_str, file_path):
    with open(file_path, 'w') as f:
        for s in list_str:
            f.write(s+'\n')

# create a folder if it doesn't exist
def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def jsonl_to_list(path):
    with jsonlines.open(path) as reader:
        dataset = [obj for obj in reader]
    return dataset

def list_to_jsonl(dataset,path):
    with jsonlines.open(path, mode='w') as writer:
        writer.write_all(dataset)

def print_json(ex):
    print(json.dumps(ex, indent=2, ensure_ascii=False))

def make_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_mixtral_model(model_name, precision='fp32'):
    if precision == 'fp32':
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def prepara_model(model, precision='fp32'):
    if 'mistral' in model:
        model_loaded, tokenizer = load_mixtral_model(model, precision=precision)
        tokenizer.pad_token = tokenizer.eos_token
    else: 
        model_loaded = None
        tokenizer = None
    return model_loaded, tokenizer

# functions for vllm
def prepare_vllm(model, temperature=0, max_tokens=1024, tensor_parallel_size=None):
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    tensor_parallel_size = torch.cuda.device_count() if tensor_parallel_size == None else tensor_parallel_size
    if "AWQ" in model:
        llm = LLM(model=model,quantization="AWQ", dtype="auto",tensor_parallel_size=tensor_parallel_size)
    elif "bloom" in model: 
        llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    else:
        try:
            llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size,load_format="safetensors", gpu_memory_utilization=0.6)
        except:
            llm = LLM(model=model)
    return llm, sampling_params

def get_vllm_completion(llm,prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def prompt_to_messages(role, prompt, messages=[]):
    # role: user, assistant, system
    message = {"role": role, "content": prompt}
    messages.append(message)
    return messages

def messages_to_prompt(messages, tokenizer):
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True)
    else:
        prompt = ' '.join([f"{m['content']}" for m in messages])
    return prompt

def prompt_to_chatprompt(prompt, tokenizer):
    messages = prompt_to_messages('user', prompt, messages=[])
    prompt = messages_to_prompt(messages, tokenizer)
    return prompt

# testing functions
def _test_completion():
    model = "gpt-3.5-turbo"

    if 'mistral' in model:
        model_loaded, tokenizer = load_mixtral_model(model)
    else: 
        model_loaded = None
        tokenizer = None

    res = get_completion("Qustion: what is 10 * 5 -4?\n Answer: Let's think step by step.", model=model,temperature=0,max_tokens=1024, model_loaded=model_loaded, tokenizer=tokenizer)
    print(res)

def _test_agent():
    model = "gpt-3.5-turbo-1106"
    model = "zero-one-ai/Yi-34B-Chat"
    model_type = "together"

    if 'mistral' in model:
        model_loaded, tokenizer = load_mixtral_model(model)
    else: 
        model_loaded = None
        tokenizer = None

    agent = Agent(name='test', model=model, model_loaded=model_loaded, tokenizer=tokenizer,model_type=model_type)
    res = agent.respond("Qustion: what is the captital of China?\n Answer:")
    res2 = agent.respond("Answer in Chinese.")
    # print(res)
    print_json(agent.messages)

def _test_vllm():
    model = "mistralai/Mistral-7B-Instruct-v0.2"
    # model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # model = "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ"
    # model = "TheBloke/Yi-34B-Chat-AWQ"
    # model = "01-ai/Yi-34B-Chat-4bits"
    # model = "bigscience/bloomz-7b1"
    # model = "meta-llama/Llama-2-13b-chat-hf"
    # model= "TheBloke/Llama-2-70B-Chat-AWQ"
    prompts = [
        "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "What is the capital city of France?",
        "How many sides does a square have?",
        "怎么专注于工作？",
    ]
    # prompt_template='''[INST] {prompt} [/INST]'''
    # prompts = [prompt_template.format(prompt=prompt) for prompt in prompts]
    tokenizer = AutoTokenizer.from_pretrained(model)

    list_message = [prompt_to_messages("user", prompt, []) for prompt in prompts]
    list_prompt = [messages_to_prompt(message, tokenizer) for message in list_message]

    llm, sampling_params = prepare_vllm(model)
    responses = get_vllm_completion(llm, list_prompt, sampling_params)
    for prompt, response in zip(list_prompt, responses):
        print("prompt: ", prompt)
        print("response: ", response)

def _test_together():
    prompt = "Qustion: what is a good time to go to sleed and get up?\n Answer:"
    model = "Qwen/Qwen1.5-72B-Chat"
    # model = "zero-one-ai/Yi-34B-Chat"
    # res = query_together_model(os.getenv("TOGETHER_API_KEY"), prompt, model=model, max_tokens=1024, temperature=0)
    res = query_openrouter_model(prompt, model="openai/gpt-4o-mini", max_tokens=64, temperature=0)
    print(res)

def test_openrouter():
    api_key = os.environ["OPENROUTER_API_KEY"]
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France and Germany?"
        }
    ]
    list_models = ['anthropic/claude-3.5-sonnet', 'openai/gpt-4o-2024-08-06', 'google/gemini-pro-1.5',
               'google/gemini-flash-1.5', 'openai/gpt-4o-mini', 'anthropic/claude-3-haiku', 
               'qwen/qwen-2.5-72b-instruct', 'meta-llama/llama-3.1-70b-instruct', 'meta-llama/llama-3.1-405b-instruct'
               ]
    
    list_models = ['openai/gpt-4o-mini']
    for model in list_models:
        output = query_openrouter_model(messages=messages, model=model)
        print(f"model: {output}")

if __name__ == "__main__":
    # _test()
    _test_completion()
    # _test_agent()
    # _test_translator()
    # _test_vllm()
    # _test_together()
    # test_openrouter()
