import pandas as pd
import os

dic_list_langs = {
    "mgsm":['en','de','ru','fr','zh','es','ja','sw','th','bn','te'],
    "xcopa":['zh', 'it', 'vi', 'tr', 'id', 'sw', 'th', 'et', 'ta', 'ht', 'qu'],
    "xnli":['en', 'de', 'ru', 'fr', 'zh', 'es', 'vi', 'tr', 'sw', 'ar', 'el', 'th', 'bg', 'hi', 'ur'],
    "paws-x":['en', 'de', 'fr', 'zh', 'es', 'ja', 'ko'],
    "xlsum":['en', 'fr', 'zh', 'es', 'vi', 'tr'],
    "mkqa": ['en', 'de', 'ru', 'fr', 'zh_cn', 'es', 'ja', 'vi', 'tr', 'th'],
    "shareGPT": ['zh', 'ko', 'es', 'fr', 'ja', 'id', 'vi', 'uk', 'ro', 'no']
}

def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def process_file(task):
    df = pd.read_csv(f'accuracy_{task}.csv',header=None)
    df.columns = ['model','prompt_type','lang','num','accuracy']
    # remove rows with num < 250
    if task == "shareGPT":
        df = df[df['num']>=50]
    else:
        df = df[df['num']>=250]
    # deduplicate
    df = df.drop_duplicates(subset=['model','prompt_type','lang','num'], keep='last')
    df = df.pivot_table(index=['num','model','prompt_type',],columns='lang',values='accuracy')
    df = df.reset_index()
    column_names = ['num','model','prompt_type',] + dic_list_langs[task]
    df = df.reindex(columns=column_names)

    # Assume 'column_name' is the column you want to sort and 'specific_order' is the list specifying the order
    df['prompt_type'] = pd.Categorical(df['prompt_type'], categories=["direct_native","direct","native_cot","en_cot","xlt","google","google_direct","nllb","self_trans"], ordered=True)
    # df['model'] = pd.Categorical(df['model'], categories=["gpt-3.5-turbo-1106","bigscience/bloomz-7b1","mistralai/Mistral-7B-Instruct-v0.2","meta-llama/Llama-2-13b-chat-hf","TheBloke/Llama-2-70B-Chat-AWQ","ybelkada/Mixtral-8x7B-Instruct-v0.1-AWQ"], ordered=True)
    # Then you can sort the DataFrame by 'column_name'
    df = df.sort_values(['num','model','prompt_type',])

    df = df.reset_index(drop=True)
    # save df to csv
    create_folder_if_not_exist(f'accuracy_processed')
    df.to_csv(f'accuracy_processed/accuracy_{task}_pivot.csv')

for task in ['mgsm', 'xcopa', 'xnli', 'paws-x', 'xlsum', 'mkqa', 'shareGPT']:
    # check whether file exists
    if os.path.isfile(f'accuracy_{task}.csv'):
        process_file(task)