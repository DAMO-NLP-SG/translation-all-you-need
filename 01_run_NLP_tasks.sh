num_samples=3
results_folder=results
overwrite=1
do_inference=1
post_process=1
re_evaluate=0
script=run_nlp_tasks.py

# model=meta-llama/Llama-2-7b-chat-hf
model=Qwen/Qwen2-1.5B-Instruct
# model=gpt-3.5-turbo
# model=gpt-4.1-nano

lang_list=zh
prompt_type_list=direct
task_list=all

gpu=1,2

CUDA_VISIBLE_DEVICES=$gpu python ./scripts/$script \
    --task_list $task_list \
    --model  $model \
    --prompt_type_list $prompt_type_list \
    --lang_list $lang_list \
    --overwrite $overwrite \
    --num_samples $num_samples \
    --results_folder $results_folder \
    --do_inference $do_inference \
    --post_process $post_process \
    --re_evaluate $re_evaluate
   



