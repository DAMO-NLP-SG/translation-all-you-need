num_samples=3
results_folder=results

overwrite=0
overwrite_judge=0
do_inference=1
post_process=1
re_evaluate=0
script=run_shareGPT.py
tensor_parallel_size=1
# model=gpt-4.1-nano
model=Qwen/Qwen2-0.5B-Instruct
model_judge=gpt-4.1-nano
task_list=shareGPT

lang_list=all
prompt_type_list=all

# lang_list=zh
# prompt_type_list=direct

# for model_judge in "anthropic/claude-3.5-sonnet" "google/gemini-pro-1.5"; do
for prompt_type_list in direct google; do

gpu=3
CUDA_VISIBLE_DEVICES=$gpu python ./scripts/$script \
    --task_list $task_list \
    --model  $model \
    --model_judge $model_judge \
    --prompt_type_list $prompt_type_list \
    --lang_list $lang_list \
    --overwrite $overwrite \
    --overwrite_judge $overwrite_judge \
    --num_samples $num_samples \
    --results_folder $results_folder \
    --do_inference $do_inference \
    --post_process $post_process

done 
# done
