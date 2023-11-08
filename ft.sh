#! /bin/sh

NUM_EPOCH=30
NUM_DATA=250

cd /tmp/dataset
unzip Baichuan2-7B-Chat.zip

cd /tmp/code
deepspeed --hostfile="" train.py \
  --report_to "none" \
  --model_name_or_path "/tmp/dataset/Baichuan2-7B-Chat" \
  --data_path "/tmp/code/train.json" \
  --output_dir "/tmp/output/model" \
  --model_max_length 1024 \
  --num_train_epochs $NUM_EPOCH \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy epoch \
  --learning_rate 2e-5 \
  --lr_scheduler_type constant \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --weight_decay 1e-4 \
  --warmup_ratio 0.0 \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --deepspeed ds.json \
  --bf16 True \
  --tf32 False \
  --use_lora True

for ((i = 1; i < $NUM_EPOCH; i++)); do
  num=$(expr $i \* $NUM_DATA)
  dir=$(printf /tmp/output/model/checkpoint-%d $num)
  rm -rf $dir
done
