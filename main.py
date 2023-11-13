from os import system

data_path = "/tmp/dataset"
tune_path = "/tmp/output"

cmd_unzip = f"unzip {data_path}/Baichuan2-7B-Chat.zip -d {data_path}"  # noqa
cmd_train = "deepspeed --hostfile='' train.py " \
            "--report_to none " \
            f"--model_name_or_path {data_path}/Baichuan2-7B-Chat " \
            "--data_path data/train.json " \
            f"--output_dir {tune_path} " \
            "--model_max_length 1024 " \
            "--num_train_epochs 30 " \
            "--per_device_train_batch_size 1 " \
            "--gradient_accumulation_steps 1 " \
            "--save_strategy epoch " \
            "--learning_rate 2e-5 " \
            "--lr_scheduler_type constant " \
            "--adam_beta1 0.9 " \
            "--adam_beta2 0.98 " \
            "--adam_epsilon 1e-8 " \
            "--max_grad_norm 1.0 " \
            "--weight_decay 1e-4 " \
            "--warmup_ratio 0.0 " \
            "--logging_steps 1 " \
            "--gradient_checkpointing True " \
            "--deepspeed data/deepspeed.json " \
            "--bf16 True " \
            "--tf32 False " \
            "--use_lora True"
cmd_predict = "python predict.py " \
              f"--base {data_path}/Baichuan2-7B-Chat " \
              f"--output {tune_path}"

if __name__ == "__main__":
    system(cmd_unzip)  # 预训练模型解压缩
    system(cmd_train)  # 模型微调
    system(cmd_predict)  # 模型预测
