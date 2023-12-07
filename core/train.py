from dataclasses import dataclass, field
from json import load
from typing import List, Dict, Union

from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, LongTensor
from torch.utils.data import Dataset
from transformers import TrainingArguments, HfArgumentParser, AutoModelForCausalLM, AutoTokenizer, Trainer, PreTrainedTokenizer


@dataclass
class ModelArguments(object):
    """模型参数"""
    model_name_or_path: str = field(default=None)


@dataclass
class DataArguments(object):
    """数据参数"""
    data_path: str = field(default=None)


@dataclass
class TrainArguments(TrainingArguments):
    """训练参数"""
    cache_dir: str = field(default=None)
    model_max_length: int = field(default=512)
    optim: str = field(default="adamw_torch")  # noqa
    use_lora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """有监督数据集"""

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, model_max_length: int,
                 user_tokens: List[int] = [195], assistant_tokens: List[int] = [196]):  # noqa
        super(SupervisedDataset, self).__init__()
        self.data = load(fp=open(file=data_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return self.preprocessing(data=self.data[idx])

    def preprocessing(self, data: Dict) -> Dict[str, Union[List[int], Tensor]]:  # noqa
        """数据预处理"""
        input_ids = []
        labels = []
        for message in data["conversations"]:
            from_, value = message["from"], message["value"]
            value_ids = self.tokenizer.encode(text=value)
            if from_ == "human":
                input_ids += self.user_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            elif from_ == "gpt":
                input_ids += self.assistant_tokens + value_ids
                labels += [self.ignore_index] + value_ids
            else:
                raise ValueError("error conversation structure in training data")
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        input_ids = LongTensor(input_ids)
        labels = LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def train() -> None:
    """微调主方法"""
    parser = HfArgumentParser(dataclass_types=(ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if training_args.use_lora:
        peft_config = LoraConfig(  # noqa
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model=model, peft_config=peft_config)
        model.print_trainable_parameters()
    dataset = SupervisedDataset(
        data_path=data_args.data_path, tokenizer=tokenizer, model_max_length=training_args.model_max_length
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()  # noqa
    trainer.save_model(output_dir=training_args.output_dir)
    return


if __name__ == "__main__":
    train()
