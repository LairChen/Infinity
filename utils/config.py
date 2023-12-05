from configparser import ConfigParser

cfg = ConfigParser()
cfg.read(filenames="conf/config.ini", encoding="utf-8")

appHost = cfg.get(section="app", option="app.host")
appPort = cfg.getint(section="app", option="app.port")

path_train_pretrain = cfg.get(section="path", option="path.train.pretrain")  # noqa
path_train_finetune = cfg.get(section="path", option="path.train.finetune")  # noqa
path_eval_finetune = cfg.get(section="path", option="path.eval.finetune")  # noqa

llm = {
    "model_max_length": cfg.getint(section="model", option="model.max.length"),
    "num_train_epochs": cfg.getint(section="model", option="num.train.epochs"),
    "per_device_train_batch_size": cfg.getint(section="model", option="per.device.train.batch.size")
}
