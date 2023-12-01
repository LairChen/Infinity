from configparser import ConfigParser

cfg = ConfigParser()
cfg.read(filenames="conf/config.ini", encoding="utf-8")

appHost = cfg.get(section="app", option="app.host")
appPort = cfg.getint(section="app", option="app.port")
appAddr = cfg.get(section="app", option="app.addr")

path_train_pretrain = cfg.get(section="path", option="path.train.pretrain")  # noqa
path_train_finetune = cfg.get(section="path", option="path.train.finetune")  # noqa
path_eval_finetune = cfg.get(section="path", option="path.eval.finetune")  # noqa

llm = {
    "size": cfg.getint(section="model", option="model.max.length"),
    "epoch": cfg.getint(section="model", option="num.train.epochs"),
    "batch": cfg.getint(section="model", option="per.device.train.batch.size")
}
