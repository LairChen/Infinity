from configparser import ConfigParser

cfg = ConfigParser()
cfg.read(filenames="conf/config.ini", encoding="utf-8")

llm = {
    "size": cfg.getint(section="model", option="model.max.length"),
    "epoch": cfg.getint(section="model", option="num.train.epochs"),
    "batch": cfg.getint(section="model", option="per.device.train.batch.size")
}
