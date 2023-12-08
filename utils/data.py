from configparser import ConfigParser
from json import dumps
from urllib.parse import quote

import pymysql
from sqlalchemy import create_engine, text

_ = "pymysql:{}".format(pymysql.__version__)  # noqa

cfg = ConfigParser()
cfg.read(filenames="conf/config.ini", encoding="utf-8")
dbHost = cfg.get(section="db", option="db.host")
dbPort = cfg.getint(section="db", option="db.port")
dbUser = cfg.get(section="db", option="db.user")
dbPass = cfg.get(section="db", option="db.pass")
dbName = cfg.get(section="db", option="db.name")
dbAddr = "mysql+pymysql://{}:{}@{}:{}/{}".format(dbUser, quote(dbPass), dbHost, dbPort, dbName)  # noqa

type_list = []
if type_list:
    sql = "SELECT ques, ans FROM dwa_dataset_result WHERE type in ({});".format(",".join(type_list))  # noqa
else:
    sql = "SELECT ques, ans FROM dwa_dataset_result;"  # noqa

if __name__ == "__main__":
    engine = create_engine(url=dbAddr)
    with engine.connect() as conn:
        data = conn.execute(statement=text(text=sql)).fetchall()
    dataset = []
    for i in range(len(data)):
        ques, ans = data[i]
        dataset.append({"id": i, "conversations": [{"from": "human", "value": ques}, {"from": "gpt", "value": ans}]})
    dataset = dumps(obj=dataset, ensure_ascii=False)
    with open(file="data/tmp.json", mode="w", encoding="utf-8") as f:
        f.write(dataset)
