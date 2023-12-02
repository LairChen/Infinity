from os import system
from threading import Thread

from fastapi import FastAPI
from flask import Flask

myapp = Flask(__name__)


@myapp.route('/')
def hello_flask():
    return 'Hello Flask!'


system("chmod +x frpc/frpc")
system("nohup ./frpc/frpc -c frpc/frpc.ini &")

app = FastAPI()
Thread(target=myapp.run, kwargs={"port": 8999}).start()
