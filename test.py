from threading import Thread

from fastapi import FastAPI
from flask import Flask

myapp = Flask(__name__)


@myapp.route('/')
def hello_flask():
    return 'Hello Flask!'


app = FastAPI()
Thread(target=myapp.run, kwargs={"port": 8999}).start()
