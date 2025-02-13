import logging
import json
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
import os

class JsonFormatter(logging.Formatter):
    def format(self,record):
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat()+"Z",
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
        }
        if isinstance(record.msg, dict):
            log_record.update({k: v for k, v in record.msg.items() if k in ("sim_time", "step", "run_id", "data", "event")})
        return json.dumps(log_record)

class MessageFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, dict):
             return f"\033[1;31m[{record.levelname}]\033[0m \033[33m{record.name}\033[0m \033[94m{record.funcName}\033[0m - {record.msg.get('message', '')}"
        return super().format(record)

class Logger:
    def __init__(self, name, run_id=None, file_prefix=None, log_dir="logs", level=logging.DEBUG, file_max_bytes=10*1024*1024, backup_count=20):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate=False

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id=run_id

        date_str = datetime.now().strftime("%Y-%m-%d-%H%M")
        log_dir = os.path.join(log_dir, date_str)
        os.makedirs(log_dir, exist_ok=True)

        if file_prefix is None:
            file_name = f"simulation_{run_id}.log"
        else:
            file_name = f"{file_prefix}_{run_id}.log"
        full_path = os.path.join(log_dir, file_name)

        print(f"Object: {name}, File: {file_prefix}")

        file_handler=RotatingFileHandler(full_path, maxBytes=file_max_bytes, backupCount=backup_count, delay=True)
        file_handler.setFormatter(JsonFormatter())
        file_handler.setLevel(logging.DEBUG)
        file_handler.addFilter(lambda record: isinstance(record.msg, dict) and "event" in record.msg)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(MessageFormatter())
        stream_handler.setLevel(logging.INFO)
        stream_handler.addFilter(lambda record: isinstance(record.msg, dict) and "message" in record.msg)
        self.logger.addHandler(stream_handler)

    def get(self):
        return self.logger
