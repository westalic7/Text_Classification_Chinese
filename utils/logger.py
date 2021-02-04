# -*- coding: utf-8 -*-
import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler


class LoggerClass(object):
    def __init__(self, logger_name, logger_file):
        self.logger = self.init_logger(logger_name, logger_file)

    @staticmethod
    def init_logger(logger_name, logger_file, data_fmt="%Y-%m-%d %H:%M:%S", level=logging.INFO):
        if logger_name in Logger.manager.loggerDict:
            return logging.getLogger(logger_name)

        str_fmt = '[%(asctime)s] [%(levelname)s] : %(message)s'
        formatter = logging.Formatter(str_fmt, data_fmt)

        handler = TimedRotatingFileHandler(filename=logger_file, when="D", interval=5, backupCount=5)
        handler.setFormatter(formatter)
        handler.setLevel(level)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(level)

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def info(self, *args, **kwargs):
        return self.logger.info(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self.logger.error(*args, **kwargs)

    def set_level(self, *args, **kwargs):
        return self.logger.setLevel(*args, **kwargs)
