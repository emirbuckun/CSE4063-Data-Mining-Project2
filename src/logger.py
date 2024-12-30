import sys
import os

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        if not os.path.exists('logs'):
            os.makedirs('logs')
        self.log = open(os.path.join('logs', filename), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def set_logger(filename):
    sys.stdout = Logger(filename + ".txt")
