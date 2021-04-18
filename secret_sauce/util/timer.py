from datetime import datetime


class Timer():

    def __init__(self) -> None:
        self.start_time = 0
        self.stop_time = 0

    def start(self):
        self.start_time = datetime.now()

    def stop_print(self):
        self.stop_time = datetime.now()
        diff = self.stop_time - self.start_time
        print(diff.total_seconds())
