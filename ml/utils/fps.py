import time

class FPS:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.num_frames = 0

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self):
        self.end_time = time.time()

    def update(self):
        self.num_frames += 1

    def elapsed(self):
        return (self.end_time - self.start_time) if self.end_time and self.start_time else 0

    def fps(self):
        elapsed_time = self.elapsed()
        return self.num_frames / elapsed_time if elapsed_time > 0 else 0