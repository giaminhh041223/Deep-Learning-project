import time

class FPS:
    def __init__(self):
        self.t0 = time.time()
        self.fps = 0.0
        self.n = 0
    def tick(self):
        self.n += 1
        if self.n % 10 == 0:
            t = time.time()
            dt = t - self.t0
            if dt > 0:
                self.fps = 10.0 / dt
            self.t0 = t
        return self.fps