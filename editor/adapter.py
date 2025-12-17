class EditClip:
    def __init__(self, clip):
        self._clip = clip

    @property
    def start_frame(self):
        return self._clip.start_frame

    @property
    def end_frame(self):
        return self._clip.end_frame

    @property
    def fps(self):
        return self._clip.fps

    @property
    def score(self):
        return getattr(self._clip, "score", 10)

    @property
    def start(self):
        return self.start_frame / self.fps

    @property
    def end(self):
        return self.end_frame / self.fps

    @property
    def duration(self):
        return self.end - self.start
