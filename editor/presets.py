import random

class VideoEffects:
    def __init__(self, clip, speed):
        self.clip = clip
        self.speed = speed
        self.width = 1920
        self.height = 1080

    def get_preset(self, clip_index=0):
        duration = self.clip.end - self.clip.start
        if duration < 0.5:
            return self.clean_cut
        if hasattr(self.clip, "beat_type") and self.clip.beat_type == "hard":
            return self.beat_zoom
        if hasattr(self.clip, "beat_type") and self.clip.beat_type == "soft":
            return self.smooth_zoom_push
        if clip_index % 4 == 0:
            return self.cinematic_fade
        return self.clean_cut

    def beat_zoom(self, i):
        return self._base(i, extra="""
        scale=iw*1.05:ih*1.05,
        zoompan=z='1+0.01*on':d=1,
        """)

    def cinematic_fade(self, i):
        return self._base(i, extra="""
        eq=contrast=1.05:saturation=1.05,
        """)

    def smooth_zoom_push(self, i):
        return self._base(i, extra="""
        scale=iw*1.04:ih*1.04,
        """)

    def motion_blur_swipe(self, i):
        return self._base(i, extra="""
        tmix=frames=3:weights='1 2 1',
        """)

    def flash_white(self, i):
        return self._base(i, extra="""
        eq=brightness=0.25:saturation=0.9,
        """)

    def blur_crossfade(self, i):
        return self._base(i, extra="""
        gblur=sigma=15,
        """)

    def clean_cut(self, i):
        return self._base(i, extra="")

    def _base(self, i, extra):
        return f"""
        [0:v]trim=start={self.clip.start}:end={self.clip.end},
        setpts=PTS-STARTPTS,
        setpts=PTS/{self.speed},
        {extra}
        scale={self.width}:{self.height},
        setsar=1,
        format=yuv420p[v{i}]
        """

    def next_transition(self):
        pool = [
            ("fade", 0.4),
            ("wipeleft", 0.35),
            ("zoom", 0.4),
            ("fadewhite", 0.25),
        ]
        return random.choice(pool)
