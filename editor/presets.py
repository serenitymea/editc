class VideoEffects:
    def __init__(self, clip, speed):
        self.clip = clip
        self.speed = speed
        self.width = 1280
        self.height = 720

    def get_preset(self, score):
        if score >= 18:
            return self.impact
        if score >= 13:
            return self.beat_zoom
        return self.clean_cut
    
    def clean_cut(self, i):
        return f"""
        [0:v]trim=start={self.clip.start}:end={self.clip.end},
        setpts=PTS-STARTPTS,
        setpts=PTS/{self.speed},
        scale={self.width}:{self.height},
        setsar=1,
        format=yuv420p[v{i}]
        """

    def beat_zoom(self, i):
        return f"""
        [0:v]trim=start={self.clip.start}:end={self.clip.end},
        setpts=PTS-STARTPTS,
        setpts=PTS/{self.speed},
        scale=iw*1.05:ih*1.05,
        zoompan=z='1+0.01*on':d=1,
        scale={self.width}:{self.height},
        setsar=1,
        format=yuv420p[v{i}]
        """

    def impact(self, i):
        return f"""
        [0:v]trim=start={self.clip.start}:end={self.clip.end},
        setpts=PTS-STARTPTS,
        setpts=PTS/{self.speed},
        crop=iw-20:ih-20:
        x='10*sin(2*PI*t*25)':
        y='10*cos(2*PI*t*25)',
        scale={self.width}:{self.height},
        setsar=1,
        format=yuv420p[v{i}]
        """
