import random

class VideoEffects:
    def __init__(self, clip, speed, effect_processor=None):
        self.clip = clip
        self.speed = speed
        self.width = 1920
        self.height = 1080
        self.effect_processor = effect_processor

    def get_preset(self, clip_index=0, total_clips=1):
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

    def beat_zoom(self, i, total_clips=1):
        return self._base(i, total_clips, extra="""
        scale=iw*1.05:ih*1.05,
        zoompan=z='1+0.01*on':d=1,
        """)

    def cinematic_fade(self, i, total_clips=1):
        return self._base(i, total_clips, extra="""
        eq=contrast=1.05:saturation=1.05,
        """)

    def smooth_zoom_push(self, i, total_clips=1):
        return self._base(i, total_clips, extra="""
        scale=iw*1.04:ih*1.04,
        """)

    def motion_blur_swipe(self, i, total_clips=1):
        return self._base(i, total_clips, extra="""
        tmix=frames=3:weights='1 2 1',
        """)

    def flash_white(self, i, total_clips=1):
        return self._base(i, total_clips, extra="""
        eq=brightness=0.25:saturation=0.9,
        """)

    def blur_crossfade(self, i, total_clips=1):
        return self._base(i, total_clips, extra="""
        gblur=sigma=15,
        """)

    def clean_cut(self, i, total_clips=1):
        return self._base(i, total_clips, extra="")

    def _base(self, i, total_clips, extra):
        base_filter = f"""
        [0:v]trim=start={self.clip.start}:end={self.clip.end},
        setpts=PTS-STARTPTS,
        setpts=PTS/{self.speed},
        {extra}
        scale={self.width}:{self.height},
        setsar=1,
        format=yuv420p[v{i}]
        """

        if self.effect_processor:
            effects = self.effect_processor.get_effect_for_clip(
                self.clip, i, total_clips
            )
            if effects:
                base_filter = self.effect_processor.apply_effects(base_filter, effects)
        
        return base_filter

    def next_transition(self):
        pool = [
            ("fade", 0.4),
            ("wipeleft", 0.35),
            ("zoom", 0.4),
            ("fadewhite", 0.25),
        ]
        return random.choice(pool)