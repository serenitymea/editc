from .ecore import VideoRenderer
from .adapter import EditClip
from cliper.fin import ClipP

class VideoPipeline:
    def __init__(self, input_video: str, output_video: str, music_file: str, bpm: int = 120, beats_per_clip: int = 16):
        self.input_video = input_video
        self.output_video = output_video
        self.bpm = bpm
        self.beats_per_clip = beats_per_clip
        self.pipeline = ClipP(input_video)
        self.music_file = music_file
        
        self.render = VideoRenderer(
            input_video=self.input_video,
            output_video=self.output_video,
            bpm=self.bpm,
            beats_per_clip=self.beats_per_clip,
            music_file=self.music_file
        )
        
    def run(self):
        
        raw_clips = self.pipeline.run()

        clips = (EditClip(c) for c in raw_clips)

        self.render.render_edit(clips)

        return self.output_video
