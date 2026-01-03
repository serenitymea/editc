from .ecore import VideoRenderer
from .effectsp import EffectProcessor
from cliper.fin import ClipP


class VideoPipeline:
    def __init__(
        self, 
        input_video: str, 
        output_video: str, 
        music_file: str, 
        bpm: int = 120, 
        beats_per_clip: int = 16,
        use_effects: bool = True,
        effect_intensity: float = 0.7,
        randomize_effects: bool = False
    ):

        self.input_video = input_video
        self.output_video = output_video
        self.bpm = bpm
        self.beats_per_clip = beats_per_clip
        self.music_file = music_file

        self.pipeline = ClipP(
            video_path="input/g1.mp4",
            music_path="input/m1.mp3"
        )

        effect_processor = None
        if use_effects:
            effect_processor = EffectProcessor(
                intensity=effect_intensity,
                randomize=randomize_effects
            )

        self.render = VideoRenderer(
            input_video=self.input_video,
            output_video=self.output_video,
            bpm=self.bpm,
            beats_per_clip=self.beats_per_clip,
            music_file=self.music_file,
            effect_processor=effect_processor
        )
        
    def run(self):

        clips = self.pipeline.run()

        self.render.render_edit(clips)

        return self.output_video