from .ecore import VideoRenderer
from .adapter import EditClip
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
        """
        Args:
            input_video: Путь к входному видео
            output_video: Путь к выходному видео
            music_file: Путь к музыкальному файлу
            bpm: Темп музыки (удары в минуту)
            beats_per_clip: Количество ударов на клип
            use_effects: Включить систему эффектов
            effect_intensity: Интенсивность эффектов (0.0 - 1.0)
            randomize_effects: Использовать случайные эффекты вместо паттернов
        """
        self.input_video = input_video
        self.output_video = output_video
        self.bpm = bpm
        self.beats_per_clip = beats_per_clip
        self.music_file = music_file
        
        # Инициализация pipeline для анализа клипов
        self.pipeline = ClipP(
            video_path="input/g1.mp4",
            music_path="input/m1.mp3"
        )
        
        # Создаем процессор эффектов, если включено
        effect_processor = None
        if use_effects:
            effect_processor = EffectProcessor(
                intensity=effect_intensity,
                randomize=randomize_effects
            )
        
        # Инициализация рендерера с процессором эффектов
        self.render = VideoRenderer(
            input_video=self.input_video,
            output_video=self.output_video,
            bpm=self.bpm,
            beats_per_clip=self.beats_per_clip,
            music_file=self.music_file,
            effect_processor=effect_processor
        )
        
    def run(self):
        """
        Запуск полного пайплайна обработки видео
        
        Returns:
            Путь к выходному видео файлу
        """
        # Получаем клипы из анализатора
        raw_clips = self.pipeline.run()

        # Оборачиваем клипы в адаптер
        clips = (EditClip(c) for c in raw_clips)

        # Рендерим видео с применением эффектов
        self.render.render_edit(clips)

        return self.output_video