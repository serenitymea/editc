import random
from typing import List
from .effectst import EffectTemplate

class EffectProcessor:
    
    def __init__(self, intensity: float = 1.0, randomize: bool = False):

        self.intensity = max(0.0, min(1.0, intensity))
        self.randomize = randomize
        self.template = EffectTemplate()
        
    def get_effect_for_clip(self, clip, clip_index: int, total_clips: int) -> List[str]:

        effects = []
        
        if self.randomize:
            effects.extend(self._get_random_effects(clip))
        else:
            effects.extend(self._get_pattern_effects(clip, clip_index, total_clips))
        
        return effects
    
    def _get_random_effects(self, clip) -> List[str]:

        effects = []

        color_effects = [
            self.template.color_grade_cinematic,
            self.template.color_grade_vibrant,
            self.template.color_grade_vintage,
        ]
        effects.append(random.choice(color_effects)())

        if random.random() < 0.5 * self.intensity:
            motion_effects = [
                self.template.zoom_in_smooth,
                self.template.zoom_out_smooth,
                self.template.pan_left,
                self.template.pan_right,
            ]
            effects.append(random.choice(motion_effects)())

        if random.random() < 0.3 * self.intensity:
            extra_effects = [
                self.template.vignette,
                self.template.sharpen,
                self.template.glow,
            ]
            effects.append(random.choice(extra_effects)())
        
        return effects
    
    def _get_pattern_effects(self, clip, clip_index: int, total_clips: int) -> List[str]:
        effects = []
        score = getattr(clip, 'score', 5)

        if self.intensity > 0.7:
            effects.append(self.template.color_grade_vibrant())
        elif self.intensity > 0.4:
            effects.append(self.template.color_grade_cinematic())
        else:
            effects.append(self.template.color_grade_vintage())

        position = clip_index / max(total_clips - 1, 1)
        
        if position < 0.25:
            effects.append(self.template.zoom_in_smooth())
            if score > 7:
                effects.append(self.template.sharpen())
        elif position < 0.5:
            if clip_index % 2 == 0:
                effects.append(self.template.pan_right())
            else:
                effects.append(self.template.pan_left())
        elif position < 0.75:
            effects.append(self.template.zoom_out_smooth())
            if score > 8:
                effects.append(self.template.glow())
                effects.append(self.template.vignette())
        else:
            effects.append(self.template.vignette())
            if clip_index == total_clips - 1:
                effects.append(self.template.blur_light())

        if score > 8:
            if random.random() < 0.3:
                effects.append(self.template.shake_subtle())
        
        return effects
    
    def apply_effects(self, base_filter: str, effects: List[str]) -> str:

        if not effects:
            return base_filter

        base_filter = base_filter.strip()

        effects_str = ",\n        ".join(effects)

        parts = base_filter.split("scale=")
        if len(parts) == 2:
            return f"{parts[0]}{effects_str},\n        scale={parts[1]}"
        else:
            return f"{base_filter},\n        {effects_str}"