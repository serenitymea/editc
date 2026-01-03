import random
from typing import List

class EffectTemplate:
    
    @staticmethod
    def color_grade_cinematic() -> str:
        return "eq=contrast=1.15:brightness=-0.05:saturation=1.1,curves=preset=darker"
    
    @staticmethod
    def color_grade_vintage() -> str:
        return "eq=contrast=0.9:saturation=0.7,curves=vintage"
    
    @staticmethod
    def color_grade_vibrant() -> str:
        return "eq=contrast=1.1:saturation=1.3:gamma=1.1"
    
    @staticmethod
    def zoom_in_smooth() -> str:
        return "zoompan=z='min(zoom+0.002,1.15)':d=1:s=1920x1080"
    
    @staticmethod
    def zoom_out_smooth() -> str:
        return "zoompan=z='if(lte(zoom,1.0),1.15,max(1.001,zoom-0.002))':d=1:s=1920x1080"
    
    @staticmethod
    def pan_left() -> str:
        return "crop=iw*0.9:ih:x='(iw-ow)*(1-t/10)':y=0"
    
    @staticmethod
    def pan_right() -> str:
        return "crop=iw*0.9:ih:x='(iw-ow)*(t/10)':y=0"
    
    @staticmethod
    def shake_subtle() -> str:
        return "crop=iw-10:ih-10:x='5+5*sin(n/5)':y='5+5*cos(n/7)'"
    
    @staticmethod
    def vignette() -> str:
        return "vignette=angle=PI/4"
    
    @staticmethod
    def sharpen() -> str:
        return "unsharp=5:5:1.0:5:5:0.0"
    
    @staticmethod
    def blur_light() -> str:
        return "gblur=sigma=1.5"
    
    @staticmethod
    def glow() -> str:
        return "eq=brightness=0.1:saturation=1.2,unsharp=5:5:0.5"
    
    @staticmethod
    def glitch() -> str:
        return "rgbashift=rh=2:gh=-2"
    
    @staticmethod
    def chromatic_aberration() -> str:
        return "split[a][b];[a]lutrgb=r=0[r];[b]lutrgb=b=0[b];[r][b]blend=all_mode=addition"
    
    @staticmethod
    def speed_ramp_in() -> str:
        return "setpts='if(lt(N,30),PTS-STARTPTS+(N*0.01),PTS-STARTPTS)'"
    
    @staticmethod
    def speed_ramp_out() -> str:
        return "setpts='PTS-STARTPTS+if(gt(N,TB*5),N*0.01,0)'"
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
        
