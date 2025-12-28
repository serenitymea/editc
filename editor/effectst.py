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
