import random
from videoeffects import GlitchVideoFX
from editor import VideoPipeline
from tools import AnimeEditAgent, VideoMerger, LiveFXTester, AddAudio


def main():
           
    choice = input("tap M - MergeVideo / A - AiAgent / S - Start  / G - Glitch / R - AddAudio / T - FilterTester : ").strip()
    
    if choice == "S":    
        vp = VideoPipeline(
            input_video="input/g1.mp4",
            output_video="output/final.mp4",
            music_file="input/m1.mp3",
            bpm=120,
            beats_per_clip=16
        )

        vp.run()
        print("end")
        
    elif choice == "A":
        agent = AnimeEditAgent()
        agent.run()
    
    elif choice == "M":
        merger = VideoMerger()
        merger.run(85)
        
    elif choice == "G":
        r = random.randint(0, 179)
        print(r)
        fx = GlitchVideoFX("output/final.mp4", "output/glitch.mp4")
        fx.process_video(
            brightness=90,
            contrast=96,
            saturation=48,
            tone_swap=0,
            mono_hue=r,
            rgb_split=0,
            line_glitch=2
        )
    elif choice == "T":
        tester = LiveFXTester("input/test.jpg")
        tester.run()
    
    elif choice == "R":
        merger = AddAudio(
            "output/glitch.mp4",
            "input/m1.mp3",
            "output/glitcha.mp4"
        )
        
        merger.merge()
   
    else:
        print("failed")       

if __name__ == "__main__":
    main()
