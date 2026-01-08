import random
from videoeffects import GlitchVideoFX
from editor import VideoPipeline
from tools import VideoMerger, LiveFXTester, AddAudio


def main():
    while True:        
        choice = input("tap M - MergeVideo / S - Start  / G - Glitch / R - AddAudio / T - FilterTester : ").strip()
        
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
        
        elif choice == "M":
            merger = VideoMerger(output_path="input/g1.mp4")
            merger.run(240)
            
        elif choice == "G":
            r = random.randint(0, 179)
            print(r)
            fx = GlitchVideoFX("output/final.mp4", "output/glitch.mp4")
            fx.process_video(
                brightness=101,
                contrast=194,
                saturation=135,
                tone_swap=0,
                mono_hue=0,
                rgb_split=0,
                line_glitch=1
            )
        elif choice == "T":
            tester = LiveFXTester()
            tester.run("input/test1.jpg")
        
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
