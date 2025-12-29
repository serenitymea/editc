from editor import VideoPipeline
from tools import AnimeEditAgent, VideoMerger

def main():
           
    choice = input("tap M / A / S : ").strip()
    
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
        merger.run()
    
    else:
        print("failed")       

if __name__ == "__main__":
    main()
