from editor import VideoPipeline

def main():
    
    vp = VideoPipeline(
        input_video="input/g1.mp4",
        output_video="output/final.mp4",
        music_file="input/m1.mp3",
        bpm=120,
        beats_per_clip=16
    )

    vp.run()
    print("end")

if __name__ == "__main__":
    main()
