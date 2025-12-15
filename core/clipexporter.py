from moviepy import VideoFileClip, concatenate_videoclips


class ClipExporter:
    def __init__(self, video_path):
        self.video_path = video_path
        self.clip = VideoFileClip(video_path)
    
    def export(self, frames, fps, output_path="output/epic_clips.mp4", window=30):
        clips = []
        for f in frames:
            start = max(f/fps - window/2/fps, 0)
            end = min(f/fps + window/2/fps, self.clip.duration)
            clips.append(self.clip.subclipped(start, end))
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, codec="libx264")
        
    def close(self):
        self.clip.close()
