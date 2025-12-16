from moviepy import VideoFileClip, concatenate_videoclips


class ClipExporter:
    def __init__(self, video_path):
        self.video = VideoFileClip(video_path)

    def export(
        self,
        clips,
        output_path="output/epic_clips.mp4",
        pad=0.3,
        merge_gap=0.5
    ):
        result = []
        merged = self._merge_clips(clips, merge_gap)

        for clip in merged:
            start = max(clip.start_time - pad, 0)
            end = min(clip.end_time + pad, self.video.duration)
            result.append(self.video.subclipped(start, end))

        if result:
            final = concatenate_videoclips(result, method="compose")
            final.write_videofile(output_path, codec="libx264")

    def _merge_clips(self, clips, gap):
        if not clips:
            return []

        clips = sorted(clips, key=lambda c: c.start_time)
        merged = [clips[0]]

        for c in clips[1:]:
            last = merged[-1]
            if c.start_time - last.end_time <= gap:
                last.end_frame = max(last.end_frame, c.end_frame)
                last.score = max(last.score, c.score)
            else:
                merged.append(c)

        return merged

    def close(self):
        self.video.close()
