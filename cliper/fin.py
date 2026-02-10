import subprocess
from typing import List, Optional
from .vinloader import VideoLoader
from .audioanalyzer import AudioAnalyzer
from .epicdetector import EpicDetector

import glob
import joblib

def _get_mlmodel():
    """Load the latest trained model"""
    model_files = sorted(glob.glob("model_output/epic_model_*.pkl"))

    if not model_files:
        print("[ClipP] No trained model found - using heuristic scoring")
        return None

    latest_model_path = model_files[-1]
    model = joblib.load(latest_model_path)
    print(f"[ClipP] Loaded model: {latest_model_path}")
    return model


class ClipP:
    """
    Main clip detection pipeline with ML integration.
    
    Usage:
        processor = ClipP(video_path, music_path)
        clips = processor.run(max_clips=10)
    """

    def __init__(self, video_path: str, music_path: str):
        self.video_path = video_path
        self.music_path = music_path
        
        print("\n[ClipP] Initializing...")

        self.model = _get_mlmodel()

        self.loader = VideoLoader(video_path)
        self.audio = AudioAnalyzer(video_path, music_path)
        self.detector = EpicDetector(self.loader, self.audio, self.model)
        
        print("[ClipP] Ready to detect clips")

    def run(
        self,
        target_duration: Optional[float] = None,
        max_clips: Optional[int] = None,
    ) -> List:
        """
        Detect epic clips from the video.
        
        Args:
            target_duration: Total duration of clips to generate (in seconds)
                           If None, uses the full music duration
            max_clips: Maximum number of clips to return
                      If None, determined by target_duration and beat intervals
        
        Returns:
            List of Clip objects with start_frame, end_frame, score, and features
        """

        if target_duration is None:
            target_duration = self._get_audio_duration()
            print(f"[ClipP] Target duration: {target_duration:.2f}s (from music)")

        if max_clips is None and target_duration:
            beat_intervals = self.audio.get_beat_intervals()
            cumulative = 0
            for i, (_, duration) in enumerate(beat_intervals):
                cumulative += duration
                if cumulative >= target_duration:
                    max_clips = i + 1
                    break
            print(f"[ClipP] Max clips: {max_clips} (from beat intervals)")
        
        try:

            clips = self.detector.detect_perfect_clips(max_clips=max_clips)

            print(f"\n[ClipP] ✓ Found {len(clips)} clips")
            if clips:
                total = sum(c.duration for c in clips)
                avg_score = sum(c.score for c in clips) / len(clips)
                print(f"[ClipP]   Total duration: {total:.2f}s")
                print(f"[ClipP]   Average score: {avg_score:.3f}")

                if len(clips) > 1:
                    min_score = min(c.score for c in clips)
                    max_score = max(c.score for c in clips)
                    print(f"[ClipP]   Score range: {min_score:.3f} - {max_score:.3f}")
            
            return clips
            
        finally:
            self.loader.release()

    def _get_audio_duration(self) -> float:
        """Get the duration of the music file using ffprobe"""
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                self.music_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                f"ffprobe failed: {result.stderr.strip()}"
            )

        return float(result.stdout.strip())
    
    def export_clips_info(self, clips: List, output_path: str = "clips_info.json"):
        """
        Export clip information to JSON for external use.
        
        Args:
            clips: List of Clip objects
            output_path: Path to save JSON file
        """
        import json
        
        clips_data = []
        for i, clip in enumerate(clips):
            clip_info = {
                "index": i,
                "start_time": clip.start,
                "end_time": clip.end,
                "duration": clip.duration,
                "score": clip.score,
                "song_start_time": clip.song_start_time,
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
            }

            if clip.features:
                top_features = {
                    "motion_p90": clip.features.get("motion_p90", 0),
                    "beat_alignment": clip.features.get("beat_alignment_score", 0),
                    "bass_energy": clip.features.get("bass_energy", 0),
                    "blur_score": clip.features.get("blur_score", 0),
                    "combined_buildup": clip.features.get("combined_buildup", 0),
                }
                clip_info["top_features"] = top_features
            
            clips_data.append(clip_info)
        
        with open(output_path, "w") as f:
            json.dump(clips_data, f, indent=2)
        
        print(f"[ClipP] Exported clip info to {output_path}")
    
    def get_feature_importance(self):
        """
        Get feature importance from the ML model if available.
        
        Returns:
            Dict mapping feature names to importance scores, or None
        """
        if self.model is None:
            print("[ClipP] No ML model loaded")
            return None
        
        try:
            # CatBoost model
            if hasattr(self.model, 'get_feature_importance'):
                importance = self.model.get_feature_importance()
                feature_names = self._get_feature_names()
                
                return dict(zip(feature_names, importance))
            
            # LogisticRegression model
            elif hasattr(self.model, 'coef_'):
                coef = self.model.coef_[0]
                feature_names = self._get_feature_names()
                
                return dict(zip(feature_names, coef))
            
        except Exception as e:
            print(f"[ClipP] Could not get feature importance: {e}")
        
        return None
    
    def _get_feature_names(self):
        """Standard feature names in order"""
        return [
            "motion_mean", "motion_max", "motion_p90", "motion_std", "motion_peak_ratio",
            "rms_mean", "rms_peak", "rms_contrast",
            "spectral_centroid", "spectral_rolloff", "mfcc_variance",
            "bass_energy", "vocal_probability", "onset_density",
            "beat_alignment_score",
            "blur_score", "edge_density", "color_variance",
            "avg_brightness", "contrast_mean",
            "face_present", "face_size_ratio",
            "symmetry", "rule_of_thirds", "text_presence",
            "motion_momentum", "audio_momentum", "combined_buildup",
            "relative_position", "motion_derivative",
            "duration"
        ]