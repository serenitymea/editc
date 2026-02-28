import os
import json
import yaml
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


PARAM_RANGES = {
    "scoring": {
        "heuristic_weights": {
            "motion_p90":        (0.05, 0.60),
            "motion_peak_ratio": (0.01, 0.15),
            "beat_alignment":    (0.05, 0.40),
            "bass_energy":       (0.05, 0.40),
            "blur_inverse":      (0.01, 0.20),
            "buildup":           (0.05, 0.50),
            "rms_peak":          (0.05, 0.40),
        },
        "global_weights": {
            "motion":  (0.20, 0.70),
            "audio":   (0.15, 0.60),
            "visual":  (0.02, 0.25),
            "context": (0.02, 0.30),
        },
    },
    "windows": {
        "motion_window_seconds":     (0.5, 10.0),
        "temporal_window_frames":    (2, 20),
        "clip_search_step_seconds":  (0.05, 1.0),
        "flow_sample_fps_divider":   (2, 20),
        "visual_sample_fps_divider": (2, 20),
    },
    "optical_flow": {
        "pyr_scale":  (0.3, 0.8),
        "levels":     (1, 6),
        "winsize":    (5, 30),
        "iterations": (1, 10),
        "poly_n":     (3, 9),
        "poly_sigma": (0.5, 2.5),
        "flags":      (0, 0),
    },
    "video_processing": {
        "resize_height":      (240, 1080),
        "symmetry_max_width": (40, 200),
    },
    "face_detection": {
        "scale_factor":  (1.1, 1.8),
        "min_neighbors": (1, 15),
    },
    "clip_selection": {
        "overlap_decay_ratio": (0.1, 1.0),
    },
}


def deep_update(base, updates, path=None):
    if path is None:
        path = []
    for k, v in updates.items():
        if k not in base:
            continue
        if isinstance(v, dict) and isinstance(base[k], dict):
            deep_update(base[k], v, path + [k])
        elif isinstance(v, bool):
            base[k] = v
        elif isinstance(v, list):
            base[k] = v
        elif isinstance(v, (int, float)):
            node = PARAM_RANGES
            for p in path:
                node = node.get(p, {})
            bounds = node.get(k)
            if bounds:
                base[k] = type(v)(max(bounds[0], min(bounds[1], v)))
        elif isinstance(v, str):
            try:
                v = float(v) if "." in v else int(v)
                deep_update(base, {k: v}, path)
            except Exception:
                pass


class PromptConfigGenerator:
    def __init__(self, model="gpt-4.1-mini", max_attempts=5, retry_delay=1.0, debug=True):
        self.config_path = Path.cwd() / "config.yaml"
        self.model = model
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.debug = debug

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        if not self.client:
            print("OPENAI_API_KEY not f")

    def generate(self, video_prompt: str = None) -> str:
        template = self._template()

        if self.client and video_prompt:
            schema = json.dumps(template, indent=2)
            ranges = json.dumps(PARAM_RANGES, indent=2)
            prompt = (
                f"Tune these JSON parameters for the described video.\n"
                f"SCHEMA (structure is immutable):\n{schema}\n\n"
                f"ALLOWED RANGES:\n{ranges}\n\n"
                f"Rules: return ONLY valid JSON, same structure, only change numbers/booleans within ranges.\n\n"
                f"Video: {video_prompt}"
            )

            for attempt in range(1, self.max_attempts + 1):
                print(f"attemp {attempt}...")
                try:
                    raw = self.client.chat.completions.create(
                        model=self.model,
                        temperature=0.3,
                        messages=[
                            {"role": "system", "content": "Return ONLY valid JSON. No text, no markdown."},
                            {"role": "user", "content": prompt},
                        ],
                    ).choices[0].message.content.strip().replace("```json", "").replace("```", "")

                    if self.debug:
                        print(raw)

                    deep_update(template, json.loads(raw))
                    break
                except Exception as e:
                    print(f"{e}")
                    if attempt < self.max_attempts:
                        time.sleep(self.retry_delay)
                    else:
                        print("base")

        template["ml"] = {
            "feature_order": [
                "motion_mean", "motion_max", "motion_p90", "motion_std", "motion_peak_ratio",
                "rms_mean", "rms_peak", "rms_contrast", "spectral_centroid", "spectral_rolloff",
                "mfcc_variance", "bass_energy", "vocal_probability", "onset_density", "beat_alignment_score",
                "blur_score", "edge_density", "color_variance", "avg_brightness", "contrast_mean",
                "face_present", "face_size_ratio", "symmetry", "rule_of_thirds", "text_presence",
                "motion_momentum", "audio_momentum", "combined_buildup", "relative_position",
                "motion_derivative", "duration",
            ]
        }
        template["audio_loops"] = {"default": 1}

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(template, f, allow_unicode=True, sort_keys=False)

        return str(self.config_path)

    def _template(self) -> dict:
        return {
            "scoring": {
                "heuristic_weights": {
                    "motion_p90": 0.30, "motion_peak_ratio": 0.02, "beat_alignment": 0.18,
                    "bass_energy": 0.15, "blur_inverse": 0.07, "buildup": 0.20, "rms_peak": 0.15,
                },
                "global_weights": {
                    "motion": 0.42, "audio": 0.38, "visual": 0.08, "context": 0.12,
                },
            },
            "windows": {
                "motion_window_seconds": 2.5, "temporal_window_frames": 5,
                "clip_search_step_seconds": 0.2, "flow_sample_fps_divider": 8,
                "visual_sample_fps_divider": 5,
            },
            "optical_flow": {
                "pyr_scale": 0.5, "levels": 3, "winsize": 12,
                "iterations": 3, "poly_n": 5, "poly_sigma": 1.2, "flags": 0,
            },
            "video_processing": {
                "resize_height": 480, "flow_resolution": [320, 180], "symmetry_max_width": 80,
            },
            "face_detection": {
                "enabled": False, "scale_factor": 1.3, "min_neighbors": 5,
            },
            "clip_selection": {
                "overlap_decay_ratio": 0.6,
            },
        }