import json
from typing import Dict, List, Tuple


class DatasetBuilder:
    def __init__(self):
        self.samples: List[Dict] = []

    def add_sample(self, features: Dict[str, float], label: int) -> None:
        self.samples.append({
            "x": list(features.values()),
            "y": int(label)
        })

    def get_xy(self) -> Tuple[List[List[float]], List[int]]:
        X = [s["x"] for s in self.samples]
        y = [s["y"] for s in self.samples]
        return X, y

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.samples, f, indent=2)
