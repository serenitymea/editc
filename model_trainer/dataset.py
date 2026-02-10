import json
import numpy as np
from typing import Dict, List, Tuple, Optional


class DatasetBuilder:
    def __init__(self):
        self.samples: List[Dict] = []

    def add_sample(self, features: Dict[str, float], label: int) -> None:
        """Add a labeled sample to the dataset"""
        self.samples.append({
            "x": list(features.values()),
            "y": int(label)
        })

    def get_xy(self) -> Tuple[List[List[float]], List[int]]:
        """Get features and labels as separate lists"""
        X = [s["x"] for s in self.samples]
        y = [s["y"] for s in self.samples]
        return X, y

    def save(self, path: str) -> None:
        """Save dataset to JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.samples, f, indent=2)
        print(f"[Dataset] Saved {len(self.samples)} samples to {path}")
    
    def load(self, path: str) -> None:
        """Load dataset from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        print(f"[Dataset] Loaded {len(self.samples)} samples from {path}")
    
    def generate_hard_negatives(self, positive_clips, all_features, fps) -> None:
        """
        Generate hard negative examples by:
        1. Shifting clips in time
        2. Truncating good clips
        3. Extending good clips with boring parts
        """
        print(f"[Dataset] Generating hard negatives from {len(positive_clips)} positives...")
        
        hard_negatives = []
        
        for clip in positive_clips:

            for offset in [-2, -1, +1, +2]:
                shifted_start = clip.start_frame + int(offset * fps * 0.5)
                shifted_end = clip.end_frame + int(offset * fps * 0.5)
                
                if 0 <= shifted_start < len(all_features) and shifted_end < len(all_features):
                    shifted_features = self._aggregate_features(
                        all_features, shifted_start, shifted_end
                    )
                    hard_negatives.append(shifted_features)

            if clip.features:
                truncated = self._truncate_peak(clip, all_features)
                if truncated:
                    hard_negatives.append(truncated)

        for features in hard_negatives:
            self.add_sample(features, 0)
        
        print(f"[Dataset] Added {len(hard_negatives)} hard negatives")
    
    def generate_easy_negatives(self, all_features, count=None) -> None:
        """
        Generate easy negative examples:
        1. Low motion + low audio
        2. High blur (out of focus)
        3. Random low-scoring segments
        """
        print(f"[Dataset] Generating easy negatives...")
        
        if count is None:
            count = len([s for s in self.samples if s["y"] == 1])
        
        easy_negatives = []

        for i in range(0, len(all_features), max(1, len(all_features) // (count * 2))):
            feat = all_features[i]

            is_boring = (
                feat.get("motion_mean", 0) < 0.2 and
                feat.get("rms_mean", 0) < 0.2
            )

            is_blurry = feat.get("blur_score", 0) > 0.7
            
            if is_boring or is_blurry:
                easy_negatives.append(feat)
                
                if len(easy_negatives) >= count:
                    break

        for features in easy_negatives[:count]:
            self.add_sample(features, 0)
        
        print(f"[Dataset] Added {len(easy_negatives[:count])} easy negatives")
    
    def _aggregate_features(self, all_features, start_frame, end_frame):
        """Aggregate features over a frame range"""
        segment = all_features[start_frame:end_frame]
        
        if not segment:
            return all_features[start_frame] if start_frame < len(all_features) else {}

        aggregated = {}
        for key in segment[0].keys():
            values = [f[key] for f in segment]
            aggregated[key] = float(np.mean(values))
        
        return aggregated
    
    def _truncate_peak(self, clip, all_features):
        """Create a negative by cutting off the peak of a good clip"""
        if not clip.features:
            return None

        duration_frames = clip.end_frame - clip.start_frame
        truncate_frames = int(duration_frames * 0.3)
        
        new_end = clip.end_frame - truncate_frames
        
        if new_end <= clip.start_frame:
            return None
        
        return self._aggregate_features(all_features, clip.start_frame, new_end)
    
    def balance_dataset(self, target_ratio=1.0) -> None:
        """
        Balance positive and negative samples.
        
        Args:
            target_ratio: Desired ratio of negatives to positives (1.0 = equal)
        """
        X, y = self.get_xy()
        y = np.array(y)
        
        n_positive = (y == 1).sum()
        n_negative = (y == 0).sum()
        
        print(f"[Dataset] Before balance: {n_positive} pos, {n_negative} neg")
        
        target_negatives = int(n_positive * target_ratio)
        
        if n_negative > target_negatives:

            neg_indices = np.where(y == 0)[0]
            keep_indices = np.random.choice(neg_indices, target_negatives, replace=False)
            pos_indices = np.where(y == 1)[0]
            
            keep_all = np.concatenate([pos_indices, keep_indices])
            keep_all.sort()
            
            self.samples = [self.samples[i] for i in keep_all]
        
        X, y = self.get_xy()
        y = np.array(y)
        print(f"[Dataset] After balance: {(y == 1).sum()} pos, {(y == 0).sum()} neg")
    
    def split_train_val(self, val_ratio=0.2) -> Tuple[Tuple[List, List], Tuple[List, List]]:
        """
        Split dataset into train and validation sets.
        
        Returns:
            ((X_train, y_train), (X_val, y_val))
        """
        X, y = self.get_xy()
        
        n_samples = len(X)
        n_val = int(n_samples * val_ratio)

        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_val = [X[i] for i in val_indices]
        y_val = [y[i] for i in val_indices]
        
        print(f"[Dataset] Split: {len(X_train)} train, {len(X_val)} val")
        
        return (X_train, y_train), (X_val, y_val)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        X, y = self.get_xy()
        y = np.array(y)
        
        return {
            "total_samples": len(X),
            "num_features": len(X[0]) if X else 0,
            "positive_samples": int((y == 1).sum()),
            "negative_samples": int((y == 0).sum()),
            "positive_ratio": float((y == 1).sum() / len(y)) if len(y) > 0 else 0.0
        }
    
    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_statistics()
        print("\n[Dataset] Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Features per sample: {stats['num_features']}")
        print(f"  Positive samples: {stats['positive_samples']}")
        print(f"  Negative samples: {stats['negative_samples']}")
        print(f"  Positive ratio: {stats['positive_ratio']:.2%}")