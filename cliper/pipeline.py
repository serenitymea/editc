from __future__ import annotations

import queue
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

_SENTINEL = None

class _IOReader(threading.Thread):

    def __init__(
        self,
        loader,
        frames_count: int,
        out_queue: queue.Queue,
        flow_step: int,
        visual_step: int,
    ):
        super().__init__(name="IO_READER", daemon=True)
        self.loader        = loader
        self.frames_count  = frames_count
        self.out_queue     = out_queue
        self.flow_step     = flow_step
        self.visual_step   = visual_step
        self.error: Optional[Exception] = None

    def run(self):
        try:
            for i, frame in self.loader.frames():
                if i >= self.frames_count:
                    break

                need_flow   = (i % self.flow_step   == 0)
                need_visual = (i % self.visual_step == 0)
                
                if need_flow or need_visual:
                    self.out_queue.put((i, frame.copy(), need_flow, need_visual))
            
        except Exception as e:
            self.error = e
            logger.error("[IO_READER] Error: %s", e)
        finally:
            self.out_queue.put(_SENTINEL)

class _CPUWorkerPool:

    def __init__(
        self,
        config,
        visual_extractor,
        frames_count: int,
        fps: float,
        in_queue: queue.Queue,
        num_visual_workers: int = 4,
    ):
        self.config            = config
        self.visual_extractor  = visual_extractor
        self.frames_count      = frames_count
        self.fps               = fps
        self.in_queue          = in_queue
        self.num_visual_workers = num_visual_workers

        self.motion_scores   = np.zeros(frames_count, dtype=np.float32)
        self.visual_features: List[Tuple[int, dict]] = []
        
        self._vis_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=num_visual_workers,
            thread_name_prefix="VIS_WORKER",
        )
        self._vis_futures = []
        self._prev_gray: Optional[np.ndarray] = None
        self._done = threading.Event()
        self.error: Optional[Exception] = None

    def process_all(self):

        try:
            opt = self.config["optical_flow"]
            flow_res = tuple(self.config["video_processing"]["flow_resolution"])

            while True:
                item = self.in_queue.get()
                if item is _SENTINEL:
                    break
                
                frame_idx, frame, need_flow, need_visual = item

                if frame_idx % 1000 == 0:
                    print(f" [Pipeline]frame {frame_idx}/{self.frames_count}", end="\r")

                if need_flow:
                    gray = cv2.cvtColor(cv2.resize(frame, flow_res), cv2.COLOR_BGR2GRAY)
                    
                    if self._prev_gray is not None:
                        flow = cv2.calcOpticalFlowFarneback(
                            self._prev_gray, gray, None,
                            opt["pyr_scale"], opt["levels"], opt["winsize"],
                            opt["iterations"], opt["poly_n"], opt["poly_sigma"], opt["flags"],
                        )

                        mag = np.hypot(flow[..., 0], flow[..., 1])
                        self.motion_scores[frame_idx] = self._compute_motion_score(mag)
                    
                    self._prev_gray = gray

                if need_visual:

                    future = self._executor.submit(
                        self._extract_visual, frame_idx, frame
                    )
                    self._vis_futures.append(future)

        except Exception as e:
            self.error = e
            logger.error("[CPUWorkerPool] Error: %s", e)
        finally:
            for f in self._vis_futures:
                try:
                    idx, feat = f.result()
                    with self._vis_lock:
                        self.visual_features.append((idx, feat))
                except Exception as e:
                    logger.warning("[CPUWorkerPool] Visual future failed: %s", e)
            
            self._executor.shutdown(wait=False)
            self.visual_features.sort(key=lambda x: x[0])
            print()

    def _extract_visual(self, frame_idx: int, frame: np.ndarray) -> Tuple[int, dict]:

        feat = self.visual_extractor.extract(frame)
        return frame_idx, feat

    @staticmethod
    def _compute_motion_score(mag: np.ndarray) -> float:
        flat = mag.ravel()
        p90  = float(np.percentile(flat, 90))
        return 0.6 * p90 + 0.4 * float(flat.mean())

    def get_results(self):
        return self.motion_scores, self.visual_features

class _AudioPrefetcher(threading.Thread):


    def __init__(
        self,
        audio_analyzer,
        frames_count: int,
        fps: float,
        audio_step: int,
        audio_keys: List[str],
    ):
        super().__init__(name="AUDIO_PREFETCH", daemon=True)
        self.audio         = audio_analyzer
        self.frames_count  = frames_count
        self.fps           = fps
        self.audio_step    = audio_step
        self.audio_keys    = audio_keys

        self.audio_matrix  = np.zeros(
            (frames_count, len(audio_keys) + 1), dtype=np.float32
        )
        self.beat_col      = len(audio_keys)
        self.done          = threading.Event()
        self.error: Optional[Exception] = None

    def run(self):
        try:
            cache: dict = {}
            step = self.audio_step

            for fi in range(self.frames_count):
                gk = fi // step
                if gk not in cache:
                    cache[gk] = self.audio.get_advanced_audio_features(fi, self.frames_count)
                af = cache[gk]

                for ci, ak in enumerate(self.audio_keys):
                    self.audio_matrix[fi, ci] = af.get(ak, 0.0)

                self.audio_matrix[fi, self.beat_col] = \
                    self.audio.get_beat_alignment_score(fi / self.fps)

        except Exception as e:
            self.error = e
            logger.error("[AUDIO_PREFETCH] Error: %s", e)
        finally:
            self.done.set()

class ParallelFeaturePipeline:

    AUDIO_KEYS = ["rms", "rms_peak", "rms_contrast", "bass_energy"]

    def __init__(
        self,
        config,
        loader,
        audio_analyzer,
        visual_extractor,
        frames_count: int,
        fps: float,
        num_workers: int = 4,
        queue_size: int = 64,
    ):
        self.config           = config
        self.loader           = loader
        self.audio            = audio_analyzer
        self.visual_extractor = visual_extractor
        self.frames_count     = frames_count
        self.fps              = fps
        self.num_workers      = num_workers
        self.queue_size       = queue_size

    def run(self) -> Tuple[np.ndarray, List, np.ndarray]:

        flow_step   = max(1, int(self.fps / self.config["windows"]["flow_sample_fps_divider"]))
        visual_step = max(1, int(self.fps / self.config["windows"]["visual_sample_fps_divider"]))
        audio_step  = max(1, int(self.fps / self.config["windows"].get("visual_sample_fps_divider", 4)))

        t0 = time.perf_counter()
        print(f"[Pipeline] Starting parallel pipeline: {self.num_workers} workers, "
              f"queue_size={self.queue_size}, flow_step={flow_step}, visual_step={visual_step}")

        raw_queue = queue.Queue(maxsize=self.queue_size)

        audio_prefetcher = _AudioPrefetcher(
            audio_analyzer=self.audio,
            frames_count=self.frames_count,
            fps=self.fps,
            audio_step=audio_step,
            audio_keys=self.AUDIO_KEYS,
        )
        audio_prefetcher.start()

        io_reader = _IOReader(
            loader=self.loader,
            frames_count=self.frames_count,
            out_queue=raw_queue,
            flow_step=flow_step,
            visual_step=visual_step,
        )
        io_reader.start()

        cpu_pool = _CPUWorkerPool(
            config=self.config,
            visual_extractor=self.visual_extractor,
            frames_count=self.frames_count,
            fps=self.fps,
            in_queue=raw_queue,
            num_visual_workers=self.num_workers,
        )

        cpu_pool.process_all()
        io_reader.join()

        audio_prefetcher.join(timeout=30)
        if not audio_prefetcher.done.is_set():
            logger.warning("[Pipeline] Audio prefetcher timed out — using partial results")

        for worker, name in [(io_reader, "IO_READER"), (audio_prefetcher, "AUDIO_PREFETCH")]:
            if worker.error:
                raise RuntimeError(f"[Pipeline] {name} failed: {worker.error}")
        if cpu_pool.error:
            raise RuntimeError(f"[Pipeline] CPU_POOL failed: {cpu_pool.error}")

        motion_scores, visual_features = cpu_pool.get_results()
        audio_matrix = audio_prefetcher.audio_matrix

        motion_scores = self._postprocess_motion(motion_scores)

        elapsed = time.perf_counter() - t0
        print(f"\n[Pipeline] Done in {elapsed:.1f}s | "
              f"motion samples: {(motion_scores > 0).sum()} | "
              f"visual samples: {len(visual_features)}")

        return motion_scores, visual_features, audio_matrix

    def _postprocess_motion(self, motion_scores: np.ndarray) -> np.ndarray:
        non_zero = motion_scores > 0
        if non_zero.any():
            motion_scores = np.interp(
                np.arange(self.frames_count),
                np.where(non_zero)[0],
                motion_scores[non_zero],
            ).astype(np.float32)

        motion_scores = gaussian_filter1d(motion_scores, sigma=1.5).astype(np.float32)
        
        max_val = motion_scores.max()
        if max_val > 0:
            motion_scores /= max_val
        
        return motion_scores

def optimal_num_workers() -> int:

    import os
    cpu_count = os.cpu_count() or 4
    return max(1, min(cpu_count - 2, 12))