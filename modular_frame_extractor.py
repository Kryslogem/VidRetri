"""
Pipeline Architecture:
  [Pass 1] Uniform Temporal Sampling      -> Parameter Constant: TARGET_FPS
  [Pass 2] Edge Change Ratio Cut Detection -> Parameter Constant: ECR_THRESHOLD
  [Pass 3] Smart Local Blur Filter         -> Parameter Constant: MIN_SHARPNESS
  [Pass 4] Semantic CLIP Clustering        -> Parameter Constant: COSINE_DISTANCE_THRESHOLD
  [Pass 5] Hybrid Visual Deduplication     -> Parameter Constant: SIMILARITY_THRESHOLD
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# PyTorch / CUDA detection
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
except ImportError:
    HAS_TORCH = False
    CUDA_AVAILABLE = False
    DEVICE = None

# Optional Scikit-Learn for Clustering
try:
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional Transformers for CLIP
try:
    from transformers import AutoProcessor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ============================================================================
# PASS CONSTANTS CONFIGURATION
# ============================================================================
@dataclass
class PassConstants:
    """Explicit parameters/constants for each Pass in the extraction pipeline."""
    pass1_target_fps: float = 4.0             # Pass 1: Sampling FPS
    pass2_ecr_threshold: float = 0.18          # Pass 2: Edge Change Ratio Cut Threshold (0.0 to 1.0)
    pass2_suppress_camera_motion: bool = True  # Pass 2: Filter out camera zoom, pan, and hand shake
    pass3_min_sharpness: float = 15.0          # Pass 3: Minimum Laplacian Variance Sharpness
    pass4_cosine_distance: float = 0.1         # Pass 4: CLIP Embedding Cosine Distance Threshold (0.0 to 1.0)
    pass5_similarity_threshold: float = 0.95   # Pass 5: SSIM + dHash + Color Similarity Threshold (0.0 to 1.0)

    @classmethod
    def from_json(cls, json_path: str) -> "PassConstants":
        """Loads constants directly from a JSON configuration file."""
        path = Path(json_path)
        if not path.is_file():
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return cls(
                pass1_target_fps=float(data.get("pass1_target_fps", cls.pass1_target_fps)),
                pass2_ecr_threshold=float(data.get("pass2_ecr_threshold", cls.pass2_ecr_threshold)),
                pass2_suppress_camera_motion=bool(data.get("pass2_suppress_camera_motion", cls.pass2_suppress_camera_motion)),
                pass3_min_sharpness=float(data.get("pass3_min_sharpness", cls.pass3_min_sharpness)),
                pass4_cosine_distance=float(data.get("pass4_cosine_distance", data.get("pass5_cosine_distance", cls.pass4_cosine_distance))),
                pass5_similarity_threshold=float(data.get("pass5_similarity_threshold", data.get("pass4_similarity_threshold", cls.pass5_similarity_threshold)))
            )
        except Exception as e:
            print(f"[WARNING] Could not parse config JSON ({json_path}): {e}. Using defaults.", file=sys.stderr)
            return cls()


# ============================================================================
# MODULE 1: UNIFORM SAMPLING
# ============================================================================
class UniformSamplingModule:
    """Pass 1: Decimates raw video frames down to a specified target sampling FPS."""

    def __init__(self, target_fps: float = 5.0):
        self.target_fps = max(0.01, float(target_fps))

    def process(self, video_path: str, video_metadata: dict) -> List[dict]:
        video_fps = video_metadata["fps"]
        total_frames = video_metadata["total_frames"]
        sample_stride = max(1, int(round(video_fps / self.target_fps)))

        print(f"\n[PASS 1] Uniform Sampling Constant (TARGET_FPS = {self.target_fps}, Stride = {sample_stride})")

        cap = cv2.VideoCapture(video_path)
        sampled_frames = []

        pbar = tqdm(total=total_frames, desc="[Pass 1] Uniform Sampling", unit="frame")
        curr_frame_idx = 0

        while curr_frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp_sec = curr_frame_idx / video_fps
            sampled_frames.append({
                "frame_idx": curr_frame_idx,
                "timestamp_sec": round(timestamp_sec, 3),
                "frame": frame
            })

            curr_frame_idx += 1
            pbar.update(1)

            # Fast skip (sample_stride - 1) frames using grab() without RGB decoding
            for _ in range(sample_stride - 1):
                if curr_frame_idx >= total_frames:
                    break
                if not cap.grab():
                    break
                curr_frame_idx += 1
                pbar.update(1)

        pbar.close()
        cap.release()
        print(f" -> Retained {len(sampled_frames)} / {total_frames} frames ({len(sampled_frames)/total_frames*100:.1f}%).")
        return sampled_frames

# ============================================================================
# MODULE 2:  ECR Cut Detection & Anti-Camera Motion
# ============================================================================
class ECRModule:
    """Pass 2: Detects scene cuts and structural content shifts using Edge Change Ratio while filtering out camera zoom/shake."""

    def __init__(self, ecr_threshold: float = 0.20, suppress_camera_motion: bool = True):
        self.ecr_threshold = ecr_threshold
        self.suppress_camera_motion = suppress_camera_motion

    @staticmethod
    def is_camera_motion_or_zoom(gray1: np.ndarray, gray2: np.ndarray) -> bool:
        """Detects camera zoom in/out, pan, or hand shake using Farneback Optical Flow."""
        h, w = gray1.shape[:2]
        if h > 360:
            scale = 360.0 / h
            g1 = cv2.resize(gray1, (int(w * scale), 360), interpolation=cv2.INTER_AREA)
            g2 = cv2.resize(gray2, (int(w * scale), 360), interpolation=cv2.INTER_AREA)
        else:
            g1, g2 = gray1, gray2

        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]

        # 1. Check Zoom (Radial Optical Flow)
        cy, cx = g1.shape[0] / 2.0, g1.shape[1] / 2.0
        y_indices, x_indices = np.indices(g1.shape)
        rx = x_indices - cx
        ry = y_indices - cy
        r_norm = np.sqrt(rx**2 + ry**2) + 1e-5
        radial_component = (fx * rx + fy * ry) / r_norm

        mean_radial = np.mean(radial_component)
        std_radial = np.std(radial_component)
        if abs(mean_radial) > 0.8 and std_radial < 2.5:
            return True  # Camera Zooming

        # 2. Check Pan / Hand Shake (Uniform Motion Vector)
        mean_fx = float(np.mean(np.abs(fx)))
        mean_fy = float(np.mean(np.abs(fy)))
        std_fx = float(np.std(fx))
        std_fy = float(np.std(fy))

        if (mean_fx > 1.2 or mean_fy > 1.2) and (std_fx < 2.0 and std_fy < 2.0):
            return True  # Camera Shake / Pan

        return False

    @staticmethod
    def compute_ecr(gray1: np.ndarray, gray2: np.ndarray) -> float:
        h, w = gray1.shape[:2]
        if h > 360:
            scale = 360.0 / h
            gray1 = cv2.resize(gray1, (int(w * scale), 360), interpolation=cv2.INTER_AREA)
            gray2 = cv2.resize(gray2, (int(w * scale), 360), interpolation=cv2.INTER_AREA)

        e1 = cv2.Canny(gray1, 50, 150)
        e2 = cv2.Canny(gray2, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        e1_dilated = cv2.dilate(e1, kernel)
        e2_dilated = cv2.dilate(e2, kernel)

        e_in = cv2.bitwise_and(e2, cv2.bitwise_not(e1_dilated))
        e_out = cv2.bitwise_and(e1, cv2.bitwise_not(e2_dilated))

        c1 = np.count_nonzero(e1)
        c2 = np.count_nonzero(e2)

        r_in = np.count_nonzero(e_in) / float(c2) if c2 > 0 else 0.0
        r_out = np.count_nonzero(e_out) / float(c1) if c1 > 0 else 0.0

        return float(max(r_in, r_out))

    def process(self, sampled_frames: List[dict]) -> List[dict]:
        print(f"\n[PASS 2] Edge Change Ratio Cut Detection (ECR_THRESHOLD = {self.ecr_threshold}, AntiCameraMotion = {self.suppress_camera_motion})")
        if not sampled_frames:
            return []

        candidates = [sampled_frames[0]]
        candidates[0]["ecr_score"] = 1.0
        candidates[0]["trigger_reason"] = "initial_frame"

        last_gray = cv2.cvtColor(sampled_frames[0]["frame"], cv2.COLOR_BGR2GRAY)
        skipped_camera_motions = 0

        for i in range(1, len(sampled_frames)):
            item = sampled_frames[i]
            curr_gray = cv2.cvtColor(item["frame"], cv2.COLOR_BGR2GRAY)

            if self.suppress_camera_motion and self.is_camera_motion_or_zoom(last_gray, curr_gray):
                skipped_camera_motions += 1
                continue

            ecr = self.compute_ecr(last_gray, curr_gray)

            if ecr >= self.ecr_threshold:
                item["ecr_score"] = round(ecr, 4)
                item["trigger_reason"] = f"structural_change (ECR: {ecr:.2f})"
                candidates.append(item)
                last_gray = curr_gray

        print(f" -> Retained {len(candidates)} / {len(sampled_frames)} candidates (Filtered {skipped_camera_motions} camera zoom/shake frames).")
        return candidates


# ============================================================================
# MODULE 3: BLUR FILTER
# ============================================================================
class BlurCheckModule:
    """Pass 3: Computes Laplacian variance sharpness and picks crisp frames within local window."""

    def __init__(self, min_sharpness: float = 30.0, search_window: int = 3):
        self.min_sharpness = min_sharpness
        self.search_window = search_window

    @staticmethod
    def compute_sharpness(bgr_img: np.ndarray) -> float:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def process(self, candidates: List[dict], all_sampled_frames: List[dict]) -> List[dict]:
        print(f"\n[PASS 3] Blur Filter Constant (MIN_SHARPNESS = {self.min_sharpness})")
        if not candidates:
            return []

        # CPU Multi-threaded Parallel Sharpness Computation
        max_workers = min(32, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            sharpness_scores = list(executor.map(lambda c: self.compute_sharpness(c["frame"]), candidates))

        for cand, sharp in zip(candidates, sharpness_scores):
            cand["sharpness"] = round(sharp, 2)

        frame_dict = {f["frame_idx"]: idx for idx, f in enumerate(all_sampled_frames)}
        sharp_candidates = []

        for cand in candidates:
            sharpness = cand["sharpness"]
            if sharpness >= self.min_sharpness:
                sharp_candidates.append(cand)
            else:
                # Search nearby frames in the window for a sharper replacement
                curr_pos = frame_dict.get(cand["frame_idx"], -1)
                best_sub = cand
                best_sharp = sharpness

                if curr_pos != -1:
                    left = max(0, curr_pos - self.search_window)
                    right = min(len(all_sampled_frames), curr_pos + self.search_window + 1)
                    for j in range(left, right):
                        sub_img = all_sampled_frames[j]["frame"]
                        sub_sharp = self.compute_sharpness(sub_img)
                        if sub_sharp > best_sharp:
                            best_sharp = sub_sharp
                            best_sub = dict(all_sampled_frames[j])
                            best_sub["sharpness"] = round(sub_sharp, 2)
                            best_sub["trigger_reason"] = cand["trigger_reason"] + f" [SharpReplacement +{j-curr_pos}]"

                if best_sharp >= self.min_sharpness:
                    sharp_candidates.append(best_sub)

        print(f" -> Retained {len(sharp_candidates)} / {len(candidates)} sharp frames ({len(sharp_candidates)/len(candidates)*100:.1f}%).")
        return sharp_candidates


# ============================================================================
# MODULE 4: SEMANTIC CLIP EMBEDDING & CLUSTERING
# ============================================================================
class CLIPClusteringModule:
    """Pass 4: Extracts deep visual embeddings (CLIP or ResNet) and clusters semantic duplicates."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", distance_threshold: float = 0.15):
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.processor = None
        self.model = None

        if HAS_TRANSFORMERS and HAS_TORCH:
            try:
                print(f" Loading CLIP Model ({model_name})...")
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
                self.model.eval()
            except Exception as e:
                print(f" [INFO] CLIP Model load skipped: {e}. Falling back to Color-Perceptual Embeddings.")

    def get_clip_embeddings(self, bgr_images: List[np.ndarray]) -> np.ndarray:
        pil_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in bgr_images]
        inputs = self.processor(images=pil_imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = F.normalize(features, p=2, dim=-1)
        return features.cpu().numpy()

    def get_fallback_embeddings(self, bgr_images: List[np.ndarray]) -> np.ndarray:
        embeds = []
        for img in bgr_images:
            resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).flatten()
            norm_feat = hsv / (np.linalg.norm(hsv) + 1e-6)
            embeds.append(norm_feat)
        return np.array(embeds)

    def process(self, frames: List[dict]) -> List[dict]:
        print(f"\n[PASS 4] Semantic CLIP Clustering Constant (COSINE_DISTANCE_THRESHOLD = {self.distance_threshold})")
        if not frames or len(frames) <= 1:
            return frames

        bgr_imgs = [f["frame"] for f in frames]

        if self.model is not None:
            embeddings = self.get_clip_embeddings(bgr_imgs)
        else:
            embeddings = self.get_fallback_embeddings(bgr_imgs)

        if HAS_SKLEARN:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=self.distance_threshold
            )
            labels = clustering.fit_predict(embeddings)

            cluster_dict = {}
            for idx, label in enumerate(labels):
                cluster_dict.setdefault(label, []).append((idx, frames[idx]["sharpness"]))

            selected_indices = []
            for label, items in cluster_dict.items():
                items.sort(key=lambda x: x[1], reverse=True)
                selected_indices.append(items[0][0])

            selected_indices.sort()
            semantic_frames = [frames[i] for i in selected_indices]
        else:
            kept = [0]
            for i in range(1, len(embeddings)):
                sims = [np.dot(embeddings[i], embeddings[k]) for k in kept]
                if max(sims) < (1.0 - self.distance_threshold):
                    kept.append(i)
            semantic_frames = [frames[i] for i in kept]

        print(f" -> Retained {len(semantic_frames)} / {len(frames)} semantically unique keyframes ({len(semantic_frames)/len(frames)*100:.1f}%).")
        return semantic_frames


# ============================================================================
# MODULE 5: HYBRID VISUAL DEDUPLICATION (SSIM + dHash + Color Hist)
# ============================================================================
class HybridDeduplicationModule:
    """Pass 5: Removes visually redundant frames using SSIM, Color Histograms, and dHash."""

    def __init__(self, similarity_threshold: float = 0.92):
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def compute_hsv_hist(bgr_img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    @staticmethod
    def extract_features(bgr_img: np.ndarray) -> dict:
        h, w = bgr_img.shape[:2]
        thumb_h = 360
        thumb_w = int(w * (thumb_h / h))
        img = cv2.resize(bgr_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist_flat = hist.flatten()

        hash_val = imagehash.dhash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        return {"gray": gray, "hist": hist_flat, "hash": hash_val}

    def compute_hybrid_similarity_features(self, feat_a: dict, feat_b: dict) -> float:
        gray_a, gray_b = feat_a["gray"], feat_b["gray"]
        if HAS_TORCH and CUDA_AVAILABLE:
            t_a = torch.from_numpy(gray_a).to(DEVICE, dtype=torch.float32) / 255.0
            t_b = torch.from_numpy(gray_b).to(DEVICE, dtype=torch.float32) / 255.0
            luma_diff = float(torch.mean(torch.abs(t_a - t_b)).cpu().item())
            ssim_score = max(0.0, 1.0 - luma_diff)
        else:
            ssim_val, _ = ssim(gray_a, gray_b, full=True)
            ssim_score = max(0.0, float(ssim_val))

        color_score = max(0.0, float(cv2.compareHist(feat_a["hist"], feat_b["hist"], cv2.HISTCMP_CORREL)))

        dhash_dist = feat_a["hash"] - feat_b["hash"]
        dhash_score = max(0.0, 1.0 - (dhash_dist / 64.0))

        return float(np.clip(0.55 * ssim_score + 0.35 * color_score + 0.10 * dhash_score, 0.0, 1.0))

    def process(self, frames: List[dict]) -> List[dict]:
        print(f"\n[PASS 5] Hybrid Visual Deduplication Constant (SIMILARITY_THRESHOLD = {self.similarity_threshold})")
        if not frames:
            return []

        n = len(frames)
        max_workers = min(32, os.cpu_count() or 4)

        # Precompute visual features in parallel across all CPU cores
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            features_list = list(executor.map(lambda f: self.extract_features(f["frame"]), frames))

        to_delete = set()
        for i in range(n):
            if i in to_delete:
                continue
            for j in range(i + 1, n):
                if j in to_delete:
                    continue

                sim = self.compute_hybrid_similarity_features(features_list[i], features_list[j])
                if sim >= self.similarity_threshold:
                    if frames[i]["sharpness"] >= frames[j]["sharpness"]:
                        to_delete.add(j)
                    else:
                        to_delete.add(i)
                        break

        deduped = [f for idx, f in enumerate(frames) if idx not in to_delete]
        print(f" -> Retained {len(deduped)} / {n} visually unique frames ({len(deduped)/n*100:.1f}%).")
        return deduped


# ============================================================================
# PIPELINE
# ============================================================================
class ModularFrameExtractor:
    """Orchestrates the 5-Pass Frame Extraction Pipeline using explicit PassConstants."""

    def __init__(self, constants: Optional[PassConstants] = None):
        self.config = constants if constants is not None else PassConstants()

        # Initialize Modules with explicit constants
        self.pass1 = UniformSamplingModule(target_fps=self.config.pass1_target_fps)
        self.pass2 = ECRModule(
            ecr_threshold=self.config.pass2_ecr_threshold,
            suppress_camera_motion=self.config.pass2_suppress_camera_motion
        )
        self.pass3 = BlurCheckModule(min_sharpness=self.config.pass3_min_sharpness)
        self.pass4 = CLIPClusteringModule(distance_threshold=self.config.pass4_cosine_distance)
        self.pass5 = HybridDeduplicationModule(similarity_threshold=self.config.pass5_similarity_threshold)

    def run(self, video_path: str, output_dir: str) -> List[dict]:
        start_time = time.time()
        path = Path(video_path)

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_metadata = {
            "video_name": path.stem,
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": round(total_frames / fps, 2)
        }

        print("=" * 75)
        print(" MODULAR 5-PASS PRODUCTION FRAME EXTRACTOR (EXPLICIT PASS CONSTANTS)")
        print("=" * 75)
        print(f" Video Name                 : {path.name}")
        print(f" Total Frames               : {total_frames} ({video_metadata['duration_sec']}s)")
        print(f" Pass 1 TARGET_FPS          : {self.config.pass1_target_fps}")
        print(f" Pass 2 ECR_THRESHOLD       : {self.config.pass2_ecr_threshold}")
        print(f" Pass 3 MIN_SHARPNESS       : {self.config.pass3_min_sharpness}")
        print(f" Pass 4 COSINE_DISTANCE     : {self.config.pass4_cosine_distance}")
        print(f" Pass 5 SIMILARITY_THRESHOLD: {self.config.pass5_similarity_threshold}")
        print("=" * 75)

        # Execution Chain
        p1_frames = self.pass1.process(video_path, video_metadata)
        p2_frames = self.pass2.process(p1_frames)
        p3_frames = self.pass3.process(p2_frames, p1_frames)
        p4_frames = self.pass4.process(p3_frames)
        p5_frames = self.pass5.process(p4_frames)

        # Output Disk Save
        out_path = Path(output_dir) / path.stem
        out_path.mkdir(parents=True, exist_ok=True)

        final_records = []
        for idx, rec in enumerate(p5_frames):
            frame_name = f"frame_{idx + 1:04d}_{rec['frame_idx']:06d}.jpg"
            save_file = out_path / frame_name
            cv2.imwrite(str(save_file), rec["frame"], [cv2.IMWRITE_JPEG_QUALITY, 95])

            final_records.append({
                "id": idx + 1,
                "filename": frame_name,
                "filepath": str(save_file.resolve()),
                "frame_idx": rec["frame_idx"],
                "timestamp_sec": rec["timestamp_sec"],
                "sharpness": rec.get("sharpness", 0.0),
                "trigger_reason": rec.get("trigger_reason", "n/a")
            })

        # Save metadata.json
        meta_file = out_path / "metadata.json"
        summary = {
            "video_metadata": video_metadata,
            "pass_constants": asdict(self.config),
            "pipeline_stats": {
                "total_frames": total_frames,
                "pass1_uniform": len(p1_frames),
                "pass2_ecr": len(p2_frames),
                "pass3_sharp": len(p3_frames),
                "pass4_semantic_clip": len(p4_frames),
                "pass5_visual_dedupe": len(p5_frames),
                "overall_retention_ratio": f"{len(p5_frames)/total_frames*100:.2f}%",
                "execution_time_sec": round(time.time() - start_time, 2)
            },
            "keyframes": final_records
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 75)
        print(f" COMPLETE! Saved {len(final_records)} keyframes to: {out_path}")
        print(f" Overall Reduction: {total_frames} -> {len(final_records)} frames ({summary['pipeline_stats']['overall_retention_ratio']})")
        print(f" Metadata written to: {meta_file}")
        print("=" * 75 + "\n")

        return final_records


# ============================================================================
# CLI INTERFACE
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular 5-Pass Video Frame Extractor (Configurable Constants)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration JSON file (default: config.json)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file (overrides config.json)")
    parser.add_argument("--output", type=str, default=None, help="Output base directory (overrides config.json)")

    # Direct Pass Constant Arguments (overrides config.json if provided)
    parser.add_argument("--pass1-fps", type=float, default=None, help="Pass 1 Constant: Target Sampling FPS (e.g. 4.0)")
    parser.add_argument("--pass2-ecr", type=float, default=None, help="Pass 2 Constant: ECR Cut Threshold (e.g. 0.18)")
    parser.add_argument("--pass3-sharpness", type=float, default=None, help="Pass 3 Constant: Min Sharpness Variance (e.g. 30.0)")
    parser.add_argument("--pass4-distance", "--pass4-cosine", type=float, default=None, help="Pass 4 Constant: CLIP Cosine Distance Threshold (e.g. 0.10)")
    parser.add_argument("--pass5-similarity", type=float, default=None, help="Pass 5 Constant: Visual Similarity Threshold (e.g. 0.95)")

    args = parser.parse_args()

    # 1. Load constants from config.json first
    constants = PassConstants.from_json(args.config)

    # 2. Override with explicit CLI flags if user provided them
    if args.pass1_fps is not None:
        constants.pass1_target_fps = args.pass1_fps
    if args.pass2_ecr is not None:
        constants.pass2_ecr_threshold = args.pass2_ecr
    if args.pass3_sharpness is not None:
        constants.pass3_min_sharpness = args.pass3_sharpness
    if args.pass4_distance is not None:
        constants.pass4_cosine_distance = args.pass4_distance
    if args.pass5_similarity is not None:
        constants.pass5_similarity_threshold = args.pass5_similarity

    # 3. Resolve video path and output dir from config.json if not provided in CLI
    video_path = args.video
    output_dir = args.output

    if Path(args.config).is_file():
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                if not video_path:
                    video_path = cfg_data.get("default_video_path")
                if not output_dir:
                    output_dir = cfg_data.get("default_output_dir", "frames")
        except Exception:
            pass

    if not video_path:
        video_path = r"D:\.HandOnRAG\frameExtract\video\VID003.mp4"
    if not output_dir:
        output_dir = "frames"

    extractor = ModularFrameExtractor(constants=constants)
    extractor.run(video_path, output_dir)
