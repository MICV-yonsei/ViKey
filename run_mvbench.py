"""
ViKey inference on MVBench using cached frame images.

This script is Stage 3 of the ViKey pipeline:
    Stage 1  add_VP.py / add_VP_outline.py  --  overlay visual prompts on frames
    Stage 2  keyword_extractor_mvbench.py   --  extract question keywords
    Stage 3  this file                      --  run LLaVA-Video on VP-annotated frames

Expected cache layout
---------------------
    <cache-dir>/<video_id_without_ext>/frameXXX_*.png

Example
-------
    python run_mvbench.py \\
        --cache-dir /path/to/VP_frames \\
        --labels-path /path/to/mvbench_labels.jsonl \\
        --save-jsonl /path/to/output_dir \\
        --score-thres 0.2

Notes
-----
- Auto-resume: if pred.jsonl already exists in --save-jsonl, processing
  continues from the last completed sample.
- The system prompt is auto-selected from the --cache-dir name; override
  with --custom-system-prompt or disable with --disable-auto-system-prompt.
- Run ``python run_mvbench.py --help`` for the full flag list.
"""

import argparse
import warnings
import copy
import json
import os
import sys
import shlex
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image

# LLaVA-NeXT Video
# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# Import CLIP - OpenAI's CLIP model
# pip install git+https://github.com/openai/CLIP.git
try:
    import clip as openai_clip
except ImportError:
    print("[WARNING] OpenAI CLIP not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
    import clip as openai_clip

warnings.filterwarnings("ignore")


def get_clip_model_name(clip_model_name: str) -> str:
    """Return a short identifier for the CLIP backbone (e.g. ``ViT-L/14`` -> ``CLIP-L``)."""
    if "ViT-L" in clip_model_name or "vit-l" in clip_model_name.lower():
        return "CLIP-L"
    elif "ViT-B" in clip_model_name or "vit-b" in clip_model_name.lower():
        return "CLIP-B"
    else:
        return "CLIP-L"


def get_position_from_path(cache_dir_path: str) -> str:
    """Infer the VP placement corner from the cache-dir name.

    Looks for ``_posTL`` / ``_posTR`` / ``_posBL`` / ``_posBR`` (case-insensitive)
    in the path. Returns one of ``top-left``, ``top-right``, ``bottom-left``,
    ``bottom-right``; defaults to ``top-left`` if no marker is found.
    """
    cache_dir_path_lower = cache_dir_path.lower()
    
    if "_postl" in cache_dir_path_lower or "_pos_tl" in cache_dir_path_lower or "postl" in cache_dir_path_lower:
        return "top-left"
    elif "_postr" in cache_dir_path_lower or "_pos_tr" in cache_dir_path_lower or "postr" in cache_dir_path_lower:
        return "top-right"
    elif "_posbl" in cache_dir_path_lower or "_pos_bl" in cache_dir_path_lower or "posbl" in cache_dir_path_lower:
        return "bottom-left"
    elif "_posbr" in cache_dir_path_lower or "_pos_br" in cache_dir_path_lower or "posbr" in cache_dir_path_lower:
        return "bottom-right"
    else:
        return "top-left"


def extract_frame_number_from_filename(filename: str) -> int:
    """
    Extract frame number from filename.
    Example: frame002_00:01.png -> 2
    """
    match = re.match(r'frame(\d+)_.*', filename)
    if match:
        return int(match.group(1))
    return -1


def compute_clip_similarity(clip_model, clip_preprocess, frame_images: List[np.ndarray], 
                            frame_filenames: List[str], text_queries: List[str], device: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Find the most similar frame number for each text query using CLIP.
    
    Args:
        clip_model: CLIP model
        clip_preprocess: CLIP preprocessing function
        frame_images: List of frame images (numpy array)
        frame_filenames: List of frame filenames
        text_queries: List of text queries to search
        device: Device (cuda/cpu)
    
    Returns:
        (query_to_frame, query_to_score): Frame number and similarity score for each text query
    """
    if not text_queries or len(text_queries) == 0:
        return {}, {}
    
    preprocessed_frames = []
    for frame in frame_images:
        pil_image = Image.fromarray(frame)
        preprocessed = clip_preprocess(pil_image)
        preprocessed_frames.append(preprocessed)
    
    frame_batch = torch.stack(preprocessed_frames).to(device)
    text_tokens = openai_clip.tokenize(text_queries).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(frame_batch)
        text_features = clip_model.encode_text(text_tokens)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (text_features @ image_features.T)
    
    query_to_frame = {}
    query_to_score = {}
    for i, query in enumerate(text_queries):
        best_frame_idx = similarity[i].argmax().item()
        best_score = similarity[i, best_frame_idx].item()
        frame_number = extract_frame_number_from_filename(frame_filenames[best_frame_idx])
        query_to_frame[query] = frame_number
        query_to_score[query] = best_score
    
    return query_to_frame, query_to_score


def augment_question_with_frame_numbers(question: str, query_to_frame: Dict[str, int]) -> str:
    """
    Find parsed_query phrases in question text and add frame numbers.
    
    Args:
        question: Original question text
        query_to_frame: Frame number mapping for each query
    
    Returns:
        Question text with frame numbers added
    """
    if not query_to_frame:
        return question
    
    modified_question = question
    sorted_queries = sorted(query_to_frame.keys(), key=len, reverse=True)
    
    for query in sorted_queries:
        frame_num = query_to_frame[query]
        if frame_num >= 0:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            if not re.search(re.escape(query) + r'\s*\(frame\s+\d+\)', modified_question, re.IGNORECASE):
                modified_question = pattern.sub(lambda m: f"{m.group(0)} (frame {frame_num})", modified_question, count=1)
    
    return modified_question


def get_auto_system_prompt(cache_dir: str) -> Optional[str]:
    """
    Automatically select system prompt based on cache-dir name.
    
    Supported patterns:
    1. auto*_1, auto*_2, auto*_3: Number reference prompts
    2. auto*_4: Timestamp reference prompts
    3. _thumbnail1: 3-frame thumbnail prompts
    4. _thumbnail2: 5-frame thumbnail prompts
    5. optical_flow_prompts_v1: Colored region optical flow prompts
    6. optical_flow_prompts_v2: Border line optical flow prompts
    
    Args:
        cache_dir: Cache directory path
    
    Returns:
        Selected system prompt (default: same as auto1, number reference prompt)
    """
    cache_dir_path = cache_dir.rstrip("/")
    position = get_position_from_path(cache_dir_path)

    if "auto4" in cache_dir_path:
        return f"Focus on the temporal relationships by referring to the time stamp written in the {position} corner of each frame."
    
    if "auto1" in cache_dir_path or "auto2" in cache_dir_path or "auto3" in cache_dir_path:
        return f"Focus on the temporal relationships by referring to the number written in the {position} corner of each frame."

    return f"Focus on the temporal relationships by referring to the number written in the {position} corner of each frame."


def build_prompt(question: str, options: List[str]) -> str:
    """
    Build MVBench format prompt.
    
    Structure: "Question:{question}\nOption:\n(A) ...\n(B) ...\n... + post_prompt"
    - post_prompt: "Only give the best option.\n"
    
    Args:
        question: User question text
        options: Choice list (order preserved)
    Returns:
        Complete prompt string
    """
    question_part = f"Question:{question}\nOption:\n"
    
    labeled_options: List[str] = []
    for idx, opt in enumerate(options):
        label = chr(ord('A') + idx)
        labeled_options.append(f"({label}) {opt}")
    options_text = "\n".join(labeled_options)
    
    post_prompt = "Only give the best option.\n"
    
    full_prompt = question_part + options_text + "\n" + post_prompt
    return full_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-Video inference (JSONL loop mode only)")

    # Model
    parser.add_argument("--model-path", type=str,
                        default="lmms-lab/LLaVA-Video-7B-Qwen2",
                        help="Pretrained model path")
    parser.add_argument("--model-name", type=str, default="llava_qwen",
                        help="Model name key used inside LLaVA")
    parser.add_argument("--conv-template", type=str, default="qwen_1_5",
                        help="Conversation template")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda | cpu")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="accelerate device map, e.g., auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16",
                        help="float16 | bfloat16 | float32")

    # Data (cache directory - REQUIRED)
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Cache directory containing frame images in subdirectories")

    # Data (loop over JSONL)
    parser.add_argument("--labels-path", type=str, default=None,
                        help="Path to labels.jsonl (e.g., /nfs/data/yeonkyung/datasets/Video_MME/labels.jsonl)")
    parser.add_argument("--save-jsonl", type=str, default=None,
                        help="Where to save predictions as JSONL")
    
    # Visualization
    parser.add_argument("--save-visualization", action="store_true",
                        help="Save frame visualization as PNG (default: False)")
    
    # Prompt control
    parser.add_argument("--no-pre-prompt", action="store_true",
                        help="Do not add frame order instruction to prompt (default: False, adds frame prompt)")
    
    # Resume control
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing pred.jsonl and start from scratch (default: False, resume from existing)")

    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for generation (0.0 for deterministic)")
    parser.add_argument("--do-sample", action="store_true",
                        help="Enable sampling during generation")
    
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    # [system prompt]
    # Custom system prompt
    parser.add_argument("--custom-system-prompt", type=str, default=None,
                        help="Additional text to append to the system prompt")
    
    # Auto system prompt control
    parser.add_argument("--disable-auto-system-prompt", action="store_true",
                        help="Disable automatic system prompt selection based on cache-dir name (default: False, auto-selection enabled)")
    
    # CLIP score threshold
    parser.add_argument("--score-thres", type=float, default=None,
                        help="Minimum CLIP score threshold for mapping. If the highest score is below this threshold, mapping is skipped. (default: None, always perform mapping)")

    return parser.parse_args()


def ensure_parent_dir(save_path: str):
    """Create parent directory for --save-jsonl if it doesn't exist"""
    if save_path is None:
        return
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def save_individual_frames(frames_np: np.ndarray, frames_dir: str, frame_times: List[float] = None):
    """
    Save each frame as an individual image file.
    
    Args:
        frames_np: (F, H, W, C) numpy array
        frames_dir: Directory to save frames
        frame_times: List of frame times in seconds
    """
    os.makedirs(frames_dir, exist_ok=True)
    
    for idx, frame in enumerate(frames_np):
        if frame_times and idx < len(frame_times):
            time_sec = frame_times[idx]
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            filename = f"frame{idx+1:03d}_{time_str}.png"
        else:
            filename = f"frame{idx+1:03d}.png"
        
        frame_path = os.path.join(frames_dir, filename)
        img = Image.fromarray(frame)
        img.save(frame_path)


def load_frames_from_directory(frames_dir: str, expected_count: int = None) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Load saved frame images from directory (sorted alphabetically).
    
    Args:
        frames_dir: Directory containing frame images
        expected_count: Expected number of frames (for validation)
    
    Returns:
        (frames_np, frame_filenames): (F, H, W, C) numpy array and filename list, None on failure
    """
    if not os.path.exists(frames_dir):
        return None
    
    all_files = os.listdir(frames_dir)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    frame_files = sorted([f for f in all_files 
                         if any(f.lower().endswith(ext) for ext in image_extensions)])
    
    if len(frame_files) == 0:
        return None
    
    if expected_count is not None and len(frame_files) != expected_count:
        return None
    
    frames = []
    valid_filenames = []
    target_size = (384, 384)
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        try:
            img = Image.open(frame_path).convert('RGB')
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            frame_np = np.array(img_resized)
            frames.append(frame_np)
            valid_filenames.append(frame_file)
        except Exception as e:
            print(f"[WARN] Failed to load image {frame_path}: {e}")
            continue
    
    if len(frames) == 0:
        return None
    
    return np.array(frames, dtype=np.uint8), valid_filenames


def save_frames_visualization(frames_np: np.ndarray, save_path: str, video_id: str = "", 
                             frame_times: List[float] = None, query_to_frame: Dict[str, int] = None,
                             query_to_score: Dict[str, float] = None):
    """
    Visualize and save selected frames.
    
    Args:
        frames_np: (F, H, W, C) numpy array
        save_path: Path to save image
        video_id: Video ID (for title)
        frame_times: List of frame times in seconds
        query_to_frame: Frame number mapping for each query (optional)
        query_to_score: CLIP similarity score mapping for each query (optional)
    """
    num_frames = len(frames_np)
    
    frame_to_queries = {}
    all_queries = []
    if query_to_frame:
        for query, frame_num in query_to_frame.items():
            all_queries.append(query)
            if frame_num >= 0 and frame_num < num_frames:
                if frame_num not in frame_to_queries:
                    frame_to_queries[frame_num] = []
                score = query_to_score.get(query, 0.0) if query_to_score else 0.0
                frame_to_queries[frame_num].append((query, score))
    
    cols = min(8, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig_width = cols * 2.5
    fig_height = rows * 2.5
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    title = f'Selected Frames: {video_id}' if video_id else 'Selected Frames'
    if frame_to_queries:
        title += f' (🔴 = CLIP matched)'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    if all_queries:
        keywords_text = "Keywords: " + " | ".join(all_queries)
        fig.text(0.5, 0.96, keywords_text, ha='center', va='top', 
                fontsize=10, wrap=True, color='darkblue', weight='bold')
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx in range(len(axes)):
        ax = axes[idx]
        if idx < num_frames:
            ax.imshow(frames_np[idx])
            
            is_matched = idx in frame_to_queries
            title_parts = []
            
            if frame_times and idx < len(frame_times):
                time_sec = frame_times[idx]
                minutes = int(time_sec // 60)
                seconds = int(time_sec % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                title_parts.append(f'Frame {idx+1} ({time_str})')
            else:
                title_parts.append(f'Frame {idx+1}')
            
            if is_matched:
                queries_with_scores = frame_to_queries[idx]
                query_texts = []
                for q, score in queries_with_scores:
                    if len(q) > 25:
                        query_text = q[:22] + '...'
                    else:
                        query_text = q
                    query_texts.append(f"{query_text} ({score:.3f})")
                title_parts.append('🔴 ' + '\n'.join(query_texts))
            
            ax.set_title('\n'.join(title_parts), fontsize=9, 
                        color='red' if is_matched else 'black',
                        weight='bold' if is_matched else 'normal')
            
            if is_matched:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout(pad=1.5, w_pad=2.0, h_pad=2.0)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert timestamp in "HH:MM:SS" or "MM:SS" format to seconds.
    Example: "00:01:06" -> 66.0, "01:12" -> 72.0
    """
    parts = timestamp.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    else:
        return float(parts[0])


def load_video_at_timestamps(video_path: str, frame_times: List[str]):
    """
    Extract frames only at specified timestamps in frame_times.
    
    Args:
        video_path: Video file path
        frame_times: List of timestamps (e.g., ["00:01:06", "00:01:12"])
    
    Return:
      frames_np: (F, H, W, C) uint8
      frame_time_str: "t0s,t1s,..."
      video_time: seconds (float)
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    fps = vr.get_avg_fps()
    video_time = total / fps if fps > 0 else 0.0
    
    if not frame_times or len(frame_times) == 0:
        idx = [0]
        times = [0.0]
    else:
        target_seconds = [parse_timestamp_to_seconds(t) for t in frame_times]
        
        idx = []
        times = []
        for sec in target_seconds:
            frame_idx = int(sec * fps)
            frame_idx = min(max(0, frame_idx), total - 1)
            idx.append(frame_idx)
            times.append(sec)
    
    frame_time_str = ",".join([f"{t:.2f}s" for t in times])
    frames_np = vr.get_batch(idx).asnumpy().astype(np.uint8)
    return frames_np, frame_time_str, video_time


def load_video_uniform(video_path: str, max_frames: int, force_sample: bool = False):
    """
    Uniform sampling (kept for backward compatibility).
    Return:
      frames_np: (F, H, W, C) uint8
      frame_time_str: "t0s,t1s,..."
      video_time: seconds (float)
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    fps = vr.get_avg_fps()
    video_time = total / fps if fps > 0 else 0.0

    idx = list(range(total))
    if len(idx) > max_frames or force_sample:
        idx = np.linspace(0, total - 1, max_frames, dtype=int).tolist()

    times = [i / fps if fps > 0 else 0.0 for i in idx]
    frame_time_str = ",".join([f"{t:.2f}s" for t in times])

    frames_np = vr.get_batch(idx).asnumpy().astype(np.uint8)
    return frames_np, frame_time_str, video_time


def build_llava_inputs(
    image_processor, tokenizer, conv_template: str, video: List[torch.Tensor],
    video_time: float, frame_time: str, user_question_text: str, device: str,
    num_frames: int = None, frames_from_cache: bool = False, add_frame_prompt: bool = True,
    custom_system_prompt: str = None, cache_dir: str = None, disable_auto_system_prompt: bool = False,
):
    """
    Build LLaVA input structure.
    
    Structure:
    1. System prompt: Model role and behavior instructions
    2. User prompt: pre_prompt + question + options + post_prompt
    """
    if num_frames is None:
        num_frames = len(video[0]) if video and len(video) > 0 else 0
    
    conv = copy.deepcopy(conv_templates[conv_template])
    
    final_system_prompt = None
    
    auto_system_prompt = None
    if cache_dir and not disable_auto_system_prompt:
        auto_system_prompt = get_auto_system_prompt(cache_dir)
    
    if auto_system_prompt:
        final_system_prompt = auto_system_prompt
        print(f"[INFO] Auto-selected system prompt: {auto_system_prompt}")
    elif custom_system_prompt:
        final_system_prompt = custom_system_prompt
        print(f"[INFO] Using custom system prompt: {custom_system_prompt}")
    elif disable_auto_system_prompt:
        print(f"[INFO] Auto system prompt selection disabled, no system prompt added")
    
    if final_system_prompt:
        if conv_template.startswith("qwen"):
            conv.system = conv.system.replace(
                "You are a helpful assistant.",
                f"You are a helpful assistant. {final_system_prompt}"
            )
        elif conv_template.startswith("llava_llama"):
            conv.system += f" {final_system_prompt}"
        else:
            conv.system += f" {final_system_prompt}"
    
    pre_prompt = ""
    if add_frame_prompt:
        if num_frames > 0:
            pre_prompt = f"Please examine frames from Frame #01 to Frame #{num_frames:02d} in sequential order and answer the following question."
        else:
            pre_prompt = "Please examine the frames in sequential order and answer the following question."
    
    post_prompt = ""
    
    user_prompt_parts = []
    if pre_prompt:
        user_prompt_parts.append(pre_prompt)
    user_prompt_parts.append(user_question_text)
    if post_prompt:
        user_prompt_parts.append(post_prompt)
    
    user_msg = DEFAULT_IMAGE_TOKEN + "\n".join(user_prompt_parts)
    
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)

    return input_ids, attention_mask




@torch.no_grad()
def run_inference_for_video(
    model, tokenizer, image_processor, args, user_question_text: str,
    frames_cache_dir: str, clip_model=None, clip_preprocess=None, parsed_query: List[str] = None,
    cached_similarity: Optional[Dict[str, Any]] = None, clip_model_name: str = "CLIP-L"
) -> Dict[str, Any]:
    """
    Load frame images from cache directory and perform inference.
    
    Args:
        frames_cache_dir: Directory path containing frame images (required)
        clip_model: CLIP model (optional)
        clip_preprocess: CLIP preprocessing function (optional)
        parsed_query: List of keywords extracted from question (optional)
        cached_similarity: Cached similarity information (optional)
        clip_model_name: CLIP model name (e.g., "CLIP-L")
    """
    result = load_frames_from_directory(frames_cache_dir)
    
    if result is None:
        raise ValueError(f"Cannot load frames from cache directory: {frames_cache_dir}")
    
    frames_np, frame_filenames = result
    num_frames = len(frames_np)
    print(f"  [INFO] Loaded {num_frames} frames from cache: {frames_cache_dir}")
    
    modified_question_text = user_question_text
    query_to_frame = {}
    query_to_score = {}
    
    similarity_key = f"{clip_model_name}_sim"
    if cached_similarity and similarity_key in cached_similarity:
        cached_sim = cached_similarity[similarity_key]
        if isinstance(cached_sim, dict):
            query_to_frame = cached_sim.get("query_to_frame", {})
            query_to_score = cached_sim.get("query_to_score", {})
            print(f"  [INFO] Using cached similarity from {similarity_key}")
            print(f"  [INFO] Query to frame mapping: {query_to_frame}")
            print(f"  [INFO] Query to score mapping: {query_to_score}")
        else:
            print(f"  [WARN] Cached similarity format not recognized, recomputing...")
            cached_similarity = None
    
    if not query_to_frame and clip_model is not None and clip_preprocess is not None and parsed_query and len(parsed_query) > 0:
        print(f"  [INFO] Computing CLIP similarity for {len(parsed_query)} queries...")
        query_to_frame, query_to_score = compute_clip_similarity(
            clip_model, clip_preprocess, frames_np, frame_filenames, parsed_query, args.device
        )
        print(f"  [INFO] Query to frame mapping: {query_to_frame}")
        print(f"  [INFO] Query to score mapping: {query_to_score}")
        
        if args.score_thres is not None:
            max_score = max(query_to_score.values()) if query_to_score else 0.0
            if max_score < args.score_thres:
                print(f"  [INFO] Max CLIP score ({max_score:.4f}) is below threshold ({args.score_thres}), skipping mapping")
                query_to_frame = {}
                query_to_score = {}
            else:
                print(f"  [INFO] Max CLIP score ({max_score:.4f}) is above threshold ({args.score_thres}), applying mapping")
                modified_question_text = augment_question_with_frame_numbers(user_question_text, query_to_frame)
                print(f"  [INFO] Modified question: {modified_question_text[:200]}...")
        else:
            modified_question_text = augment_question_with_frame_numbers(user_question_text, query_to_frame)
            print(f"  [INFO] Modified question: {modified_question_text[:200]}...")
    elif query_to_frame:
        if args.score_thres is not None:
            max_score = max(query_to_score.values()) if query_to_score else 0.0
            if max_score < args.score_thres:
                print(f"  [INFO] Max CLIP score ({max_score:.4f}) is below threshold ({args.score_thres}), skipping mapping")
                query_to_frame = {}
                query_to_score = {}
            else:
                print(f"  [INFO] Max CLIP score ({max_score:.4f}) is above threshold ({args.score_thres}), applying mapping")
                modified_question_text = augment_question_with_frame_numbers(user_question_text, query_to_frame)
                print(f"  [INFO] Modified question: {modified_question_text[:200]}...")
        else:
            modified_question_text = augment_question_with_frame_numbers(user_question_text, query_to_frame)
            print(f"  [INFO] Modified question: {modified_question_text[:200]}...")
    
    video_time = float(num_frames)
    frame_time = ",".join([f"{i}s" for i in range(num_frames)])
    
    torch_dtype = {
        "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32
    }.get(args.torch_dtype.lower(), torch.bfloat16)

    video_tensor = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(args.device, dtype=torch_dtype)
    video = [video_tensor]
    
    add_frame_prompt = not args.no_pre_prompt
    input_ids, attention_mask = build_llava_inputs(
        image_processor, tokenizer, args.conv_template, video, video_time, frame_time,
        modified_question_text, args.device, num_frames=num_frames, frames_from_cache=True,
        add_frame_prompt=add_frame_prompt, custom_system_prompt=args.custom_system_prompt,
        cache_dir=frames_cache_dir, disable_auto_system_prompt=args.disable_auto_system_prompt
    )

    output_ids = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        images=video,
        modalities=["video"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        use_cache=True,
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    del video_tensor, video, input_ids, output_ids
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    frame_times = list(range(num_frames))
    
    return {
        "prediction": output_text,
        "video_time": video_time,
        "frame_time": frame_time,
        "num_frames_used": num_frames,
        "frames_np": frames_np,
        "frame_times": frame_times,
        "frames_from_cache": True,
        "query_to_frame": query_to_frame,
        "query_to_score": query_to_score
    }




def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_mvbench_question_and_candidates(sample: Dict[str, Any]) -> (str, List[str]):
    """Extract question and candidates from MVBench format"""
    question = sample.get("question", "")
    candidates = sample.get("candidates", [])
    
    if isinstance(candidates, str):
        try:
            candidates = json.loads(candidates)
        except Exception:
            candidates = [c.strip() for c in candidates.split(",") if c.strip()]
    
    return str(question), list(candidates)


def get_processed_question_ids(pred_jsonl_path: str) -> set:
    """Extract already processed question_ids from existing pred.jsonl file"""
    processed_ids = set()
    if not os.path.exists(pred_jsonl_path):
        return processed_ids
    
    try:
        with open(pred_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question_id = None
                    for k in ["question_id", "questionId", "QuestionID", "qid", "id"]:
                        if k in data and data[k]:
                            question_id = str(data[k])
                            break
                    if question_id:
                        processed_ids.add(question_id)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[WARN] Error reading existing pred.jsonl: {e}")
    
    return processed_ids


def resolve_video_id(sample: Dict[str, Any]) -> Optional[str]:
    """
    MVBench format: Remove .mp4 extension from video_path field.
    Example: "ZS9XR_1.5_17.1.mp4" -> "ZS9XR_1.5_17.1"
    """
    if "video_path" in sample and sample["video_path"]:
        video_path = str(sample["video_path"])
        if video_path.endswith(".mp4"):
            return video_path[:-4]
        return video_path
    
    for k in ["videoID", "video_id", "videoId", "video", "id"]:
        if k in sample and sample[k]:
            video_val = str(sample[k])
            if video_val.endswith(".mp4"):
                return video_val[:-4]
            return video_val
    return None


def resolve_question_id(sample: Dict[str, Any]) -> Optional[str]:
    """
    Resolve question ID from sample with priority order.
    """
    for k in ["question_id", "questionId", "QuestionID", "qid", "id"]:
        if k in sample and sample[k]:
            return str(sample[k])
    return None


def resolve_question_and_options(sample: Dict[str, Any]) -> (str, List[str]):
    """Extract question and options from Video MME format"""
    question = sample.get("question") or sample.get("Question") or ""
    options = sample.get("options") or sample.get("Options") or []
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            options = [o.strip() for o in options.split(",") if o.strip()]
    return str(question), list(options)


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to: {args.seed}")

    # dtype
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(args.torch_dtype.lower(), torch.bfloat16)

    # Load model once
    print("Loading model...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        torch_dtype=args.torch_dtype,   # LLaVA loader handles string dtype
        attn_implementation="sdpa",
        device_map=args.device_map,
    )
    model.eval()
    print(f"Model: {model.__class__.__name__}, device: {next(model.parameters()).device}")
    
    print("Loading CLIP model...")
    clip_model_path = "ViT-L/14"
    clip_model, clip_preprocess = openai_clip.load(clip_model_path, device=args.device)
    clip_model.eval()
    clip_model_name = get_clip_model_name(clip_model_path)
    print(f"CLIP model loaded successfully: {clip_model_name} ({clip_model_path})")

    if args.labels_path:
        cache_root = args.cache_dir.rstrip("/")

        save_dir = None
        out_f = None
        pred_jsonl_path = None
        processed_question_ids = set()
        
        if args.save_jsonl:
            if args.save_jsonl.endswith('.jsonl'):
                save_dir = args.save_jsonl[:-6]
            else:
                save_dir = args.save_jsonl
            
            os.makedirs(save_dir, exist_ok=True)
            
            pred_jsonl_path = os.path.join(save_dir, "pred.jsonl")
            try:
                executable = sys.executable or "python"
                script_path = os.path.abspath(__file__)
                command_list = [executable, script_path] + sys.argv[1:]
                full_command = shlex.join(command_list)

                run_sh_path = os.path.join(save_dir, "run_command.sh")
                with open(run_sh_path, "w", encoding="utf-8") as f_sh:
                    f_sh.write("#!/usr/bin/env bash\n")
                    f_sh.write("set -euo pipefail\n\n")
                    f_sh.write(f"{full_command}\n")
                try:
                    os.chmod(run_sh_path, 0o755)
                except Exception:
                    pass

                args_json_path = os.path.join(save_dir, "args_used.json")
                args_payload = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "script": script_path,
                    "executable": executable,
                    "args": vars(args),
                    "command": full_command,
                }
                with open(args_json_path, "w", encoding="utf-8") as f_args:
                    json.dump(args_payload, f_args, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to save run command and args: {e}")
            
            if args.overwrite:
                print(f"Overwrite mode: Starting from scratch")
                if os.path.exists(pred_jsonl_path):
                    print(f"Removing existing file: {pred_jsonl_path}")
                    os.remove(pred_jsonl_path)
                processed_question_ids = set()
                out_f = open(pred_jsonl_path, "w", encoding="utf-8")
            elif os.path.exists(pred_jsonl_path):
                processed_question_ids = get_processed_question_ids(pred_jsonl_path)
                print(f"Resume mode: Found {len(processed_question_ids)} already processed items")
                out_f = open(pred_jsonl_path, "a", encoding="utf-8")
            else:
                out_f = open(pred_jsonl_path, "w", encoding="utf-8")
            
            print(f"Results will be saved to: {pred_jsonl_path}")
            print(f"Cache directory: {cache_root}")
        
        processed, skipped, resumed = 0, 0, 0
        missing_frames_list = []

        samples = list(read_jsonl(args.labels_path))

        for sample in tqdm(samples, desc="Processing JSONL", ncols=100):
            question_id = resolve_question_id(sample)
            if not question_id:
                skipped += 1
                print(f"[WARN] No question_id found in sample")
                continue
            
            if question_id in processed_question_ids:
                resumed += 1
                if resumed <= 5:
                    print(f"[RESUME] Skipping already processed question_id: {question_id}")
                continue
            
            video_id = resolve_video_id(sample)
            
            if not video_id:
                print(f"[DEBUG] video_id extraction failed. Sample keys: {list(sample.keys())}")
                print(f"[DEBUG] video_path value: {sample.get('video_path', 'N/A')}")
                print(f"[DEBUG] video value: {sample.get('video', 'N/A')}")
                skipped += 1
                print(f"[WARN] No video_path found in sample")
                continue
            
            frames_cache_dir = None
            used_id = None
            
            if video_id:
                candidate_dir = os.path.join(cache_root, video_id)
                if os.path.exists(candidate_dir):
                    frames_cache_dir = candidate_dir
                    used_id = video_id
                    print(f"[DEBUG] Found directory using video_path: {candidate_dir}")
                else:
                    print(f"[DEBUG] Tried video_path path but not found: {candidate_dir} (cache_root={cache_root}, video_id={video_id})")
            
            if not frames_cache_dir and "video" in sample and sample["video"]:
                video_base = str(sample["video"])
                candidate_dir = os.path.join(cache_root, video_base)
                if os.path.exists(candidate_dir):
                    frames_cache_dir = candidate_dir
                    used_id = video_base
                    print(f"[DEBUG] Found directory using video field: {candidate_dir}")
                else:
                    print(f"[DEBUG] Tried video field path but not found: {candidate_dir}")
            if not frames_cache_dir:
                skipped += 1
                missing_frames_list.append({
                    "question_id": question_id,
                    "video_path": sample.get("video_path", video_id),
                    "video": sample.get("video", "N/A"),
                    "video_id": video_id,
                    "reason": "directory_not_found",
                    "tried_paths": {
                        "video_path_path": os.path.join(cache_root, video_id) if video_id else None,
                        "video_field_path": os.path.join(cache_root, sample.get("video", "")) if sample.get("video") else None,
                        "cache_root": cache_root
                    }
                })
                print(f"[WARN] Cache directory not found for question_id={question_id}, video_id={video_id}")
                print(f"[WARN] Tried paths: video_path={os.path.join(cache_root, video_id) if video_id else None}, video={os.path.join(cache_root, sample.get('video', '')) if sample.get('video') else None}")
                continue

            question, candidates = resolve_mvbench_question_and_candidates(sample)
            prompt = build_prompt(question, candidates)
            
            parsed_query = []
            try:
                raw_query = sample.get("parsed_query", [])
                if raw_query is None or raw_query == "":
                    parsed_query = []
                elif isinstance(raw_query, list):
                    parsed_query = raw_query
                elif isinstance(raw_query, str):
                    try:
                        parsed_query = json.loads(raw_query)
                        if not isinstance(parsed_query, list):
                            parsed_query = []
                    except Exception:
                        parsed_query = []
                else:
                    parsed_query = []
            except Exception:
                parsed_query = []
            
            try:
                cached_similarity = None
                similarity_key = f"{clip_model_name}_sim"
                if similarity_key in sample:
                    cached_similarity = {similarity_key: sample[similarity_key]}
                
                out = run_inference_for_video(
                    model, tokenizer, image_processor, args, prompt,
                    frames_cache_dir=frames_cache_dir,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    parsed_query=parsed_query,
                    cached_similarity=cached_similarity,
                    clip_model_name=clip_model_name
                )
                result = {
                    **sample,
                    "question_id": question_id,
                    "video_id": video_id,
                    "used_id": used_id,
                    "pred": out["prediction"],
                    "num_frames_used": out["num_frames_used"],
                }
                if "query_to_frame" in out:
                    result["query_to_frame"] = out["query_to_frame"]
                if "query_to_score" in out:
                    result["query_to_score"] = out["query_to_score"]
                
                if "query_to_frame" in out and "query_to_score" in out:
                    if out["query_to_frame"] or out["query_to_score"]:
                        result[similarity_key] = {
                            "query_to_frame": out["query_to_frame"],
                            "query_to_score": out["query_to_score"]
                        }
                
                if out_f:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    os.fsync(out_f.fileno())
                else:
                    print(json.dumps(result, ensure_ascii=False))
                
                if args.save_visualization and save_dir and "frames_np" in out:
                    frame_times = out.get("frame_times", [])
                    query_to_frame = out.get("query_to_frame", {})
                    query_to_score = out.get("query_to_score", {})
                    viz_frames_dir = os.path.join(save_dir, "viz_frames")
                    os.makedirs(viz_frames_dir, exist_ok=True)
                    display_id = used_id if used_id else video_id
                    safe_display_id = display_id.replace("/", "_").replace("\\", "_")
                    frames_image_path = os.path.join(viz_frames_dir, f"{safe_display_id}_frames.png")
                    save_frames_visualization(out["frames_np"], frames_image_path, display_id, 
                                            frame_times, query_to_frame, query_to_score)
                
                processed += 1
            except ValueError as e:
                skipped += 1
                display_id = used_id if used_id else video_id
                missing_frames_list.append({
                    "question_id": question_id,
                    "video_path": sample.get("video_path", video_id),
                    "video": sample.get("video", "N/A"),
                    "video_id": video_id,
                    "used_id": used_id,
                    "reason": "no_frames_in_directory",
                    "error": str(e)
                })
                print(f"[WARN] No frames in {display_id} (question_id: {question_id}): {e}")
            except Exception as e:
                skipped += 1
                display_id = used_id if used_id else video_id
                missing_frames_list.append({
                    "question_id": question_id,
                    "video_path": sample.get("video_path", video_id),
                    "video": sample.get("video", "N/A"),
                    "video_id": video_id,
                    "used_id": used_id,
                    "reason": "inference_failed",
                    "error": str(e)
                })
                print(f"[WARN] Failed on {display_id} (question_id: {question_id}): {e}")

        if out_f:
            out_f.close()

        if missing_frames_list:
            output_dir = save_dir if save_dir else "."
            
            missing_json_path = os.path.join(output_dir, "missing_frames.json")
            with open(missing_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_missing": len(missing_frames_list),
                    "missing_items": missing_frames_list
                }, f, ensure_ascii=False, indent=2)
            print(f"\nSaved missing frames info to: {missing_json_path}")
            
            missing_txt_path = os.path.join(output_dir, "missing_frames.txt")
            with open(missing_txt_path, "w", encoding="utf-8") as f:
                f.write(f"Total missing: {len(missing_frames_list)}\n")
                f.write("=" * 50 + "\n\n")
                for item in missing_frames_list:
                    f.write(f"Video Path: {item.get('video_path', 'N/A')}\n")
                    f.write(f"Video ID: {item['video_id']}\n")
                    f.write(f"Reason: {item['reason']}\n")
                    if 'error' in item:
                        f.write(f"Error: {item['error']}\n")
                    f.write("-" * 50 + "\n")
            print(f"Saved missing frames list to: {missing_txt_path}")

        print(f"\nDone. processed={processed}, skipped={skipped}, resumed={resumed}, missing_frames={len(missing_frames_list)}")
        return

    print("[ERROR] This script only supports JSONL loop mode. Please provide --labels-path.")
    return


if __name__ == "__main__":
    main()
