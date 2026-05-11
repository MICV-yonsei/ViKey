"""
ViKey inference on TempCompass using cached frame images.

This script is Stage 3 of the ViKey pipeline:
    Stage 1  add_VP.py / add_VP_outline.py     --  overlay visual prompts on frames
    Stage 2  keyword_extractor_tempcompass.py  --  extract question keywords
    Stage 3  this file                         --  run LLaVA-Video on VP-annotated frames

Expected cache layout
---------------------
    <cache-dir>/<video_id_without_ext>/frameXXX_*.png

Example
-------
    python run_tempcompass.py \\
        --cache-dir /path/to/VP_frames \\
        --labels-path /path/to/tempcompass_labels.jsonl \\
        --save-jsonl /path/to/output_dir \\
        --score-thres 0.2

Notes
-----
- Auto-resume: if pred.jsonl already exists in --save-jsonl, processing
  continues from the last completed sample.
- The system prompt is auto-selected from the --cache-dir name; override
  with --custom-system-prompt or disable with --disable-auto-system-prompt.
- Run ``python run_tempcompass.py --help`` for the full flag list.
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image

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
    
    if "_thumbnail1" in cache_dir_path:
        return "Focus on the temporal relationships by analyzing the current frame and the previous frame (bottom-left) and next frame (bottom-right) shown within the current frame. Pay attention to the temporal progression between these three key moments."
    
    if "_thumbnail2" in cache_dir_path:
        return "Focus on the temporal relationships by examining the current frame and the sequence of previous 2 frames and next 2 frames shown below the current frame. Analyze the temporal dynamics across this 5-frame sequence."
    
    if "optical_flow_prompts_v1" in cache_dir_path:
        return "Focus on the temporal relationships by analyzing the colored regions that indicate motion and changes between frames. Pay attention to the colored areas that highlight temporal dynamics and movement patterns."
    
    if "optical_flow_prompts_v2" in cache_dir_path:
        return "Focus on the temporal relationships by examining the outlined regions that represent motion and changes between frames. Pay attention to the border lines that indicate temporal dynamics and movement boundaries."
    
    return f"Focus on the temporal relationships by referring to the number written in the {position} corner of each frame."


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
    Maps all occurrences (case-insensitive, preserves original form).
    
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
            question_lower = modified_question.lower()
            query_lower = query.lower()
            
            matches = []
            start = 0
            while True:
                pos = question_lower.find(query_lower, start)
                if pos == -1:
                    break
                matches.append(pos)
                start = pos + 1
            
            for pos in reversed(matches):
                original_text = modified_question[pos:pos+len(query)]
                check_text = modified_question[pos:pos+len(query)+20]
                if not re.search(r'\(frame\s+#\d+\)', check_text, re.IGNORECASE):
                    replacement = f"{original_text} (frame {frame_num})"
                    modified_question = modified_question[:pos] + replacement + modified_question[pos+len(query):]
    
    return modified_question


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
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        try:
            img = Image.open(frame_path).convert('RGB')
            frame_np = np.array(img)
            frames.append(frame_np)
            valid_filenames.append(frame_file)
        except Exception as e:
            print(f"[WARN] Failed to load image {frame_path}: {e}")
            continue
    
    if len(frames) == 0:
        return None
    
    return np.array(frames, dtype=np.uint8), valid_filenames

DEFAULT_TEMPCOMPASS_KWARGS = {
    "pre_prompt": "",
    "post_prompt": {
        "multi-choice": "\nPlease directly give the best option:",
        "yes_no": "\nPlease answer yes or no:",
        "caption_matching": "\nPlease directly give the best option:",
        "captioning": "\nPlease directly give the best option:"
    }
}

def build_tempcompass_prompt(question: str, task_type: str, lmms_eval_specific_kwargs=None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = DEFAULT_TEMPCOMPASS_KWARGS
    
    pre_prompt = ""
    post_prompt = ""
    
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"].get(task_type, "")
    
    return f"{pre_prompt}{question}{post_prompt}"

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-Video inference from cached frames for TempCompass")
    parser.add_argument("--model-path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--model-name", type=str, default="llava_qwen")
    parser.add_argument("--conv-template", type=str, default="qwen_1_5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--save-jsonl", type=str, default=None)
    parser.add_argument("--task-type", type=str, required=False,
                        choices=["multi-choice", "yes_no", "caption_matching", "captioning"],
                        help="Optional fallback. If labels have task_type per record, that value is used.")
    parser.add_argument("--save-visualization", action="store_true")
    parser.add_argument("--no-pre-prompt", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    # Custom system prompt
    parser.add_argument("--custom-system-prompt", type=str, default=None,
                        help="Additional text to append to the system prompt")
    
    # Auto system prompt control
    parser.add_argument("--disable-auto-system-prompt", action="store_true",
                       help="Disable automatic system prompt selection based on cache-dir name (default: False, auto-selection enabled)")
    
    parser.add_argument("--copy-frames", type=int, default=1,
                    help="Number of times to duplicate each frame. If total exceeds 64, evenly distribute and fill from front.")
    
    # CLIP score threshold
    parser.add_argument("--score-thres", type=float, default=None,
                        help="Minimum CLIP score threshold for mapping. If the highest score is below this threshold, mapping is skipped. (default: None, always perform mapping)")
    
    return parser.parse_args()

def build_llava_inputs(
    image_processor, tokenizer, conv_template: str, video: List[torch.Tensor],
    video_time: float, frame_time: str, user_question_text: str, device: str,
    num_frames: int = None, frames_from_cache: bool = False, add_frame_prompt: bool = True,
    custom_system_prompt: str = None, cache_dir: str = None, disable_auto_system_prompt: bool = False
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
    
    copy_frames = getattr(args, 'copy_frames', 1)
    if frames_np is not None and copy_frames > 1:
        n = len(frames_np)
        limit = 64
        if n * copy_frames <= limit:
            rep = [copy_frames] * n
        else:
            base = limit // n
            left = limit - base * n
            rep = [base + 1 if i < left else base for i in range(n)]
        frames_np = np.concatenate(
            [np.repeat(frames_np[i][None], rep[i], axis=0) for i in range(n)], axis=0)
        frame_filenames = [frame_filenames[i] for i in range(n) for _ in range(rep[i])]
        stat = [{'frame_index': i, 'repeat': rep[i]} for i in range(n)]
        stat_path = os.path.join(frames_cache_dir + '_repeat_stat.jsonl')
        with open(stat_path, 'w', encoding='utf-8') as f:
            for d in stat:
                f.write(json.dumps(d, ensure_ascii=False) + '\\n')
    
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

def save_frames_visualization(frames_np: np.ndarray, save_path: str, video_id: str = "", frame_times: List[float] = None):
    """
    Visualize and save selected frames.
    
    Args:
        frames_np: (F, H, W, C) numpy array
        save_path: Path to save image
        video_id: Video ID (for title)
        frame_times: List of frame times in seconds
    """
    num_frames = len(frames_np)
    
    cols = min(8, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig_width = cols * 2.5
    fig_height = rows * 2.5
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    fig.suptitle(f'Selected Frames: {video_id}' if video_id else 'Selected Frames', 
                 fontsize=16, y=0.995)
    
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
            
            if frame_times and idx < len(frame_times):
                time_sec = frame_times[idx]
                minutes = int(time_sec // 60)
                seconds = int(time_sec % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                ax.set_title(f'Frame {idx+1}\n{time_str}', fontsize=10)
            else:
                ax.set_title(f'Frame {idx+1}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout(pad=1.5, w_pad=2.0, h_pad=2.0)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)

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
                    question_id = data.get("question_id") or data.get("video_id") or data.get("video")
                    if question_id:
                        processed_ids.add(question_id)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[WARN] Error reading existing pred.jsonl: {e}")
    
    return processed_ids

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

    print("Loading model...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.model_path, None, args.model_name,
        torch_dtype=args.torch_dtype,
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
            
            if os.path.exists(pred_jsonl_path):
                processed_question_ids = get_processed_question_ids(pred_jsonl_path)
                print(f"Resume mode: Found {len(processed_question_ids)} already processed items")
                out_f = open(pred_jsonl_path, "a", encoding="utf-8")
            else:
                out_f = open(pred_jsonl_path, "w", encoding="utf-8")
            
            print(f"Results will be saved to: {pred_jsonl_path}")
        
        processed, skipped, resumed = 0, 0, 0
        missing_frames_list = []
        samples = list(read_jsonl(args.labels_path))

        for sample in tqdm(samples, desc=f"Processing merged labels", ncols=100):
            resume_key = sample.get("question_id") or sample.get("video_id")
            if not resume_key:
                skipped += 1
                continue

            video_id = sample.get("video_id")
            if not video_id:
                skipped += 1
                continue

            if resume_key in processed_question_ids:
                resumed += 1
                continue

            frames_cache_dir = os.path.join(cache_root, video_id)
            
            if not os.path.exists(frames_cache_dir):
                skipped += 1
                missing_frames_list.append({"question_id": resume_key, "video_id": video_id, "reason": "directory_not_found"})
                continue

            record_task_type = sample.get("task_type") or args.task_type
            if not record_task_type:
                skipped += 1
                missing_frames_list.append({"question_id": resume_key, "video_id": video_id, "reason": "task_type_missing"})
                continue

            if record_task_type == "captioning":
                question = sample.get("mc_question", "")
            else:
                question = sample.get("question", "")
            prompt = build_tempcompass_prompt(question, record_task_type, DEFAULT_TEMPCOMPASS_KWARGS)
            
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
                    "model_response": out["prediction"],
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
                
                if args.save_visualization and save_dir and "frames_np" in out:
                    frame_times = out.get("frame_times", [])
                    viz_frames_dir = os.path.join(save_dir, "viz_frames")
                    os.makedirs(viz_frames_dir, exist_ok=True)
                    frames_image_path = os.path.join(viz_frames_dir, f"{video_id}_frames.png")
                    save_frames_visualization(out["frames_np"], frames_image_path, video_id, frame_times)
                
                processed += 1
            except Exception as e:
                skipped += 1
                missing_frames_list.append({"question_id": resume_key, "video_id": video_id, "reason": "inference_failed", "error": str(e)})

        if out_f:
            out_f.close()

        if missing_frames_list and save_dir:
            missing_json_path = os.path.join(save_dir, "missing_frames.json")
            with open(missing_json_path, "w", encoding="utf-8") as f:
                json.dump({"total_missing": len(missing_frames_list), "missing_video_ids": missing_frames_list}, 
                         f, ensure_ascii=False, indent=2)

        print(f"\nDone. processed={processed}, skipped={skipped}, resumed={resumed}, missing_frames={len(missing_frames_list)}")

if __name__ == "__main__":
    main()
