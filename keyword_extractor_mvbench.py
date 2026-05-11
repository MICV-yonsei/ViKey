"""
ViKey Stage 2: keyword extraction for MVBench.

Reads questions from ``IN_PATH`` (JSONL, one record per line), uses
Qwen2.5-7B-Instruct via vLLM to extract question-relevant key phrases, and
writes the same records back to ``OUT_PATH`` with an additional
``parsed_query`` field. The output JSONL is consumed by ``run_mvbench.py``
for Keyword-Frame Mapping.

Edit ``IN_PATH`` / ``OUT_PATH`` at the top of the file before running:

    python keyword_extractor_mvbench.py
"""

import json
import os
import re

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# -- Configuration -----------------------------------------------------------
IN_PATH = "/path/to/your/file.jsonl"
OUT_PATH = "/path/to/your/dir/output.jsonl"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 8


# -- Prompt ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

SYSTEM_PROMPT = (
    "You are a helpful assistant that only extracts keywords and outputs "
    "them as a Python list."
)

INSTRUCTION_RULES = (
    'Follow these rules carefully:\n\n'
    ' 1. **Identify Key Phrases**: Your goal is to extract key phrases from '
    'the question that refer to specific scenes, events, actions, or distinct items.\n'
    ' 2. **Exact Extraction**: The extracted phrases must appear exactly as '
    'they do in the question. Do not modify or rephrase them.\n'
    ' 3. **Empty List Condition**: If no relevant key phrases (as defined in '
    'Rule 1) are found in the question, you must return an empty list [].'
)

# Benchmark-specific few-shot examples.
INSTRUCTION_EXAMPLES = (
    'Example 1:\n'
    'Question: What happened after the person took the food?\n'
    'Your Answer: ["the person took the food"]\n'

    'Example 2:\n'
    'Question: What happened after the person closed the door?\n'
    'Your Answer: ["the person closed the door"]\n'

    'Example 3:\n'
    'Question: Based on the video, which choice shows the scene changes accurately?\n'
    'Your Answer: []\n'
)

INSTRUCTION = (
    INSTRUCTION_RULES + '\n\n' + INSTRUCTION_EXAMPLES +
    'Now:'
    'Question: {question}\n'
    'Your Answer:'
)


# -- Parsing -----------------------------------------------------------------
def extract_keywords_list(s: str) -> list[str]:
    """Parse the model output and return the list inside the first ``[...]``."""
    match = re.search(r'\[(.*?)\]', s)
    if not match:
        return []
    try:
        return re.findall(r"['\"]([^'\"]*)['\"]", match.group(1))
    except Exception:
        return []


def postprocess(keywords: list[str]) -> list[str]:
    """Drop the generic ``video`` token and any non-alphabetic / too-short entries."""
    if 'video' in keywords:
        keywords.remove('video')
    return [
        k.strip()
        for k in keywords
        if re.fullmatch(r"[A-Za-z ]+", k.strip()) and len(k.strip()) > 1
    ]


# -- Main --------------------------------------------------------------------
def main() -> None:
    """Run batched keyword extraction over ``IN_PATH`` and write ``OUT_PATH``."""
    model = LLM(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        max_model_len=1024,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    def format_prompt(question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INSTRUCTION.format(question=question)},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    prompts: list[str] = []
    examples: list[dict] = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(format_prompt(data.get("question", "")))
            examples.append(data)

    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_examples = examples[i:i + BATCH_SIZE]
            results = model.generate(batch_prompts, sampling_params)
            for ex, result in zip(batch_examples, results):
                output_text = result.outputs[0].text.strip()
                keywords = postprocess(extract_keywords_list(output_text))
                rec_out = dict(ex)
                rec_out["parsed_query"] = keywords
                fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
