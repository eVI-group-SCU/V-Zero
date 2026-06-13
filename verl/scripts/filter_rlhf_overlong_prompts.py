#!/usr/bin/env python3
"""Offline filter RLHF parquet files with rollout/teacher-crop prompt lengths."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load RLHF data through RLHFDataset, filter student rollout prompts, "
            "then filter teacher-crop prompts using multimodal-expanded token lengths."
        )
    )
    parser.add_argument("--model", required=True, help="Tokenizer/processor model path.")
    parser.add_argument("--train-file", required=True, action="append", help="Input parquet/json/jsonl file.")
    parser.add_argument("--output-file", help="Output parquet path. Required unless --dry-run is set.")
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=0,
        help=(
            "Reserved response budget when filtering teacher-crop replay sequences. "
            "Set this to data.max_response_length for a training-compatible filter."
        ),
    )
    parser.add_argument(
        "--teacher-max-model-len",
        type=int,
        default=int(os.environ.get("TEACHER_MAX_MODEL_LEN", 32768)),
        help=(
            "Teacher vLLM max_model_len. Defaults to TEACHER_MAX_MODEL_LEN env, then 32768, "
            "matching the crop training scripts."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="Randomly select this many samples before filtering.")
    parser.add_argument("--seed", type=int, help="Random seed for --max-samples.")
    parser.add_argument("--num-proc", type=int, default=4, help="Number of HF datasets filter workers.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--video-key", default="videos")
    parser.add_argument("--audio-key", default="audios")
    parser.add_argument(
        "--image-patch-size",
        type=int,
        help=(
            "Image patch size for qwen_vl_utils. Defaults to processor.image_processor.patch_size "
            "(Qwen3-VL variants may differ), then 14."
        ),
    )
    parser.add_argument("--teacher-crop-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-crop-prompt-key", default="teacher_raw_prompt")
    parser.add_argument("--teacher-crop-image-key", default="teacher_images")
    parser.add_argument(
        "--skip-student-prompt-filter",
        action="store_true",
        help="Assume the student rollout prompt has already been length-filtered and only filter teacher-crop prompts.",
    )
    parser.add_argument(
        "--progress-log-interval",
        type=int,
        default=1000,
        help="When --num-proc=1, print a fixed progress line every N samples in addition to tqdm.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only report counts; do not write a parquet file.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolved_image_patch_size(processor, explicit_patch_size: int | None) -> int:
    if explicit_patch_size is not None:
        return explicit_patch_size
    return int(getattr(getattr(processor, "image_processor", None), "patch_size", 14))


def _get_mm_processor_kwargs(processor, audio_data, configured_kwargs):
    mm_processor_kwargs = dict(configured_kwargs or {})
    if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
        sampling_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
        if sampling_rate is not None:
            mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
    return mm_processor_kwargs


def _expanded_prompt_length(
    *,
    dataset: RLHFDataset,
    processor,
    tokenizer,
    doc,
    prompt_key: str,
    crop_image_key: str | None = None,
) -> int:
    messages = dataset._build_messages(doc, key=prompt_key, crop_image_key=crop_image_key)
    apply_kwargs = dict(**dataset.apply_chat_template_kwargs)
    if dataset.tool_schemas is not None:
        apply_kwargs["tools"] = dataset.tool_schemas

    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **apply_kwargs)
    images, videos, audios = dataset._process_multi_modal_info(
        messages,
        image_patch_size=dataset.image_patch_size,
        config=dataset.config,
    )
    if images is None and videos is None and audios is None:
        return len(
            tokenizer(
                text=raw_prompt,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        )

    model_inputs = build_multimodal_processor_inputs(
        processor,
        text=[raw_prompt],
        images=images,
        videos=videos,
        audio=audios,
        mm_processor_kwargs=_get_mm_processor_kwargs(processor, audios, dataset.mm_processor_kwargs),
    )
    return len(normalize_token_ids(model_inputs["input_ids"]))


def filter_teacher_crop_prompts(
    dataset: RLHFDataset,
    processor,
    tokenizer,
    *,
    teacher_prompt_key: str,
    teacher_image_key: str,
    teacher_prompt_budget: int,
    num_proc: int,
    progress_log_interval: int,
):
    def keep_doc(doc) -> bool:
        try:
            teacher_prompt_len = _expanded_prompt_length(
                dataset=dataset,
                processor=processor,
                tokenizer=tokenizer,
                doc=doc,
                prompt_key=teacher_prompt_key,
                crop_image_key=teacher_image_key,
            )
            return teacher_prompt_len <= teacher_prompt_budget
        except Exception:
            print("Error processing one of the teacher-crop samples, skipping...", flush=True)
            traceback.print_exc()
            return False

    before = len(dataset.dataframe)
    desc = f"Filtering teacher-crop prompts longer than {teacher_prompt_budget} expanded tokens"
    if num_proc == 1:
        kept_indices: list[int] = []
        with tqdm(total=before, desc=desc, unit="sample", dynamic_ncols=True) as progress:
            for row_index, doc in enumerate(dataset.dataframe):
                if keep_doc(doc):
                    kept_indices.append(row_index)
                processed = row_index + 1
                progress.update(1)
                if progress_log_interval > 0 and processed % progress_log_interval == 0:
                    print(
                        "teacher-crop filter progress: "
                        f"processed={processed}/{before} "
                        f"kept={len(kept_indices)} "
                        f"filtered={processed - len(kept_indices)}",
                        flush=True,
                    )
        filtered = dataset.dataframe.select(kept_indices)
    else:
        filtered = dataset.dataframe.filter(
            keep_doc,
            num_proc=num_proc,
            desc=desc,
        )
    print(f"teacher-crop filter dataset len: {len(filtered)} / {before}", flush=True)
    return filtered


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.output_file:
        raise ValueError("--output-file is required unless --dry-run is set")
    output = Path(args.output_file) if args.output_file else None
    if output is not None:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer/processor from {args.model}", flush=True)
    tokenizer = hf_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    processor = hf_processor(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    image_patch_size = _resolved_image_patch_size(processor, args.image_patch_size)

    config = OmegaConf.create(
        {
            "prompt_key": args.prompt_key,
            "image_key": args.image_key,
            "video_key": args.video_key,
            "audio_key": args.audio_key,
            "image_patch_size": image_patch_size,
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": not args.skip_student_prompt_filter,
            "filter_overlong_prompts_workers": args.num_proc,
            "truncation": "error",
            "return_raw_chat": True,
            "return_multi_modal_inputs": True,
            "shuffle": args.max_samples > 0,
            "seed": args.seed,
            "teacher_crop": {
                "enabled": args.teacher_crop_enabled,
                "prompt_key": args.teacher_crop_prompt_key,
                "image_key": args.teacher_crop_image_key,
            },
        }
    )

    print(f"Input files: {args.train_file}", flush=True)
    print(
        "Filtering with "
        f"max_prompt_length={args.max_prompt_length}, num_proc={args.num_proc}, "
        f"skip_student_prompt_filter={args.skip_student_prompt_filter}, "
        f"prompt_key={args.prompt_key}, image_key={args.image_key}, "
        f"image_patch_size={image_patch_size}, "
        f"max_response_length={args.max_response_length}, max_samples={args.max_samples}, seed={args.seed}",
        flush=True,
    )
    dataset = RLHFDataset(
        data_files=args.train_file,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
        max_samples=args.max_samples,
    )
    if args.teacher_crop_enabled:
        teacher_max_model_len = args.teacher_max_model_len
        teacher_prompt_budget = min(args.max_prompt_length, teacher_max_model_len - args.max_response_length)
        if teacher_prompt_budget < 0:
            raise ValueError(
                f"teacher prompt budget is negative: {teacher_max_model_len=} - {args.max_response_length=}"
            )
        if args.max_response_length <= 0:
            print(
                "WARNING: --max-response-length is 0; teacher-crop filtering will not reserve response tokens.",
                flush=True,
            )
        print(
            "Teacher-crop filtering with expanded multimodal length: "
            f"teacher_max_model_len={teacher_max_model_len}, "
            f"reserved_response_tokens={args.max_response_length}, "
            f"teacher_prompt_budget={teacher_prompt_budget}, "
            f"teacher_prompt_key={args.teacher_crop_prompt_key}, "
            f"teacher_image_key={args.teacher_crop_image_key}",
            flush=True,
        )
        dataset.dataframe = filter_teacher_crop_prompts(
            dataset,
            processor,
            tokenizer,
            teacher_prompt_key=args.teacher_crop_prompt_key,
            teacher_image_key=args.teacher_crop_image_key,
            teacher_prompt_budget=teacher_prompt_budget,
            num_proc=args.num_proc,
            progress_log_interval=args.progress_log_interval,
        )

    if args.dry_run:
        print(f"Dry run complete: filtered_len={len(dataset.dataframe)}", flush=True)
        return

    assert output is not None
    tmp_output = output.with_name(f"{output.name}.tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    print(f"Saving filtered dataset len={len(dataset.dataframe)} to {output}", flush=True)
    dataset.dataframe.to_parquet(str(tmp_output))
    os.replace(tmp_output, output)
    print(f"Done: {output}", flush=True)


if __name__ == "__main__":
    main()
