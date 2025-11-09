#!/usr/bin/env python
"""Optimize a prompt so a model mimics the user's tone in replai.json using DSPy GEPA."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Iterable, List

import dspy
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a ReplAI-style prompt with GEPA using the replai.json dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        default="data/training-data/replai.json",
        help="Path to the JSON conversation file that contains the user's chat history.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Number of assistant turns to use when fitting GEPA (after filtering).",
    )
    parser.add_argument(
        "--max-context-turns",
        type=int,
        default=6,
        help="Maximum number of recent turns to feed into each training example.",
    )
    parser.add_argument(
        "--output",
        default="train/gepa/optimized_replai_prompt.json",
        help="Where to save the optimized DSPy module.",
    )
    parser.add_argument(
        "--generator-model",
        default=os.getenv("REPLAI_GENERATOR_MODEL", "NousResearch/Hermes-4-14B"),
        help="Model ID served by your inference endpoint for generation.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.getenv("ENDPOINT_URL"),
        help="Inference endpoint URL that serves the generator model.",
    )
    parser.add_argument(
        "--generator-api-key",
        default=os.getenv("REPLAI_GENERATOR_API_KEY", os.getenv("OPENAI_API_KEY", "local")),
        help="API key expected by the inference endpoint.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("REPLAI_JUDGE_MODEL", "anthropic/claude-sonnet-4-5-20250929"),
        help="LM used for the style-matching metric and GEPA reflection.",
    )
    parser.add_argument(
        "--anthropic-api-key",
        default=os.getenv("ANTHROPIC_API_KEY"),
        help="API key for the Anthropic judge/reflection model.",
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="GEPA auto setting that trades off speed vs. search depth.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Parallel GEPA workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def format_turn(role: str, content: str) -> str:
    speaker = "Me" if role == "assistant" else "Them"
    return f"{speaker}: {content.strip()}"


def iter_assistant_examples(
    raw_conversations: Iterable[dict],
    max_context_turns: int,
) -> Iterable[dspy.Example]:
    for conversation in raw_conversations:
        messages = conversation.get("messages") or conversation.get("openai_messages") or []
        if not isinstance(messages, list):
            continue
        for idx, turn in enumerate(messages):
            if turn.get("role") != "assistant":
                continue
            history = messages[max(0, idx - max_context_turns): idx]
            if not history:
                continue
            reference = turn.get("content", "").strip()
            if len(reference) < 2:
                continue
            context = "\n".join(format_turn(m["role"], m.get("content", "")) for m in history)
            yield dspy.Example(
                context=context,
                reference_response=reference,
            ).with_inputs("context")


def load_trainset(data_path: Path, num_examples: int, max_context_turns: int) -> List[dspy.Example]:
    if not data_path.exists():
        raise FileNotFoundError(f"Conversation file not found: {data_path}")
    with data_path.open("r", encoding="utf-8") as fh:
        raw_data = json.load(fh)
    examples = list(iter_assistant_examples(raw_data, max_context_turns))
    if not examples:
        raise ValueError("No usable assistant turns were found in the dataset.")
    random.shuffle(examples)
    subset = examples[:num_examples]
    print(f"Loaded {len(subset)} examples (from {len(examples)} available) for GEPA.")
    return subset


class ReplAIResponse(dspy.Signature):
    """Signature for predicting the next message from the user's perspective."""

    context = dspy.InputField(desc="Recent conversation turns, alternating between Them and Me")
    response = dspy.OutputField(desc="The next short reply I would send, matching my tone and slang")


class StyleJudge(dspy.Signature):
    """Judge that compares a candidate reply with the real user reply."""

    context = dspy.InputField(desc="Conversation context to ground the decision")
    candidate_response = dspy.InputField(desc="Model response produced with the current prompt")
    reference_response = dspy.InputField(desc="Authentic response written by the user")
    style_score = dspy.OutputField(
        desc="Float between 0 and 1 indicating how well the candidate matches the user's style",
    )


class ReplAIBehaviorGenerator(dspy.Module):
    """Thin wrapper around the LM so GEPA can rewrite its prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(ReplAIResponse)

    def forward(self, context: str) -> dspy.Prediction:
        result = self.generate(context=context)
        return dspy.Prediction(response=result.response)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def parse_score(raw_value) -> float:
    if raw_value is None:
        return 0.0
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    match = re.search(r"(\d+(\.\d+)?)", str(raw_value))
    return float(match.group(1)) if match else 0.0


def make_style_metric(judge_lm: dspy.LM):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
        candidate = (pred.response or "").strip()
        if not candidate:
            return 0.0
        with dspy.context(lm=judge_lm):
            judge = dspy.Predict(StyleJudge)
            result = judge(
                context=gold.context,
                candidate_response=candidate,
                reference_response=gold.reference_response,
            )
        score = parse_score(result.style_score)
        if score < 0:
            score = 0.0
        if score > 1:
            score = score / 100 if score <= 100 else 1 / (1 + math.exp(-score))
        return clamp(score)

    return metric


def configure_generator_lm(model_id: str, endpoint_url: str | None, api_key: str) -> None:
    if not endpoint_url:
        raise ValueError("An inference ENDPOINT_URL must be provided via flag or environment.")
    lm = dspy.LM(
        f"openai/{model_id}",
        api_base=endpoint_url,
        api_key=api_key,
        model_type="chat",
        temperature=0.7,  # Increase from 0.3 to reduce repetition
        max_tokens=256,   # Reduce from 16384 - chat replies should be SHORT!
    )
    dspy.configure(lm=lm)


def build_anthropic_lm(model_id: str, api_key: str | None) -> dspy.LM:
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required to run the judge/reflection models.")
    return dspy.LM(
        model_id,
        api_key=api_key,
        temperature=0.3,
        max_tokens=8096,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    random.seed(args.seed)

    configure_generator_lm(args.generator_model, args.endpoint_url, args.generator_api_key)
    judge_lm = build_anthropic_lm(args.judge_model, args.anthropic_api_key)

    trainset = load_trainset(Path(args.data_path), args.num_examples, args.max_context_turns)
    module = ReplAIBehaviorGenerator()

    print("Launching GEPA to optimize the user-behavior prompt...")
    tp = dspy.GEPA(
        metric=make_style_metric(judge_lm),
        auto=args.auto,
        num_threads=args.num_threads,
        reflection_lm=judge_lm,
    )
    optimized_module = tp.compile(module, trainset=trainset)
    optimized_module.save(args.output)
    print(f"Optimized module saved to {args.output}")


if __name__ == "__main__":
    main()
