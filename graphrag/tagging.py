#!/usr/bin/env python3
"""
Utilities for deriving lightweight tags and signals from normalized conversations.

This module enriches the unified conversation schema with additional metadata that
can later be promoted to graph properties or used as weak labels during retrieval.

Key capabilities:

* Conversation level aggregates (turn-taking ratios, participant statistics, etc.)
* Message level annotations (time-of-day buckets, lexical statistics, gaps)
* Simple heuristics for tagging topical signals (emoji usage, question density)
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytz

from utils.filter import (  # noqa: F401 - re-exported helpers
    calculate_my_avg_message_length,
    calculate_my_turn_proportion,
    get_time_of_day_pacific,
)


PACIFIC_TZ = pytz.timezone("America/Los_Angeles")


def _parse_timestamp(timestamp: Optional[str]) -> Optional[datetime]:
    if not timestamp:
        return None

    ts = timestamp.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def _seconds_between(prev: Optional[datetime], current: Optional[datetime]) -> Optional[float]:
    if not prev or not current:
        return None
    return (current - prev).total_seconds()


def _hash_message(conversation_id: str, message: Dict[str, Any]) -> str:
    message_id = message.get("message_id")
    payload = "|".join(
        [
            conversation_id,
            message_id or "",
            message.get("author") or "",
            message.get("timestamp") or "",
            (message.get("content") or "").strip(),
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class MessageAnnotation:
    message_id: str
    conversation_id: str
    author: Optional[str]
    role: Optional[str]
    timestamp: Optional[str]
    time_of_day: Optional[str]
    word_count: int
    char_count: int
    contains_question: bool
    contains_emoji: bool
    gap_seconds: Optional[float]
    synthetic_id: str


@dataclass
class ConversationAnnotation:
    conversation_id: str
    participants: List[str]
    participant_turn_counts: Dict[str, int]
    assistant_turn_ratio: float
    user_turn_ratio: float
    assistant_avg_word_count: float
    time_of_day_buckets: List[str]
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    duration_hours: Optional[float]
    message_count: int
    emoji_density: float
    question_density: float
    dominant_time_bucket: Optional[str]


class TagGenerator:
    """
    Generate derived tags for the unified conversation schema.
    """

    def __init__(self, timezone_name: str = "America/Los_Angeles") -> None:
        self.timezone_name = timezone_name
        self.local_tz = pytz.timezone(timezone_name)

    def annotate_conversation(
        self, conversation: Dict[str, Any], conversation_id: Optional[str] = None
    ) -> Tuple[ConversationAnnotation, List[MessageAnnotation]]:
        full_messages = conversation.get("full_metadata_messages", [])
        openai_messages = conversation.get("openai_messages", [])

        convo_id = conversation_id or self._build_conversation_id(conversation)
        participants = conversation.get("recipients", []) or []
        participants = list(dict.fromkeys(participants))

        role_counter = Counter(msg.get("role") for msg in openai_messages if msg.get("role"))
        assistant_turns = role_counter.get("assistant", 0)
        user_turns = role_counter.get("user", 0)

        assistant_avg_words = calculate_my_avg_message_length(openai_messages)
        assistant_turn_ratio = calculate_my_turn_proportion(openai_messages)
        user_turn_ratio = 100.0 - assistant_turn_ratio

        toc_counter: Counter[str] = Counter()
        emoji_count = 0
        question_count = 0
        message_annotations: List[MessageAnnotation] = []

        prev_timestamp: Optional[datetime] = None

        for msg in full_messages:
            timestamp = msg.get("timestamp")
            dt = _parse_timestamp(timestamp)
            localized_bucket: Optional[str] = None
            if timestamp:
                localized_bucket = get_time_of_day_pacific(timestamp)
                toc_counter[localized_bucket] += 1

            content = msg.get("content") or ""
            message_emoji_count = _count_emojis(content)
            emoji_count += message_emoji_count
            question = "?" in content
            if question:
                question_count += 1

            gap_seconds = _seconds_between(prev_timestamp, dt)
            prev_timestamp = dt

            word_count = len(content.split())
            char_count = len(content)

            synthetic_id = _hash_message(convo_id, msg)

            message_annotations.append(
                MessageAnnotation(
                    message_id=msg.get("message_id") or synthetic_id,
                    synthetic_id=synthetic_id,
                    conversation_id=convo_id,
                    author=msg.get("author"),
                    role=self._infer_role(msg, conversation),
                    timestamp=timestamp,
                    time_of_day=localized_bucket,
                    word_count=word_count,
                    char_count=char_count,
                    contains_question=question,
                    contains_emoji=bool(message_emoji_count),
                    gap_seconds=gap_seconds,
                )
            )

        first_ts = conversation.get("first_message_timestamp")
        last_ts = conversation.get("last_message_timestamp")
        duration_hours = None
        start_dt = _parse_timestamp(first_ts)
        end_dt = _parse_timestamp(last_ts)
        if start_dt and end_dt:
            duration_hours = (end_dt - start_dt).total_seconds() / 3600.0

        total_messages = len(full_messages) if full_messages else 0
        emoji_density = (emoji_count / total_messages) if total_messages else 0.0
        question_density = (question_count / total_messages) if total_messages else 0.0

        annotation = ConversationAnnotation(
            conversation_id=convo_id,
            participants=participants,
            participant_turn_counts=self._participant_turn_counts(full_messages),
            assistant_turn_ratio=assistant_turn_ratio,
            user_turn_ratio=user_turn_ratio,
            assistant_avg_word_count=assistant_avg_words,
            time_of_day_buckets=sorted(toc_counter.keys()),
            first_timestamp=first_ts,
            last_timestamp=last_ts,
            duration_hours=duration_hours,
            message_count=total_messages,
            emoji_density=emoji_density,
            question_density=question_density,
            dominant_time_bucket=toc_counter.most_common(1)[0][0] if toc_counter else None,
        )

        return annotation, message_annotations

    def enrich_conversation(
        self, conversation: Dict[str, Any], conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrich a single conversation dictionary with derived metadata inplace.
        """
        conv_annotation, message_annotations = self.annotate_conversation(conversation, conversation_id)

        conversation["derived"] = {
            "conversation_id": conv_annotation.conversation_id,
            "participants": conv_annotation.participants,
            "participant_turn_counts": conv_annotation.participant_turn_counts,
            "assistant_turn_ratio": conv_annotation.assistant_turn_ratio,
            "user_turn_ratio": conv_annotation.user_turn_ratio,
            "assistant_avg_word_count": conv_annotation.assistant_avg_word_count,
            "time_of_day_buckets": conv_annotation.time_of_day_buckets,
            "dominant_time_bucket": conv_annotation.dominant_time_bucket,
            "duration_hours": conv_annotation.duration_hours,
            "message_count": conv_annotation.message_count,
            "emoji_density": conv_annotation.emoji_density,
            "question_density": conv_annotation.question_density,
        }

        enriched_messages: List[Dict[str, Any]] = []
        for msg, annotation in zip(conversation.get("full_metadata_messages", []), message_annotations):
            enriched = dict(msg)
            enriched["derived"] = {
                "synthetic_id": annotation.synthetic_id,
                "role": annotation.role,
                "time_of_day": annotation.time_of_day,
                "word_count": annotation.word_count,
                "char_count": annotation.char_count,
                "contains_question": annotation.contains_question,
                "contains_emoji": annotation.contains_emoji,
                "gap_seconds": annotation.gap_seconds,
            }
            enriched_messages.append(enriched)

        conversation["full_metadata_messages"] = enriched_messages

        return conversation

    def enrich_corpus(
        self, conversations: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        enriched = []
        seen_ids = set()
        for convo in conversations:
            annotation, _ = self.annotate_conversation(convo)
            convo_id = annotation.conversation_id
            if convo_id in seen_ids:
                suffix = hashlib.sha256(f"{convo_id}|{len(seen_ids)}".encode("utf-8")).hexdigest()[:8]
                convo_id = f"{convo_id}-{suffix}"
            seen_ids.add(convo_id)
            enriched.append(self.enrich_conversation(convo, conversation_id=convo_id))
        return enriched

    def _build_conversation_id(self, conversation: Dict[str, Any]) -> str:
        recipients = "|".join(sorted(conversation.get("recipients") or []))
        first_ts = conversation.get("first_message_timestamp") or ""
        last_ts = conversation.get("last_message_timestamp") or ""
        source = conversation.get("source") or ""
        raw = "|".join([source, recipients, first_ts, last_ts])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _participant_turn_counts(messages: List[Dict[str, Any]]) -> Dict[str, int]:
        counter: Dict[str, int] = defaultdict(int)
        for msg in messages:
            author = msg.get("author")
            if author:
                counter[author] += 1
        return dict(counter)

    @staticmethod
    def _infer_role(message: Dict[str, Any], conversation: Dict[str, Any]) -> Optional[str]:
        author = message.get("author")
        if not author:
            return None

        recipients = conversation.get("recipients") or []
        if author in recipients:
            return "user"
        return "assistant"


def _count_emojis(text: str) -> int:
    count = 0
    for ch in text:
        if unicodedata.category(ch) in {"So", "Sk"} and ord(ch) >= 0x1F000:
            count += 1
    return count


__all__ = [
    "TagGenerator",
    "ConversationAnnotation",
    "MessageAnnotation",
    "calculate_my_avg_message_length",
    "calculate_my_turn_proportion",
    "get_time_of_day_pacific",
]


