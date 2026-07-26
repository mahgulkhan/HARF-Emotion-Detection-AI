"""
Parses a WhatsApp "Export Chat" .txt file into a pandas DataFrame.

Handles both common export formats:
  iOS:      [12/07/2026, 2:23:45 PM] Ali: yar mood off ho gaya today
  Android:  12/07/2026, 14:23 - Ali: yar mood off ho gaya today

Multi-line messages (no new timestamp) are merged into the previous message.
System lines (e.g. "Messages and calls are end-to-end encrypted...") are dropped.
"""
import re
import pandas as pd

IOS_PATTERN = re.compile(
    r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s([^:]+):\s?(.*)$"
)
ANDROID_PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.*)$"
)

SYSTEM_LINE_MARKERS = [
    "Messages and calls are end-to-end encrypted",
    "created group", "changed the subject", "changed this group's icon",
    "You were added", "added you", "left the group",
    "changed the group description", "Missed voice call", "Missed video call",
]

NO_CONTENT_MARKERS = [
    "<Media omitted>", "image omitted", "video omitted", "sticker omitted",
    "audio omitted", "GIF omitted", "document omitted",
    "This message was deleted", "You deleted this message",
]


def _is_system_line(text: str) -> bool:
    return any(marker.lower() in text.lower() for marker in SYSTEM_LINE_MARKERS)


def _has_no_content(text: str) -> bool:
    return any(marker.lower() in text.lower() for marker in NO_CONTENT_MARKERS)


def parse_chat(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    records = []
    current = None

    for line in raw_lines:
        line = line.rstrip("\n")
        line = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", line).strip()
        if not line:
            continue

        m = IOS_PATTERN.match(line) 
        if m:
            if current is not None:
                records.append(current)
            date, time, sender, message = m.groups()
            if _is_system_line(message):
                current = None
                continue
            current = {"timestamp": f"{date} {time}", "sender": sender.strip(), "message": message.strip()}
        else:
            if current is not None:
                current["message"] += " " + line

    if current is not None:
        records.append(current)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(
            "No messages could be parsed. Double check this is a raw WhatsApp "
            "'Export Chat' .txt file (not edited), and that it wasn't exported 'With Media'."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True, format="mixed")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df = df[~df["message"].apply(_has_no_content)].reset_index(drop=True)
    return df