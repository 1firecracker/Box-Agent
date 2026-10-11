"""Per-turn deliverable-language hint shared by the CLI and ACP adapters.

The system prompt says deliverables follow the user's language, but evals
showed a model writing a Chinese blurb for an English request because the
reference file was Chinese ("matching the source material"). A one-line hint
next to the user's message is applied where the decision is made.

Only non-Chinese requests get a hint: the base prompt is Chinese, so Chinese
requests already default correctly, and very short messages ("ok", "continue")
carry no reliable language signal. Reply language is not addressed here; hosts
own it (ACP ``ui_language``).
"""

from __future__ import annotations

import re

# About six English words: short continuations ("continue after restart") carry no
# deliverable and should not gain a hint.
_MIN_LETTERS = 30
_MAX_CJK_SHARE = 0.1
# Spans that say nothing about the user's language: code, URLs, paths, tags,
# and slash commands such as "/pptx".
_NOISE = re.compile(
    r"```.*?```|`[^`]*`|https?://\S+|(?:[\w.-]+/)+[\w.-]+|<[^>]+>|(?<!\S)/[\w-]+", re.DOTALL,
)

_HINTS = {
    "English": (
        "[Deliverable language: English. The user wrote this request in English; write any "
        "files you create (documents, copy, reports, slides) in English too, even when the "
        "reference material is in another language, unless the user asks for a specific language.]"
    ),
    "Japanese": (
        "[成果物の言語：日本語。ユーザーは日本語で依頼しています。参考資料が別の言語でも、"
        "作成するファイル（文書・コピー・レポート・スライド）は日本語で書いてください。"
        "ユーザーが別の言語を指定した場合はそれに従ってください。]"
    ),
}


def request_language(text: str) -> str | None:
    """Return "English" or "Japanese" for a clearly non-Chinese request, else None."""
    content = _NOISE.sub(" ", text)
    kana = sum(1 for char in content if "぀" <= char <= "ヿ")
    han = sum(1 for char in content if "一" <= char <= "鿿")
    latin = sum(1 for char in content if char.isascii() and char.isalpha())
    if kana >= 5 and kana >= han * 0.3:
        return "Japanese"
    if latin >= _MIN_LETTERS and (han + kana) / (latin + han + kana) < _MAX_CJK_SHARE:
        return "English"
    return None


def deliverable_language_hint(text: str) -> str | None:
    """The hint to place beside this user message, or None."""
    language = request_language(text)
    return _HINTS.get(language) if language else None


def with_deliverable_language_hint(text: str) -> str:
    """Append the hint to a user message when its language warrants one."""
    hint = deliverable_language_hint(text)
    return f"{text.rstrip()}\n\n{hint}" if hint else text


__all__ = ["deliverable_language_hint", "request_language", "with_deliverable_language_hint"]
