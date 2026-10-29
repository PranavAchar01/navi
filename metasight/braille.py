"""Grade 1 (uncontracted) English Braille, and a compressor for slow haptic reading.

A cell is a 6-bit mask: bit 0 is dot 1 ... bit 5 is dot 6 (dots 1-2-3 down the left column,
4-5-6 down the right). Haptic Braille runs at a few words a minute, so answers are first cut to
terse cues (`compress`) and only then encoded. The abbreviation table lives in one place: ABBREV
and PHRASES below.
"""

from __future__ import annotations

import re

LETTERS = {
    "a": (1,), "b": (1, 2), "c": (1, 4), "d": (1, 4, 5), "e": (1, 5),
    "f": (1, 2, 4), "g": (1, 2, 4, 5), "h": (1, 2, 5), "i": (2, 4), "j": (2, 4, 5),
    "k": (1, 3), "l": (1, 2, 3), "m": (1, 3, 4), "n": (1, 3, 4, 5), "o": (1, 3, 5),
    "p": (1, 2, 3, 4), "q": (1, 2, 3, 4, 5), "r": (1, 2, 3, 5), "s": (2, 3, 4),
    "t": (2, 3, 4, 5), "u": (1, 3, 6), "v": (1, 2, 3, 6), "w": (2, 4, 5, 6),
    "x": (1, 3, 4, 6), "y": (1, 3, 4, 5, 6), "z": (1, 3, 5, 6),
}  # fmt: skip
PUNCT = {
    ",": (2,), ";": (2, 3), ":": (2, 5), ".": (2, 5, 6), "!": (2, 3, 5), "?": (2, 3, 6),
    "'": (3,), "-": (3, 6), "/": (3, 4), "(": (1, 2, 6), ")": (3, 4, 5),
    "%": (4, 5, 6),
}  # fmt: skip
NUMBER_SIGN = (3, 4, 5, 6)
CAPITAL_SIGN = (6,)
LETTER_SIGN = (5, 6)  # a-j straight after a number would read as digits without it
DIGITS = {d: LETTERS[c] for d, c in zip("1234567890", "abcdefghij", strict=True)}


def mask(dots: tuple[int, ...]) -> int:
    m = 0
    for d in dots:
        m |= 1 << (d - 1)
    return m


def dots_of(m: int) -> list[int]:
    return [d + 1 for d in range(6) if m >> d & 1]


LETTER_MASK = {c: mask(d) for c, d in LETTERS.items()}
DIGIT_MASK = {c: mask(d) for c, d in DIGITS.items()}
PUNCT_MASK = {c: mask(d) for c, d in PUNCT.items()}
NUMBER_MASK = mask(NUMBER_SIGN)
CAPITAL_MASK = mask(CAPITAL_SIGN)
LETTER_SIGN_MASK = mask(LETTER_SIGN)
_A_TO_J = set("abcdefghij")


def encode_word(word: str) -> list[int]:
    """Cells for one word (no spaces). Unknown characters are dropped."""
    letters = [c for c in word if c.isalpha() and c.lower() in LETTER_MASK]
    all_caps = len(letters) > 1 and all(c.isupper() for c in letters)
    out: list[int] = []
    num = False  # inside a number run
    caps_word = False
    for i, ch in enumerate(word):
        if ch.isdigit() and ch in DIGIT_MASK:
            if not num:
                out.append(NUMBER_MASK)
                num = True
            out.append(DIGIT_MASK[ch])
        elif num and ch in ".," and i + 1 < len(word) and word[i + 1].isdigit():
            out.append(PUNCT_MASK[ch])  # decimal point or thousands comma keeps the number open
        elif ch.lower() in LETTER_MASK and ch.isalpha():
            if num:
                num = False
                if ch.lower() in _A_TO_J:
                    out.append(LETTER_SIGN_MASK)
            if all_caps:
                if not caps_word:
                    out += [CAPITAL_MASK, CAPITAL_MASK]
                    caps_word = True
            elif ch.isupper():
                out.append(CAPITAL_MASK)
            out.append(LETTER_MASK[ch.lower()])
        elif ch in PUNCT_MASK:
            num = False
            out.append(PUNCT_MASK[ch])
    return out


def encode(text: str) -> list[list[int]]:
    """Text to words of cells; word boundaries are kept so the player can pause between them."""
    return [cells for w in text.split() if (cells := encode_word(w))]


def decode(words: list[list[int]]) -> str:
    """Inverse of `encode`, for tests and the judge view's caption."""
    rev = {v: k for k, v in LETTER_MASK.items()}
    rev_d = {v: k for k, v in DIGIT_MASK.items()}
    rev_p = {v: k for k, v in PUNCT_MASK.items()}
    out = []
    for cells in words:
        s, num, cap, caps_word = "", False, False, False
        for c in cells:
            if c == NUMBER_MASK:
                num = True
            elif c == LETTER_SIGN_MASK:
                num = False
            elif c == CAPITAL_MASK:
                if cap:
                    caps_word, cap = True, False
                else:
                    cap = True
            elif num and c in rev_d:
                s += rev_d[c]
            elif c in rev and not (num and c in rev_d):
                ch = rev[c]
                s += ch.upper() if cap or caps_word else ch
                cap = False
            elif c in rev_p:
                s += rev_p[c]
        out.append(s)
    return " ".join(out)


# ---------------------------------------------------------------- short forms
# Multi-word phrases are replaced first, then single words. Words absent from both stay as they
# are; STOP words are dropped. Keep entries lowercase.
PHRASES = {
    "in front of you": "ahead",
    "straight ahead": "ahead",
    "right in front": "ahead",
    "to your left": "left",
    "on your left": "left",
    "to your right": "right",
    "on your right": "right",
    "behind you": "behind",
    "next to": "beside",
    "on top of": "on",
    "i do not know": "?",
    "i don't know": "?",
    "not sure": "?",
    "no idea": "?",
    "last seen": "seen",
    "which object do you mean": "which",
}
ABBREV = {
    "stairs": "stairs", "staircase": "stairs", "sofa": "couch", "people": "people",
    "meters": "m", "meter": "m", "metres": "m", "metre": "m", "feet": "ft", "foot": "ft",
    "minutes": "min", "minute": "min", "seconds": "s", "second": "s",
    "approximately": "~", "about": "~", "around": "~", "roughly": "~",
    "and": "&", "with": "w/", "without": "w/o",
    "please": "", "careful": "caution", "warning": "warning", "danger": "warning",
    "hallway": "hall", "corridor": "hall", "entrance": "entry", "elevator": "lift",
}  # fmt: skip
STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "it", "its", "it's", "that", "this",
    "there", "of", "to", "your", "you", "i", "on", "in", "at", "by", "for", "should", "can",
    "may", "might", "currently", "just", "right-now", "found", "see", "seems", "looks",
}  # fmt: skip
_NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s+(m|ft|min|s)\b")
_OCLOCK = re.compile(r"(\d{1,2})\s+o'?clock")


def compress(text: str, max_cells: int = 24) -> str:
    """Cut an answer to a terse cue, e.g. "Your keys are on the table, 4 meters ahead." ->
    "keys table 4m ahead". Whole trailing words are dropped past `max_cells` Braille cells, since
    the wearer cannot wait a minute for a sentence."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"[.,;!]+(?=\s|$)", " ", t)
    for phrase, short in sorted(PHRASES.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(rf"\b{re.escape(phrase)}\b", f" {short} ", t)
    t = _OCLOCK.sub(r"\1oc", t)
    words: list[str] = []
    for w in t.split():
        w = w.strip('"()')
        if not w:
            continue
        if w in STOP and w not in ("l", "r"):
            continue
        w = ABBREV.get(w, w)
        if w:
            words.append(w)
    t = re.sub(r"\s+\?", "?", _NUM_UNIT.sub(r"\1\2", " ".join(words)))
    out: list[str] = []
    used = 0
    for w in t.split():
        n = len(encode_word(w))
        if out and used + n + 1 > max_cells:
            break
        out.append(w)
        used += n + 1
    return " ".join(out)
