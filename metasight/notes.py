"""Conversation notes: an explicit start and stop, a spoken recording cue, a local transcript and a
summary the wearer can ask about later. Design in docs/notes.md.

The words come from the same Muse Voice Transcribe session that hears questions; while a note is
recording, finals go here instead of the question path. Nothing is stored except
`sessions/notes/<id>.json` on this Mac (owner-only). The only network calls are the transcription
stream that is already open and one summary call to whatever language model is configured.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

START_LINE = "Recording this conversation. Please tell the people with you."
REMIND_LINE = "Still recording."
STOP_LINE = "Stopped. Writing the notes."
READY_LINE = "Notes saved."
EMPTY_LINE = "Stopped. Nobody said anything, so there is nothing to save."
REMIND_S = 300.0  # a spoken reminder this often, so nobody forgets the microphone is on
MAX_S = 3 * 3600.0  # a note never records longer than this
CHUNK_CHARS = 12000  # summarised piece by piece beyond this
SHORT_CHARS = 1500  # a note this short goes whole when no line shares a word with the question
FIND_CHARS = 3000  # most transcript text one question gets back

_START = re.compile(
    r"\b(take (some )?notes|start (taking )?notes|start (the )?recording|"
    r"record (this|the|our) (conversation|meeting|talk|appointment)|start (a )?note)\b"
)
_STOP = re.compile(
    r"\b(stop (the )?(recording|notes|taking notes)|end (the )?(recording|notes)|"
    r"that'?s (all|enough) for the notes|finish (the )?notes)\b"
)
_ASK = re.compile(
    r"\b(what did (the |my |he |she |they )?\w* ?(say|tell|mention|recommend|decide)|"
    r"what was said|in (my|the) notes|from (my|the) notes|my notes|"
    r"what (were|are) the (notes|takeaways|action items)|remind me what)\b"
)
_STOPWORDS = frozenset(
    "a an the of to in on at for and or is are was were be do did does what who when where how "
    "say said tell told mention about my me i you he she they we it that this with from by "
    "doctor my notes remind".split()
)

SUMMARY_SYSTEM = (
    "You write notes from the transcript of a conversation. First line: a title of at most six "
    "words. Then at most six short bullet lines of what was said, decided or asked. Keep numbers, "
    "doses, dates and times exactly as spoken. Use only the transcript; add nothing. "
    "Do not copy the transcript line by line: say it shorter."
)
LABELS = (
    ' Lines start with "you" (the wearer) or "person 1", "person 2" (anonymous others): '
    "use those labels, never a name."
)
MERGE_SYSTEM = (
    "You merge partial notes of one conversation into one set. First line: a title of at most "
    "six words. Then at most eight short bullet lines. Keep numbers, doses, dates and times "
    "exactly. Add nothing that is not in the partial notes."
)


def _say(line: dict) -> str:
    return f"{line['who']}: {line['text']}" if line.get("who") else line["text"]


_NUM_WORDS = {
    w: str(n)
    for n, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
} | {"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "hundred": "100"}


def _norm(text: str) -> str:
    t = re.sub(r"[^a-z0-9. ]", " ", text.lower().replace("\u2019", "'"))
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = " ".join(_NUM_WORDS.get(w, w) for w in t.split())
    return re.sub(r"\.(?!\d)", "", t)


def unsupported(answer: str, transcript: str) -> list[str]:
    """What in a spoken answer the transcript does not back: quoted phrases that were never said,
    and numbers that never appear (spelled out counts: "twenty" backs 20). Empty means it holds."""
    said = f" {_norm(transcript)} "
    bad = []
    for q in re.findall(r"[\"\u201c]([^\"\u201d]{3,})[\"\u201d]", answer):
        if f" {_norm(q)} " not in said:
            bad.append(q)
    plain = re.sub(r"[\"\u201c][^\"\u201d]*[\"\u201d]", " ", answer)
    for n in re.findall(r"\d+(?:\.\d+)?", _norm(plain)):
        if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", said):
            bad.append(n)
    return bad


def caption(text: str, who: str | None, final: bool) -> dict:
    """A live caption for the phone screen: the words so far, or a finished line."""
    ev = {"type": "caption", "text": text, "final": final}
    if who:
        ev["who"] = who
    return ev


def notes_command(text: str) -> str | None:
    """ "start" or "stop" when the wearer asked for it, else None."""
    t = " ".join(text.lower().replace("’", "'").split())
    if _STOP.search(t):
        return "stop"
    if _START.search(t):
        return "start"
    return None


def asks_notes(text: str) -> bool:
    """A question about a past conversation, for the tool path."""
    return bool(_ASK.search(text.lower()))


@dataclass
class Note:
    id: str
    started: float
    ended: float | None = None
    lines: list[dict] = field(default_factory=list)  # {"t": seconds from start, "text": str}
    title: str = ""
    summary: str = ""

    def text(self) -> str:
        """The transcript, one line per turn; a line carries "you: " or "person 2: " when the
        direction evidence said who spoke."""
        return "\n".join(_say(line) for line in self.lines)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "started": self.started,
            "ended": self.ended,
            "title": self.title,
            "summary": self.summary,
            "lines": self.lines,
        }


class Notes:
    def __init__(self, root: Path, now=time.time):
        self.root, self.now = Path(root), now
        self.cur: Note | None = None
        self.last_remind = 0.0
        self.votes: list[tuple[float, object]] = []  # (time, who) for each frame with a voice
        self.who_ids: dict[object, int] = {}  # who -> the n in "person n", in order first heard
        self.line_from = 0.0  # start of the next line's window

    @property
    def recording(self) -> bool:
        return self.cur is not None

    def event(self, state: str, say: str | None = None, **extra) -> dict:
        ev = {"type": "notes", "state": state, **extra}
        if self.cur:
            ev["id"], ev["lines"] = self.cur.id, len(self.cur.lines)
        if say:
            ev["say"] = say
        return ev

    # ------------------------------------------------------------ recording
    def start(self) -> dict:
        t = self.now()
        self.cur = Note(time.strftime("%Y%m%d-%H%M%S", time.localtime(t)), t)
        self.last_remind = self.line_from = t
        self.votes, self.who_ids = [], {}
        self._save(self.cur)
        return self.event("recording", START_LINE)

    def vote(self, people: dict, ts: float | None = None) -> None:
        """One analysed frame's `people` event. While someone is talking it says who, if the
        camera and the microphones can tell: a tracked person, a voice from where nobody is in
        view (by side), or "you" (a voice no one in view owns). A lone face on a mono microphone
        is no evidence either way and is not counted."""
        if self.cur is None or not people.get("voice"):
            return
        ts = self.now() if ts is None else ts
        if people.get("speaker") is not None and people.get("evidence") != "alone":
            who: object = ("track", people["speaker"])
        elif people.get("offscreen") is not None:
            who = ("off", round(people["offscreen"] / 60))
        elif people.get("evidence") == "alone":
            return
        else:
            who = "you"
        self.votes.append((ts, who))
        del self.votes[:-4000]

    def label(self, start: float, end: float) -> str | None:
        """Who spoke in [start, end]: the most common voice, if it holds at least half."""
        got = [w for t, w in self.votes if start <= t <= end]
        if not got:
            return None
        top = max(set(got), key=got.count)
        if got.count(top) * 2 < len(got):
            return None
        if top == "you":
            return "you"
        n = self.who_ids.setdefault(top, len(self.who_ids) + 1)
        return f"person {n}"

    def hear(self, text: str, end: float | None = None) -> dict | None:
        if self.cur is None or not text.strip():
            return None
        end = self.now() if end is None else end
        line = {"t": round(self.now() - self.cur.started, 1), "text": text.strip()}
        if who := self.label(self.line_from, end):
            line["who"] = who
        self.line_from = end
        self.votes = [v for v in self.votes if v[0] > end]
        self.cur.lines.append(line)
        self._save(self.cur)
        return self.event("recording")

    def reminder(self) -> str | None:
        """The line to say now, if the recording has gone a while without one."""
        if self.cur is None:
            return None
        t = self.now()
        if t - self.cur.started > MAX_S:
            return "limit"
        if t - self.last_remind >= REMIND_S:
            self.last_remind = t
            return REMIND_LINE
        return None

    def stop(self) -> tuple[dict, Note | None]:
        """Ends the recording. The returned note still needs `summarize`."""
        note, self.cur = self.cur, None
        if note is None:
            return self.event("idle"), None
        note.ended = self.now()
        if not note.lines:
            self._path(note.id).unlink(missing_ok=True)
            return self.event("idle", EMPTY_LINE), None
        self._save(note)
        return self.event("summarizing", STOP_LINE, id=note.id), note

    # ------------------------------------------------------------ summary
    def summarize(self, note: Note, language) -> dict:
        """Title and bullets from the configured model; the transcript stays either way."""
        text = note.text()
        try:
            pieces = [text[i : i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
            system = SUMMARY_SYSTEM + (LABELS if any(x.get("who") for x in note.lines) else "")
            parts = [language.complete(system, p, max_tokens=400).strip() for p in pieces]
            out = (
                parts[0]
                if len(parts) == 1
                else language.complete(MERGE_SYSTEM, "\n\n".join(parts), max_tokens=500).strip()
            )
        except Exception:
            out = ""
        if out:
            head, _, rest = out.partition("\n")
            note.title, note.summary = head.strip(" #*-"), rest.strip()
        else:  # no model reachable: keep the first words so the note can still be found
            note.title = " ".join(note.lines[0]["text"].split()[:6])
        self._save(note)
        say = f"{READY_LINE} {note.title}." if note.title else READY_LINE
        return {
            "type": "notes",
            "state": "ready",
            "id": note.id,
            "title": note.title,
            "summary": note.summary,
            "say": say,
        }

    # ------------------------------------------------------------ later
    def all(self) -> list[Note]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            try:
                d = json.loads(p.read_text())
                out.append(
                    Note(
                        d["id"],
                        d["started"],
                        d.get("ended"),
                        d.get("lines", []),
                        d.get("title", ""),
                        d.get("summary", ""),
                    )
                )
            except (OSError, ValueError, KeyError):
                continue
        return out

    def find(self, query: str, limit: int = 3) -> dict:
        """Summaries of the newest notes plus the transcript lines that share words with the
        question, so "what did the doctor say about the dose" lands on the dose line."""
        words = {w for w in re.findall(r"[a-z0-9']+", query.lower()) if w not in _STOPWORDS}
        notes = [n for n in self.all() if n.ended][-limit:][::-1]
        out, budget = [], FIND_CHARS
        for n in notes:
            hits = []
            for i, line in enumerate(n.lines):
                low = set(re.findall(r"[a-z0-9']+", line["text"].lower()))
                if words & low:
                    hits.append(" ".join(_say(x) for x in n.lines[max(0, i - 1) : i + 2]))
            hits = list(dict.fromkeys(hits))
            keep = []
            for h in hits:
                if budget - len(h) < 0:
                    break
                budget -= len(h)
                keep.append(h)
            full = n.text()
            if not keep and len(full) <= min(budget, SHORT_CHARS):
                keep, budget = [full], budget - len(full)  # the words differ: give it all
            out.append(
                {
                    "when": time.strftime("%b %d %H:%M", time.localtime(n.started)),
                    "minutes_ago": round((self.now() - n.started) / 60),
                    "title": n.title,
                    "summary": n.summary,
                    "matching_lines": keep,
                }
            )
        return {"notes": out} if out else {"notes": [], "message": "no saved notes"}

    def check(self, answer: str) -> list[str]:
        """What in `answer` no saved note backs (see `unsupported`)."""
        words = " ".join(n.text() for n in self.all() if n.ended)
        return unsupported(answer, words)

    def fallback(self, question: str) -> str:
        """What to say when an answer is not backed: the matching line itself, or nothing found."""
        for n in self.find(question)["notes"]:
            if n["matching_lines"]:
                return f"Your notes say: {n['matching_lines'][0]}"
        return "I can't find that in your notes."

    def delete(self, note_id: str) -> bool:
        if not re.fullmatch(r"\d{8}-\d{6}", note_id):
            return False
        p = self._path(note_id)
        if not p.exists():
            return False
        p.unlink()
        return True

    # ------------------------------------------------------------ disk
    def _path(self, note_id: str) -> Path:
        return self.root / f"{note_id}.json"

    def _save(self, note: Note) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path(note.id).with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(note.to_json(), f)
        os.replace(tmp, self._path(note.id))
