import json
import stat

from metasight.notes import Notes, asks_notes, notes_command
from metasight.pins import PinStore
from metasight.providers.tools import Tools


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


class Lang:
    name = "fake"

    def __init__(self, out="Doctor visit\n- Take 20 mg once a day\n- Come back in two weeks"):
        self.out, self.calls = out, []

    def complete(self, system, user, max_tokens=120):
        self.calls.append(user)
        return self.out


def talk(n, lines=("The dose is 20 mg once a day.", "Come back in two weeks.")):
    n.start()
    for line in lines:
        n.hear(line)
    return n.stop()


def test_commands():
    assert notes_command("Please take notes") == "start"
    assert notes_command("start recording") == "start"
    assert notes_command("ok stop recording") == "stop"
    assert notes_command("where is my water bottle") is None
    assert asks_notes("what did the doctor say about the dose")
    assert not asks_notes("where are my keys")


def test_start_says_recording_and_stop_summarises(tmp_path):
    clock = Clock()
    n = Notes(tmp_path, now=clock)
    ev = n.start()
    assert ev["state"] == "recording" and "Recording" in ev["say"] and n.recording
    n.hear("The dose is 20 mg once a day.")
    clock.t += 30
    n.hear("Come back in two weeks.")
    ev, note = n.stop()
    assert ev["state"] == "summarizing" and not n.recording
    done = n.summarize(note, Lang())
    assert done["state"] == "ready" and done["title"] == "Doctor visit"
    saved = json.loads((tmp_path / f"{note.id}.json").read_text())
    assert saved["lines"][1]["t"] == 30.0 and "20 mg" in saved["summary"]


def test_files_are_owner_only(tmp_path):
    n = Notes(tmp_path, now=Clock())
    n.start()
    n.hear("hello there")
    f = next(tmp_path.glob("*.json"))
    assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_empty_recording_is_not_kept(tmp_path):
    n = Notes(tmp_path, now=Clock())
    n.start()
    ev, note = n.stop()
    assert note is None and "nothing to save" in ev["say"]
    assert list(tmp_path.glob("*.json")) == []


def test_reminder_every_five_minutes_and_limit(tmp_path):
    clock = Clock()
    n = Notes(tmp_path, now=clock)
    n.start()
    assert n.reminder() is None
    clock.t += 301
    assert n.reminder() == "Still recording."
    assert n.reminder() is None
    clock.t += 3 * 3600
    assert n.reminder() == "limit"


def test_no_model_keeps_transcript_and_a_title(tmp_path):
    n = Notes(tmp_path, now=Clock())
    _, note = talk(n)
    done = n.summarize(note, Lang(out=""))
    assert done["title"] == "The dose is 20 mg once"
    assert note.summary == ""


def test_long_transcript_is_summarised_in_pieces(tmp_path):
    n = Notes(tmp_path, now=Clock())
    n.start()
    for _ in range(40):
        n.hear("x" * 700)
    _, note = n.stop()
    lang = Lang()
    n.summarize(note, lang)
    assert len(lang.calls) == 4  # three pieces, then the merge


def test_find_returns_the_dose_line_for_the_dose_question(tmp_path):
    n = Notes(tmp_path, now=Clock())
    _, note = talk(n, ("Hello, sit down.", "Take the dose of 20 mg daily.", "See you soon."))
    n.summarize(note, Lang())
    r = n.find("what did the doctor say about the dose")
    assert r["notes"][0]["title"] == "Doctor visit"
    assert any("20 mg" in h for h in r["notes"][0]["matching_lines"])


def test_find_with_no_notes(tmp_path):
    assert Notes(tmp_path).find("anything")["notes"] == []


def test_tool_reads_notes_and_delete_checks_the_id(tmp_path):
    n = Notes(tmp_path, now=Clock())
    _, note = talk(n)
    n.summarize(note, Lang())
    t = Tools(PinStore(), None, lambda: None, notes=n)
    assert json.loads(t.call("search_notes", {"query": "dose"}))["notes"]
    assert not n.delete("../../etc/passwd")
    assert n.delete(note.id) and n.find("dose")["notes"] == []


def test_server_routes_finals_to_notes_not_questions(tmp_path):
    import asyncio
    from types import SimpleNamespace as NS

    from metasight.providers.transcribe import Heard
    from metasight.server import Server

    sent, asked = [], []

    class Hub:
        async def json(self, ev, *_):
            sent.append(ev)

    class Voice:
        async def on_heard(self, h):
            asked.append(h.text)

    srv = NS(
        notes=Notes(tmp_path, now=Clock()),
        hub=Hub(),
        voice=Voice(),
        language=Lang(),
        llm_pool=None,
        _notes_task=None,
    )
    srv.start_notes = lambda: Server.start_notes(srv)
    srv._notes_reminders = lambda: Server._notes_reminders(srv)

    async def run():
        srv.loop = asyncio.get_running_loop()
        srv.stop_notes = lambda: Server.stop_notes(srv)
        for text in ("take notes", "Where is the dose?", "The dose is 20 mg.", "stop recording"):
            await Server.heard(srv, Heard(text, True, 1, None))
        await asyncio.sleep(0)

    asyncio.run(run())
    states = [e["state"] for e in sent if e["type"] == "notes"]
    assert states[0] == "recording" and "summarizing" in states and states[-1] == "ready"
    assert asked == []  # a question asked during a recording is part of the notes
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert [x["text"] for x in saved["lines"]] == ["Where is the dose?", "The dose is 20 mg."]


def test_find_gives_a_short_note_whole_when_no_word_matches(tmp_path):
    n = Notes(tmp_path, now=Clock())
    _, note = talk(n, ("Ten milligrams before bed.", "Call in two weeks."))
    n.summarize(note, Lang())
    got = n.find("what was the dose")["notes"][0]["matching_lines"]
    assert got == ["Ten milligrams before bed.\nCall in two weeks."]


def test_labels_you_and_people_from_votes(tmp_path):
    clock = Clock()
    n = Notes(tmp_path, now=clock)
    n.start()
    t0 = clock.t

    def say(who_event, secs, text):
        for i in range(secs * 10):
            n.vote({"voice": True, **who_event}, t0 + i / 10)
        n.hear(text, t0 + secs)

    say({"speaker": None, "evidence": "none"}, 2, "How is the knee?")
    t0 += 2
    say({"speaker": 7, "evidence": "lip+doa"}, 3, "Better, the dose helped.")
    t0 += 3
    say({"speaker": 9, "evidence": "doa"}, 2, "Same here.")
    t0 += 2
    say({"speaker": 7, "evidence": "lip"}, 2, "Take it daily.")
    got = [(x.get("who"), x["text"]) for x in n.cur.lines]
    assert [g[0] for g in got] == ["you", "person 1", "person 2", "person 1"]
    assert n.cur.text().splitlines()[1] == "person 1: Better, the dose helped."


def test_no_direction_evidence_leaves_lines_unlabelled(tmp_path):
    n = Notes(tmp_path, now=Clock())
    n.start()
    n.vote({"voice": True, "speaker": 3, "evidence": "alone"})  # a lone face on a mono mic
    n.hear("Hello.")
    n.vote({"voice": True, "speaker": 3, "evidence": "lip"}, n.now() + 1)
    n.vote({"voice": True, "speaker": None, "evidence": "none"}, n.now() + 1.1)
    n.vote({"voice": True, "speaker": 4, "evidence": "doa"}, n.now() + 1.2)
    n.hear("Split.", n.now() + 2)
    assert "who" not in n.cur.lines[0] and "who" not in n.cur.lines[1]


def test_offscreen_voice_gets_its_own_person(tmp_path):
    n = Notes(tmp_path, now=Clock())
    n.start()
    n.vote({"voice": True, "speaker": None, "offscreen": 70.0, "evidence": "doa"})
    n.hear("Over here.")
    assert n.cur.lines[0]["who"] == "person 1"


def test_unsupported_quotes_and_numbers():
    from metasight.notes import unsupported

    said = "Take ten milligrams an hour before bed. Call in two weeks, or on November 12th."
    assert unsupported('He said "an hour before bed" and 10 mg.', said) == []
    assert unsupported("Come back in 2 weeks.", said) == []
    assert unsupported('He said "twice a day".', said) == ["twice a day"]
    assert unsupported("Take 20 mg.", said) == ["20"]
    assert unsupported("It was 10.5 mg.", said) == ["10.5"]


def test_label_instruction_only_when_lines_are_labelled(tmp_path):
    class Spy(Lang):
        def complete(self, system, user, max_tokens=120):
            self.calls.append(system)
            return self.out

    n = Notes(tmp_path, now=Clock())
    _, note = talk(n)
    spy = Spy()
    n.summarize(note, spy)
    assert "person 1" not in spy.calls[0]
    note.lines[0]["who"] = "you"
    spy = Spy()
    n.summarize(note, spy)
    assert "person 1" in spy.calls[0]
