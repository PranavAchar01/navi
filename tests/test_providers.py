from metasight.config import Settings
from metasight.providers.finder import parse_sam_boxes
from metasight.providers.language import TemplateLanguage, make_language

SAM_LINE = (
    "<0f>0<|box;x1=100;y1=50;x2=199;y2=149;w=400;h=300;c=0.91|><|mask;x=0;y=0;c=0.91;data=10,20,~abc|>,"
    "2<|box;x1=0;y1=0;x2=9;y2=9;w=400;h=300;c=0.30|><|mask;x=0;y=0;c=0.30;data=1,1,~a|>"
)


def test_sam_box_parsing_scales_to_our_frame_and_reads_confidence():
    dets = parse_sam_boxes(SAM_LINE, "keys", 200, 150)  # our frame is half the size SAM saw
    assert len(dets) == 2
    assert dets[0].label == "keys" and dets[0].score == 0.91
    assert dets[0].box == (50.0, 25.0, 100.0, 75.0)


def test_sam_parsing_without_confidence_and_with_no_matches():
    line = "<0f>0<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10|><|mask;x=0;y=0;data=1,1,~a|>"
    assert parse_sam_boxes(line, "x", 10, 10)[0].score == 1.0
    assert parse_sam_boxes("", "x", 10, 10) == []


def test_template_provider_is_the_no_model_fallback():
    s = Settings(language="template")
    assert isinstance(make_language(s), TemplateLanguage)


def test_meta_language_without_a_key_says_so(monkeypatch):
    import metasight.providers.language as lang

    monkeypatch.setattr(lang, "secret", lambda *names: None)
    try:
        make_language(Settings(language="meta"))
    except RuntimeError as e:
        assert "META_API_KEY" in str(e)
    else:
        raise AssertionError("expected a RuntimeError")


def test_meta_language_points_at_the_meta_endpoint_and_standard_mode_model(monkeypatch):
    import metasight.providers.language as lang

    monkeypatch.setattr(lang, "secret", lambda *names: "test-key")
    m = make_language(Settings(language="meta"))
    assert m.name == "muse-spark" and m.model == "muse-spark-1.3"
    assert str(m._client.base_url).startswith("https://api.meta.ai/v1")
