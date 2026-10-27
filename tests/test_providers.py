from metasight.config import Settings
from metasight.providers.language import TemplateLanguage, make_language


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
