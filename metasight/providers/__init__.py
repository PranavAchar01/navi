"""Model providers behind two small interfaces. Switch them in metasight.toml, one line each."""

from .finder import Detection, Finder, make_finder
from .language import LanguageModel, make_language

__all__ = ["Detection", "Finder", "LanguageModel", "make_finder", "make_language"]
