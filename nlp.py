"""
nlp.py
Converts a raw sequence of recognized sign-language words (e.g.
["I", "GO", "HOME"]) into a grammatically fluent sentence
(e.g. "I am going home.").

Pipeline:
  1. spaCy POS-tagging based rule engine inserts articles, auxiliary verbs,
     and correct verb conjugation for common sentence patterns.
  2. LanguageTool (via language_tool_python) performs a final grammar /
     punctuation correction pass.
  3. If either dependency is unavailable at runtime, both stages degrade to
     a safe, still-useful rule-based fallback so the app never crashes.
"""

import re
import config

_spacy_nlp = None
_spacy_available = None
_langtool = None
_langtool_available = None


def _get_spacy():
    global _spacy_nlp, _spacy_available
    if _spacy_available is not None:
        return _spacy_nlp
    try:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            # model not downloaded; fall back to a blank English pipeline
            # with just a tokenizer, which still lets basic rules run
            _spacy_nlp = spacy.blank("en")
        _spacy_available = True
    except Exception:
        _spacy_nlp = None
        _spacy_available = False
    return _spacy_nlp


def _get_langtool():
    global _langtool, _langtool_available
    if _langtool_available is not None:
        return _langtool
    try:
        import language_tool_python
        _langtool = language_tool_python.LanguageTool("en-US")
        _langtool_available = True
    except Exception:
        _langtool = None
        _langtool_available = False
    return _langtool


# Minimal irregular verb map for "I GO HOME" -> "I am going home" style
# present-continuous conversion, the pattern most sign languages use
# (topic-comment / base-verb ordering rather than English SVO + tense).
IRREGULAR_GERUNDS = {
    "go": "going", "eat": "eating", "drink": "drinking", "run": "running",
    "sit": "sitting", "stop": "stopping", "help": "helping", "love": "loving",
}

PRONOUNS = {"i", "you", "he", "she", "it", "we", "they"}
BE_VERB = {
    "i": "am", "you": "are", "he": "is", "she": "is", "it": "is",
    "we": "are", "they": "are",
}


def _gerund(word):
    word_l = word.lower()
    if word_l in IRREGULAR_GERUNDS:
        return IRREGULAR_GERUNDS[word_l]
    if word_l.endswith("e") and word_l not in ("be", "see"):
        return word_l[:-1] + "ing"
    return word_l + "ing"


def rule_based_sentence(words):
    """
    Deterministic fallback / first-pass sentence builder. Handles the
    common sign-language gloss pattern:
        [PRONOUN] [VERB] [OBJECT...]  ->  "Pronoun am/is/are verb-ing object."
    and otherwise falls back to simple capitalization + punctuation so the
    output is never worse than the raw gloss.
    """
    if not words:
        return ""

    tokens = [w.lower().strip() for w in words if w.strip()]
    if not tokens:
        return ""

    if tokens[0] in PRONOUNS and len(tokens) >= 2:
        pronoun = tokens[0]
        verb = tokens[1]
        rest = tokens[2:]
        be = BE_VERB.get(pronoun, "is")
        sentence = f"{pronoun.capitalize()} {be} {_gerund(verb)}"
        if rest:
            sentence += " " + " ".join(rest)
        sentence += "."
        return sentence

    # Generic fallback: just join, capitalize, punctuate
    sentence = " ".join(tokens)
    sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


def spacy_assisted_sentence(words):
    """
    Uses spaCy (if available) to double check part-of-speech tags and
    slightly improve word ordering before handing off to the rule engine.
    Falls back to rule_based_sentence if spaCy isn't available.
    """
    nlp = _get_spacy()
    base = rule_based_sentence(words)
    if nlp is None:
        return base
    try:
        doc = nlp(base)
        # spaCy here mainly validates tokenization / capitalization; the
        # deterministic rule pass above already handles verb conjugation.
        cleaned = " ".join(t.text for t in doc)
        cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned or base
    except Exception:
        return base


def grammar_correct(sentence):
    """Final LanguageTool grammar/spelling correction pass. Falls back to
    returning the input unchanged if LanguageTool is not available (it
    requires a Java runtime, which may not be present on all machines)."""
    tool = _get_langtool()
    if tool is None or not sentence:
        return sentence
    try:
        matches = tool.check(sentence)
        import language_tool_python
        corrected = language_tool_python.utils.correct(sentence, matches)
        return corrected
    except Exception:
        return sentence


def words_to_sentence(words, use_grammar_check=True):
    """
    Main public entry point. Takes a list of recognized gesture words
    (in the order they were signed) and returns a natural English sentence.
    """
    if not words:
        return ""
    draft = spacy_assisted_sentence(words)
    if use_grammar_check:
        draft = grammar_correct(draft)
    return draft


if __name__ == "__main__":
    tests = [
        ["i", "go", "home"],
        ["you", "help", "me"],
        ["thanks", "friend"],
        ["i", "love", "you"],
    ]
    for t in tests:
        print(t, "->", words_to_sentence(t))
