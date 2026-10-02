import pytest

from aidetect.preprocessing import (ModelMissingError, build_document, detect_language, get_nlp, is_heading,
                                    normalize_text, split_paragraphs, trim_incomplete_tail, word_tokens)


def test_normalize_removes_formatting_artifacts():
    raw = "He said “hello”  and—then left. Range 1990–2000 -- ok.\n\n* bullet item here."
    n = normalize_text(raw)
    assert '"hello"' in n
    assert "  " not in n
    assert " — " in n and "1990-2000" in n
    assert "* " not in n and "bullet item here." in n


def test_normalize_scrape_debris_and_hard_wraps():
    n = normalize_text("First step done.;\n, Second step , here.\n\n,,, Third.")
    assert ".;" not in n and ",," not in n and " ," not in n
    wrapped = ("This abstract line is wrapped at a fixed width so that\nit continues on the next line without any\n"
               "punctuation at the end of each line until the sentence\nfinally ends here.")
    assert "\n" not in normalize_text(wrapped)


def test_split_paragraphs_blank_and_single_newlines():
    assert len(split_paragraphs("a b c.\n\nd e f.")) == 2
    assert len(split_paragraphs("line one.\nline two.")) == 2


def test_language_detection():
    assert detect_language("The committee reviewed the proposal and decided that it was not ready.").code == "en"
    g = detect_language("Η κυβέρνηση ανακοίνωσε χθες νέα μέτρα για την οικονομία και την ανάπτυξη της χώρας.")
    assert g.code == "el" and g.supported
    f = detect_language("Le gouvernement a annoncé de nouvelles mesures pour les citoyens et la ville dans le pays.")
    assert f.code == "fr" and not f.supported
    assert detect_language("hi").code == "und"


def test_sentence_records_counts():
    doc = build_document("Furthermore, the report was written by the committee. They don't agree; however, "
                         "it's fine.\n\nIntroduction\n\nSecond paragraph here, with a comma.", "en")
    assert len(doc.sentences) == 3
    s0, s1, s2 = doc.sentences
    assert s0.opening == "transition" and s0.c("passive") == 1
    assert s1.c("contraction") == 2 and s1.c("semicolon") == 1
    assert s2.para == 1 and s2.para_initial  # heading paragraph was skipped
    assert any(t[0] == "contrastive" for t in s1.transitions)


def test_greek_document():
    doc = build_document("Η κυβέρνηση ανακοίνωσε νέα μέτρα. Ωστόσο, τα μέτρα εφαρμόστηκαν από την αρχή του μήνα.", "el")
    assert len(doc.sentences) == 2
    assert doc.sentences[1].opening == "transition"
    assert doc.sentences[1].c("passive") == 1


def test_heading_and_tail_helpers():
    assert is_heading("Introduction") and not is_heading("This is a sentence.")
    assert trim_incomplete_tail("Done here. Not finish") == "Done here."
    assert trim_incomplete_tail("Complete.") == "Complete."
    assert word_tokens("Don't STOP, Ωστόσο!") == ["don't", "stop", "ωστοσο"]


def test_unknown_spacy_model_fails_gracefully():
    with pytest.raises(ModelMissingError):
        get_nlp("en", "no_such_spacy_model_xyz")
