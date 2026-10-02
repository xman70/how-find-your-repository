from aidetect.adversarial import TRANSFORMS, casualize, merge_split, sentence_reorder, typo_noise
from aidetect.data import (build_hybrids, clean_text, count_words, group_split, hybrid_class_for, length_augment, make_sample,
                           remove_near_duplicates, strip_label_lines, truncate_words)

LONG = " ".join(f"Sentence number {i} talks about topic {i % 7} in some detail and then it ends." for i in range(40))


def _samples():
    out = []
    for g in range(30):
        for lab in (0, 1):
            t = f"Group {g} label {lab}. " + LONG.replace("topic", f"topic{g}{lab}")
            out.append(make_sample(t, lab, source_dataset="test", domain="d" + str(g % 3), group_id=f"g{g}",
                                   generator="none" if lab == 0 else "gen", generator_family="none" if lab == 0 else "fam"))
    return out


def test_group_split_has_no_group_leakage():
    sp = group_split(_samples(), 0.2, 0.2, seed=3, force_test_groups={"g1"})
    groups = {k: {s["group_id"] for s in v} for k, v in sp.items()}
    assert not (groups["train"] & groups["test"]) and not (groups["train"] & groups["calib"])
    assert not (groups["calib"] & groups["test"])
    assert "g1" in groups["test"]


def test_near_duplicates_removed():
    a = make_sample(LONG, 0, source_dataset="t", domain="x", group_id="a")
    b = make_sample(LONG + " One extra sentence.", 1, source_dataset="t", domain="x", group_id="b")
    c = make_sample("Completely different text about cooking pasta with tomatoes and basil in summer. " * 5, 1,
                    source_dataset="t", domain="x", group_id="c")
    splits = {"train": [a], "calib": [], "test": [b, c]}
    rep = remove_near_duplicates(splits, 0.5)
    assert rep["near_duplicates_removed"]["test"] == 1
    assert [s["group_id"] for s in splits["test"]] == ["c"]


def test_hybrids_have_consistent_segments():
    hy = build_hybrids(_samples(), 10, seed=1)
    assert hy
    for h in hy:
        segs = h["segments"]
        assert segs[0][0] == 0 and segs[-1][1] == len(h["text"])
        assert {lab for _, _, lab in segs} == {0, 1}
        assert 0 < h["ai_fraction"] < 1
        assert h["label"] == int(h["ai_fraction"] >= 0.5)
        assert h["author_type"] == "hybrid_spliced"


def test_cleaning_and_labels():
    assert strip_label_lines("Prompt: write an essay\nEssay:\nThe text begins.") == "The text begins."
    assert clean_text("Write an abstract. The model output here is long.", prompt="Write an abstract.").startswith("The model")
    assert hybrid_class_for(0.05) == 0 and hybrid_class_for(0.3) == 1 and hybrid_class_for(0.6) == 2 and hybrid_class_for(0.9) == 3


def test_truncation_and_augmentation():
    t = truncate_words(LONG, 50)
    assert 50 <= count_words(t) < 65 and t.endswith(".")
    aug = length_augment([make_sample(LONG, 1, source_dataset="t", domain="x")], seed=0)
    assert aug and aug[0]["n_words"] < len(LONG.split()) and aug[0]["augmented"] == "truncation"


def test_adversarial_transforms_are_deterministic():
    text = ("It is important that we do not ignore this; the results are clear. We are sure. "
            "The second experiment was large, which helped, and the team could replicate everything across sites.")
    for name, fn in TRANSFORMS.items():
        if name == "sim_wordnet_synonyms":
            continue
        assert fn(text, seed=4) == fn(text, seed=4)
    assert typo_noise(text, rate=1.0, seed=1) != text
    assert "don't" in casualize(text, seed=0) or "do not" in casualize(text, seed=0)
    assert isinstance(sentence_reorder(text, 1), str) and isinstance(merge_split(text, 1), str)
