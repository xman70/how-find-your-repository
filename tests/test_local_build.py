import json

from aidetect.build_dataset import build_local
from aidetect.data import load_jsonl

HUMAN_EL = ("Χθες το απόγευμα πήγαμε με τη γιαγιά στη λαϊκή, όπως κάθε Πέμπτη. Ο κυρ-Γιάννης με τις ντομάτες είχε "
            "πάλι καβγά με τον διπλανό του για το ποιος έπιασε πρώτος τη γωνία. Αγοράσαμε φασολάκια, λίγα μήλα και "
            "ένα καρπούζι που μάλλον ήταν άγουρο. Στον γυρισμό έπιασε ψιχάλα και τρέχαμε σαν τρελές. Η γιαγιά "
            "γελούσε και έλεγε ότι στα νιάτα της δεν φοβόταν ούτε το χαλάζι. Το βράδυ φάγαμε φασολάκια λαδερά.")
AI_EL = ("Η κλιματική αλλαγή αποτελεί μία από τις σημαντικότερες προκλήσεις της εποχής μας. Επιπλέον, επηρεάζει "
         "τα οικοσυστήματα, την οικονομία και την καθημερινή ζωή των πολιτών. Επομένως, είναι απαραίτητο να "
         "ληφθούν άμεσα μέτρα σε τοπικό και διεθνές επίπεδο. Ταυτόχρονα, η επένδυση σε ανανεώσιμες πηγές "
         "ενέργειας μπορεί να μειώσει σημαντικά τις εκπομπές. Συμπερασματικά, απαιτείται μια ολοκληρωμένη "
         "και συντονισμένη προσέγγιση για ένα βιώσιμο μέλλον.")


def _distinct_doc(seed: int) -> str:
    import random

    rng = random.Random(seed)
    words = (HUMAN_EL + " " + AI_EL).replace(".", "").replace(",", "").split()
    sents = [" ".join(rng.choice(words) for _ in range(12)).capitalize() + "." for _ in range(6)]
    return " ".join(sents)


def test_build_local_greek_without_features(tmp_path):
    corpus = tmp_path / "greek.jsonl"
    rows = []
    for i in range(24):
        rows.append({"text": _distinct_doc(2 * i), "label": 0, "domain": "blog", "group_id": f"t{i}"})
        rows.append({"text": _distinct_doc(2 * i + 1), "label": 1, "domain": "blog", "generator": "some-llm",
                     "generator_family": "x", "group_id": f"t{i}"})
    rows.append({"text": HUMAN_EL, "label": 0, "domain": "essay", "eval_set": "greek_students"})
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    card = build_local("el", [str(corpus)], out_dir=tmp_path / "out", skip_features=True)
    assert card["language"] == "el"
    train = load_jsonl(tmp_path / "out" / "train.jsonl")
    test = load_jsonl(tmp_path / "out" / "test.jsonl")
    assert train and test and all(s["language"] == "el" for s in train)
    assert not ({s["group_id"] for s in train} & {s["group_id"] for s in test})
    assert load_jsonl(tmp_path / "out" / "fp_audit.jsonl")[0]["eval_set"] == "greek_students"


def test_build_local_removes_near_duplicates(tmp_path):
    corpus = tmp_path / "dups.jsonl"
    rows = [{"text": HUMAN_EL + f" Σημείωση {i}.", "label": i % 2, "domain": "blog", "group_id": f"d{i}"}
            for i in range(30)]
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    card = build_local("el", [str(corpus)], out_dir=tmp_path / "o", skip_features=True)
    removed = card["leakage_controls"]["deduplication"]["near_duplicates_removed"]
    assert removed["test"] + removed["calib"] > 0
