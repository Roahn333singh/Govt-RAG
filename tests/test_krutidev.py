import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "krutidev_to_unicode.py"
_spec = importlib.util.spec_from_file_location("krutidev_to_unicode", _PATH)
kru = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kru)


def test_known_sentence():
    assert kru.kru2uni("esjk uke usgy gSA") == "मेरा नाम नेहल है।"


def test_common_syllables():
    assert kru.kru2uni("dks") == "को"
    assert "के" in kru.kru2uni("ds fy,")


def test_detector_ignores_real_hindi_and_english():
    assert kru.looks_like_krutidev("vihy çkf/kdkjh fuEufyf[kr ckrksa ij fopkj djds fu;e")
    assert not kru.looks_like_krutidev("सामान्य भविष्य निधि के नियम क्या हैं और कैसे लागू होते हैं?")
    assert not kru.looks_like_krutidev("What is the official URL for the Pragati portal of Uttar Pradesh?")
