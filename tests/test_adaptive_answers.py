from replan.answers import (
    extract_answer,
    normalize_numeric_answer,
    numeric_answer_is_correct,
)


def test_extracts_last_balanced_boxed_answer() -> None:
    text = r"draft \boxed{12}; final \boxed{\frac{1}{2}}"
    assert extract_answer(text) == r"\frac{1}{2}"


def test_numeric_normalization() -> None:
    assert normalize_numeric_answer("27.0") == "27"
    assert numeric_answer_is_correct("2,009", "2009")
    assert not numeric_answer_is_correct(None, "3")
