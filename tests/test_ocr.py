from geneval.scorers.ocr import char_accuracy, gibberish_words, normalize, substring_distance


def test_normalize():
    assert normalize("World’s  okayest\ndev!") == "WORLD'S OKAYEST DEV"


def test_substring_distance_finds_target_inside_longer_text():
    assert substring_distance("CAFE", "OPEN MORNING CAFE 7AM") == 0
    assert substring_distance("CAFE", "OPEN MORNING CXFX") == 2


def test_char_accuracy():
    assert char_accuracy("Morning Brew Cafe", "open - MORNING BREW CAFE - 7am") == 1.0
    assert 0.9 < char_accuracy("MORNING BREW CAFE", "MORNNG BREW CAFE") < 1.0
    assert char_accuracy("MORNING BREW CAFE", "") == 0.0


def test_char_accuracy_line_break_inside_phrase():
    # "MORNING BREW / CAFE" read as two lines joined without the space
    assert char_accuracy("MORNING BREW CAFE", "MORNING BREWCAFE") == 1.0


def test_gibberish_ignores_real_words_and_targets():
    text = "MORNING BREW CAFE OPEN DAILY PASTRIES Tadey Spoe TunckeogFhck"
    assert gibberish_words(text, ["MORNING BREW CAFE"]) == ["TADEY", "SPOE", "TUNCKEOGFHCK"]
    assert gibberish_words("WORLD'S OKAYEST DEVELOPER", ["WORLD'S OKAYEST DEVELOPER"]) == []
