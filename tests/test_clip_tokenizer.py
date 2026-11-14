from geneval.scorers.clip_tokenizer import ClipTokenizer


def test_matches_reference_clip_ids():
    tok = ClipTokenizer()
    # ids from the original CLIP tokenizer
    assert tok.encode("hello world!") == [3306, 1002, 256]
    ids = tok(["a photo of a cat"])
    assert ids.shape == (1, 77)
    assert ids[0, 0] == tok.sot and tok.eot in ids[0]


def test_truncates_long_prompts():
    tok = ClipTokenizer()
    ids = tok(["word " * 200])
    assert ids[0, -1] == tok.eot
