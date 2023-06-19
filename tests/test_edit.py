from aries.util.edit import levenshtein_distance, basic_token_align, find_overlapping_substrings

def test_basic_token_align():
    seq1 = ['this', 'is', 'my', 'sentence']
    seq2 = ['this', 'is', 'my', 'sentence']
    d, align = basic_token_align(seq1, seq2)
    assert d == 0
    assert align == [0, 1, 2, 3]


    seq2 = ['t', 'h', 'i', 's', 'i', 's', 'm', 'y', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e']
    d, align = basic_token_align(seq1, seq2)
    assert d == 0
    assert align == [0]*4 + [1]*2 + [2]*2 + [3]*8

    seq2 = ['thisi', 's', 'mys', 'entence']
    d, align = basic_token_align(seq1, seq2)
    assert d == 2
    assert align == [0, 1, 2, 3]

    seq2 = ['this', '_is', '_my', '_sentence']
    d, align = basic_token_align(seq1, seq2)
    assert d == 3
    assert align == [0, 1, 2, 3]

    seq2 = ['this', 'is', 'my']
    try:
        d, align = basic_token_align(seq1, seq2)
        assert False, "Expected error since characters didn't match"
    except ValueError:
        pass

    seq2 = ['[this]', 'this', 'is', '[smy]', 'my', 'sentence', '[e]']
    d, align = basic_token_align(seq1, seq2, seq2_ignored_ids=[0,3,6])
    assert d == 0
    assert align == [None, 0, 1, None, 2, 3, None]

def test_levenshtein():
    assert levenshtein_distance('', '') == 0
    assert levenshtein_distance('', 'text') == 4
    assert levenshtein_distance('text', '') == 4
    assert levenshtein_distance('text', 'text') == 0
    assert levenshtein_distance('text', 'textb') == 1
    assert levenshtein_distance('textb', 'text') == 1
    assert levenshtein_distance('texta', 'textb') == 1
    assert levenshtein_distance('abba', 'acca') == 2

def test_find_overlapping_substrings():
    assert find_overlapping_substrings('', '', min_length=1) == []
    assert find_overlapping_substrings('', 'text', min_length=1) == []
    assert find_overlapping_substrings('text', '', min_length=1) == []
    assert find_overlapping_substrings('text', 'text', min_length=1) == [((0, 4), (0, 4))]
    assert find_overlapping_substrings('text', 'text', min_length=4) == [((0, 4), (0, 4))]
    assert find_overlapping_substrings('text', 'text', min_length=5) == []

    assert find_overlapping_substrings('atext', 'text', min_length=2) == [((1, 5), (0, 4))]
    assert find_overlapping_substrings('texta', 'text', min_length=2) == [((0, 4), (0, 4))]
    assert find_overlapping_substrings('text', 'atext', min_length=2) == [((0, 4), (1, 5))]
    assert find_overlapping_substrings('text', 'texta', min_length=2) == [((0, 4), (0, 4))]
    assert find_overlapping_substrings('btext', 'atext', min_length=2) == [((1, 5), (1, 5))]

    assert sorted(find_overlapping_substrings('the man and the cat', 'the cat and the man', min_length=4)) == [((0, 4), (0, 4)), ((0, 7), (12, 19)), ((7, 16), (7, 16)), ((12, 19), (0, 7))]

