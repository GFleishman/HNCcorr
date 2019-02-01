import pytest


@pytest.fixture
def neg_seeds():
    return {(5,), (6,)}


@pytest.fixture
def S(pos_seeds, neg_seeds):
    from hnccorr.seeds import Seeds

    return Seeds((5,), pos_seeds, neg_seeds)


def test_center_seed(S):
    assert S.center_seed == (5,)


def test_positive_seeds(S, pos_seeds):
    assert S.positive_seeds == pos_seeds


def test_negative_seeds(S, neg_seeds):
    assert S.negative_seeds == neg_seeds