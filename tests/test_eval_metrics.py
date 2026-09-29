from metrics import hit_rate_at_k, mean, percentile, reciprocal_rank


def test_hit_rate_true_when_any_golden_page_retrieved():
    assert hit_rate_at_k([1, 5, 12, 20], [12]) is True


def test_hit_rate_false_when_no_overlap():
    assert hit_rate_at_k([1, 5, 20], [12]) is False


def test_hit_rate_false_on_empty_retrieval():
    assert hit_rate_at_k([], [12]) is False


def test_reciprocal_rank_of_first_match():
    assert reciprocal_rank([1, 5, 12, 20], [12]) == 1 / 3


def test_reciprocal_rank_when_top_result_matches():
    assert reciprocal_rank([12, 1, 5], [12]) == 1.0


def test_reciprocal_rank_zero_when_not_found():
    assert reciprocal_rank([1, 5, 20], [12]) == 0.0


def test_reciprocal_rank_matches_any_golden_page():
    assert reciprocal_rank([1, 5, 20], [12, 20]) == 1 / 3


def test_percentile_empty_is_zero():
    assert percentile([], 50) == 0.0


def test_percentile_p50_of_sorted_values():
    assert percentile([10, 20, 30, 40, 50], 50) == 30


def test_percentile_p95_of_sorted_values():
    assert percentile(list(range(1, 101)), 95) == 95


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0


def test_mean_basic():
    assert mean([1.0, 2.0, 3.0]) == 2.0
