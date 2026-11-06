import pytest

from scripts.compare_domain_bootstrap import paired_recall


def test_identical_predictions_have_zero_paired_interval():
    rows = [{'id': str(i), 'label': 0, 'argmax_ad': bool(i % 2), 'verdict': 'allow', 'action': 'allow'} for i in range(6)]
    report = paired_recall(rows, rows, {str(i): f'domain{i//2}' for i in range(6)}, 100)
    assert all(value['delta_recall'] == 0 and value['domain_bootstrap_95_percentile'] == [0, 0] for value in report.values())


def test_paired_interval_preserves_direction_and_rejects_missing_rows():
    rows = [{'id': str(i), 'label': 0, 'argmax_ad': False, 'verdict': 'allow', 'action': 'allow'} for i in range(4)]
    improved = [{**row, 'argmax_ad': True} for row in rows]
    domains = {str(i): f'domain{i}' for i in range(4)}
    assert paired_recall(rows, improved, domains, 100)['argmax']['domain_bootstrap_95_percentile'] == [1, 1]
    with pytest.raises(ValueError, match='IDs'):
        paired_recall(rows, improved[:-1], domains, 100)
