import pytest

from evaluation.paired_statistics import exact_mcnemar, paired_statistics, wilson


def row(identifier, domain, label, prediction, action='allow'):
    return {'id': identifier, 'domain': domain, 'label': label, 'prediction': prediction, 'action': action}


def test_exact_test_and_wilson_known_values():
    assert exact_mcnemar(1, 0) == 1
    assert exact_mcnemar(0, 0) == 1
    assert exact_mcnemar(6, 0) == .03125
    assert wilson(27, 74) == pytest.approx([.264388, .478683], abs=.00001)


def test_paired_cluster_preserves_duplicate_domain_and_actions():
    left = [row('a', 'one', 0, 1), row('b', 'one', 1, 1), row('c', 'two', 0, 0, 'hide')]
    right = [row('a', 'one', 0, 0), row('b', 'one', 1, 0), row('c', 'two', 0, 0, 'hide')]
    result = paired_statistics(left, right, 200)
    metrics = result['metrics']
    assert result['domains'] == 2
    assert metrics['classification']['delta_recall'] == .5
    assert metrics['classification']['delta_recall_domain_bootstrap_95'] == [0, 1]
    assert metrics['classification']['ad_recall_mcnemar']['two_sided_exact_p'] == 1
    assert metrics['page_action']['delta_recall_domain_bootstrap_95'] == [0, 0]
    assert result['all_candidate_classification_mcnemar']['regressed'] == 1


def test_pair_validation_and_no_ad_resamples():
    left = [row('a', 'one', 0, 0), row('b', 'two', 1, 1)]
    result = paired_statistics(left, left, 100)
    assert 0 < result['metrics']['classification']['recall_valid_resamples'] < 100
    assert result['metrics']['classification']['delta_f1_domain_bootstrap_95'] == [0, 0]
    with pytest.raises(ValueError):
        paired_statistics(left, left[:1], 100)
    with pytest.raises(ValueError):
        paired_statistics(left, [row('a', 'wrong', 0, 0), left[1]], 100)
    with pytest.raises(ValueError):
        paired_statistics(left + left, left + left, 100)
