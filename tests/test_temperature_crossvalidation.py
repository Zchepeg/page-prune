import math

from scripts.crossvalidate_temperature import crossvalidate, loss


def test_temperature_loss_is_finite_for_extreme_logits():
    assert math.isfinite(loss(10000, 0, .05))
    assert loss(10000, 1, .05) == 0


def test_domain_crossvalidation_covers_all_rows_and_is_reproducible():
    rows = [{'domain': f'domain-{domain}', 'target': label, 'logit': 5} for domain in range(10) for label in (0, 1)]
    result = crossvalidate(rows, 42)
    assert result == crossvalidate(rows, 42)
    assert sum(fold['heldout_examples'] for fold in result['folds']) == len(rows)
    assert all(fold['fit_examples'] == 16 and fold['heldout_domains'] == 2 for fold in result['folds'])
    assert result['out_of_fold']['nll'] < result['baseline']['nll']
