import pytest

from scripts.export_calibration_review import prepare


def samples():
    pool, annotations = [], {}
    for i in range(60):
        for label in ('ad', 'non_ad'):
            identifier = f'{i}-{label}'
            domain = f'site{i}.test'
            pool.append({'id': identifier, 'domain': domain, 'candidate': {'page': {'top_level_domain': domain}, 'tag': 'DIV'}})
            annotations[identifier] = {'label': label, 'annotation_mode': 'human_independent'}
    return pool, annotations


def test_export_partitions_are_isolated_and_do_not_follow_labels():
    pool, annotations = samples()
    partitions, report = prepare(pool, annotations, set())
    assert report['ready']
    domains = {split: {r['source'] for r in records} for split, records in partitions.items()}
    assert not domains['calibration'] & domains['validation']
    for annotation in annotations.values():
        annotation['label'] = 'non_ad' if annotation['label'] == 'ad' else 'ad'
    swapped, _ = prepare(pool, annotations, set())
    assert {split: {r['id'] for r in records} for split, records in partitions.items()} == {split: {r['id'] for r in records} for split, records in swapped.items()}


def test_export_rejects_assisted_and_incomplete_labels():
    pool, annotations = samples()
    del annotations[pool[0]['id']]
    annotations[pool[1]['id']]['annotation_mode'] = 'human_confirmed_machine_suggestion'
    _, report = prepare(pool, annotations, set())
    assert not report['ready']
    assert 'incomplete_human_review' in report['reasons']
    assert 'non_independent_annotation' in report['reasons']


def test_export_rejects_parent_domain_leakage():
    pool, annotations = samples()
    with pytest.raises(ValueError, match='domain overlaps'):
        prepare(pool, annotations, {'test'})
