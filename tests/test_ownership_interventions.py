from scripts.compare_human_baseline import alter_ownership


def test_unknown_and_known_empty_only_change_ownership_fields():
    source = {'resource_urls': ['https://cdn.test/a'], 'direct_resource_urls': ['https://cdn.test/a'], 'resource_ownership_known': True, 'text': 'normal', 'detector_evidence': ['marker']}
    unknown = alter_ownership(source, 'unknown')
    empty = alter_ownership(source, 'known_empty')
    assert unknown['direct_resource_urls'] == empty['direct_resource_urls'] == []
    assert unknown['resource_ownership_known'] is False
    assert empty['resource_ownership_known'] is True
    assert unknown['resource_urls'] == empty['resource_urls'] == source['resource_urls']
    assert unknown['detector_evidence'] == source['detector_evidence']
    assert source['direct_resource_urls'] == ['https://cdn.test/a']
    omitted = alter_ownership(source, 'omitted')
    assert 'direct_resource_urls' not in omitted and 'resource_ownership_known' not in omitted
    assert omitted['text'] == source['text']


def test_shuffle_copies_only_donor_ownership_without_mutation():
    source = {'text': 'recipient', 'resource_urls': ['recipient'], 'direct_resource_urls': [], 'resource_ownership_known': False}
    donor = {'direct_resource_urls': ['donor'], 'resource_ownership_known': True}
    result = alter_ownership(source, 'shuffle', donor)
    result['direct_resource_urls'].append('extra')
    assert donor['direct_resource_urls'] == ['donor']
    assert result['text'] == source['text'] and result['resource_urls'] == source['resource_urls']
