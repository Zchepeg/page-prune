import json

from scripts.rebuild_ownership_state import rebuild


def test_rebuild_preserves_supervision_and_unknown_ownership(tmp_path):
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    record = {'id': 'a', 'source': 'example.test', 'state': {'detector_evidence': ['original_evidence']}, 'question': {'label': 1}, 'provenance': {'label_source': 'human_independent'}}
    (snapshot/'train.jsonl').write_text(json.dumps(record)+'\n', encoding='utf-8')
    raw = tmp_path/'raw.jsonl'
    candidate = {'page': {'top_level_domain': 'example.test'}, 'resource_urls': ['https://cdn.test/image']}
    raw.write_text(json.dumps({'id': 'a', 'candidate': candidate})+'\n', encoding='utf-8')
    output = tmp_path/'rebuilt'
    report = rebuild(snapshot, raw, output)
    result = json.loads((output/'train.jsonl').read_text(encoding='utf-8'))
    assert result['id'] == record['id'] and result['source'] == record['source']
    assert result['question'] == record['question'] and result['provenance'] == record['provenance']
    assert result['state']['detector_evidence'] == ['original_evidence']
    assert result['state']['resource_ownership_known'] is False
    assert report['splits']['train']['ownership'] == {'unknown': 1}
