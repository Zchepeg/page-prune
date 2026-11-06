import hashlib
import json

import pytest

from scripts.freeze_fresh_blind import validate_study


def study(tmp_path, exclusions=()):
    pool = tmp_path/'pool.jsonl'
    pool.write_text(json.dumps({'id':'a', 'domain':'new.test'})+'\n', encoding='utf-8')
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'models':{}, 'code_sha256':{}, 'exclusion_domains':list(exclusions)}), encoding='utf-8')
    manifest = {'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(), 'pool_sha256':hashlib.sha256(pool.read_bytes()).hexdigest(), 'screenshots':{}}
    (tmp_path/'capture-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_blind_integrity_rejects_modified_pool(tmp_path):
    study(tmp_path)
    assert len(validate_study(tmp_path)[1]) == 1
    (tmp_path/'pool.jsonl').write_text('{}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='hash changed'):
        validate_study(tmp_path)


def test_blind_integrity_rejects_prior_domain(tmp_path):
    study(tmp_path, ['new.test'])
    with pytest.raises(ValueError, match='domain overlap'):
        validate_study(tmp_path)
