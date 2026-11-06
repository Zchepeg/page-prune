import json

import pytest

joblib = pytest.importorskip('joblib')
pytest.importorskip('sklearn')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from jev.lightweight_backend import TfidfBackend
from jev.questions import AD_TREATMENT, PROBE_AD_TREATMENT


def test_lightweight_backend_label_mapping_and_schema_guard(tmp_path):
    texts = [json.dumps({'schema_version': 'candidate-features-v2', 'text': x}) for x in ('article normal', 'article navigation', 'sponsored advertisement', 'paid advertisement')]
    vectorizer = TfidfVectorizer()
    classifier = LogisticRegression(C=10).fit(vectorizer.fit_transform(texts), [0, 0, 1, 1])
    joblib.dump({'vectorizer': vectorizer, 'classifier': classifier, 'metadata': {'temperature': 1, 'state_schema': 'candidate-features-v2'}}, tmp_path/'model.joblib')
    backend = TfidfBackend(tmp_path)
    ad = backend.decide(texts[-1], AD_TREATMENT).probabilities
    normal = backend.decide(texts[0], AD_TREATMENT).probabilities
    assert ad['ad'] > normal['ad']
    assert abs(sum(ad.values())-1) < 1e-10
    assert backend.runtime_status()['gpu_required'] is False
    with pytest.raises(ValueError, match='schema'):
        backend.decide('{}', AD_TREATMENT)
    with pytest.raises(ValueError, match='candidate'):
        backend.decide(texts[0], PROBE_AD_TREATMENT)
