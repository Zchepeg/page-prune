from __future__ import annotations

import math
from pathlib import Path

from runtime.state_serialization import validate_state_schema
from .backend import BackendResult
from .questions import AD_TREATMENT
from .types import DecisionLevel


class TfidfBackend:
    def __init__(self, checkpoint):
        import joblib

        saved = joblib.load(Path(checkpoint)/'model.joblib')
        self.vectorizer = saved['vectorizer']
        self.classifier = saved['classifier']
        self.metadata = saved['metadata']
        self.temperature = float(self.metadata['temperature'])

    def decide(self, state, question=AD_TREATMENT):
        if question.name != AD_TREATMENT.name:
            raise ValueError('TF-IDF baseline supports candidate classification only')
        validate_state_schema(state, self.metadata['state_schema'])
        logit = float(self.classifier.decision_function(self.vectorizer.transform([state]))[0]) / self.temperature
        p_ad = 1/(1+math.exp(-max(-700, min(700, logit))))
        return BackendResult({'ad': p_ad, 'non_ad': 1-p_ad}, DecisionLevel.L2, 'tfidf-logistic-baseline-v1', {'backend': 'tfidf_cpu', 'truncated': False, 'temperature': self.temperature})

    def runtime_status(self):
        return {'backend': 'tfidf_cpu', 'loaded': True, 'gpu_required': False}
