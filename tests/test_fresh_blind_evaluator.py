import json

from collector.schemas import Candidate
from detector.candidate_ranker import CandidateAssessment
from evaluation.benchmark import AblationMode, BenchmarkCase
from jev.backend import BackendResult
from jev.types import DecisionLevel
from scripts.evaluate_fresh_blind import make_evaluator


class Backend:
    def __init__(self):
        self.states = []

    def decide(self, state, question):
        self.states.append(json.loads(state))
        return BackendResult({'ad':.95, 'non_ad':.05}, DecisionLevel.L2, 'test', {'truncated':False})


def test_native_cascade_and_evidence_free_diagnostic():
    backend = Backend()
    plan = {'policy':{'block_threshold':.9, 'hide_threshold':.72, 'min_decision_level':'L1', 'protected_tags':['MAIN'], 'fail_open':True}, 'detector':{'block_score':.85, 'allow_score':.12}, 'jev_thresholds':{'uncertainty_margin':.08, 'min_confidence':.55}}
    evaluator = make_evaluator(plan, backend, 'candidate-features-v3')
    evaluator.ranker.assess = lambda candidate: CandidateAssessment(.5 if candidate.element_id == 'gray' else 0, ['test_evidence'], 'jev' if candidate.element_id == 'gray' else 'allow')
    cases = []
    for identifier, label in [('gray',0), ('clear',1)]:
        candidate = Candidate.from_dict({'candidate_id':identifier, 'element_id':identifier, 'page':{'url':'https://new.test/', 'top_level_domain':'new.test'}, 'tag':'DIV', 'width':300, 'height':250, 'visible':True, 'direct_resource_urls':['https://resource.test/image'], 'resource_ownership_known':True})
        cases.append(BenchmarkCase(identifier, 'new.test', label, candidate))
    baseline = evaluator.run_mode(cases, AblationMode.RULES_ONLY)
    assert baseline['rows'][0]['prediction'] == 2
    assert backend.states == []
    cascade = evaluator.run_mode(cases, AblationMode.CASCADE)
    assert cascade['routing']['model_calls'] == 1
    assert cascade['page_action_ad_recall'] == 1
    assert cascade['page_action_false_positive_rate'] == 0
    assert backend.states[0]['schema_version'] == 'candidate-features-v3'
    assert backend.states[0]['direct_resource_urls'] == ['https://resource.test/image']
    assert backend.states[0]['detector_evidence'] == ['test_evidence']
    evaluator.run_mode(cases, AblationMode.MODEL_ONLY)
    assert len(backend.states) == 3
    assert all(state['detector_evidence'] == [] for state in backend.states[1:])
