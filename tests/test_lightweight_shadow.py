import json
import threading

from collector.schemas import Candidate
from detector.candidate_ranker import CandidateAssessment
from jev.backend import BackendResult
from jev.decider import JevDecider
from jev.types import DecisionLevel
from policy.blocking_policy import BlockingPolicy
from runtime.lightweight_shadow import LightweightShadow


def candidate(identifier='one', text='secret@example.test'):
    return Candidate.from_dict({'candidate_id':identifier,'page':{'url':'https://private.test/','top_level_domain':'private.test'},'tag':'DIV','element_id':identifier,'text':text,'visible':True,'width':350,'height':120})


class Backend:
    def __init__(self, gate=None, fail=False):
        self.gate = gate
        self.entered = threading.Event()
        self.fail = fail

    def decide(self, state, question):
        self.entered.set()
        if self.gate:
            self.gate.wait(timeout=3)
        if self.fail:
            raise RuntimeError('private error message')
        return BackendResult({'ad':.98,'non_ad':.02},DecisionLevel.L2,'test',{})


def observer(backend, **kwargs):
    return LightweightShadow(JevDecider(backend,.08,.55),BlockingPolicy(),**kwargs)


def test_shadow_never_changes_primary_and_does_not_persist_content():
    shadow = observer(Backend())
    primary = {'command':{'action':'allow'}}
    before = json.dumps(primary)
    try:
        assert shadow.offer(candidate(),CandidateAssessment(.5,['test'],'jev'),primary['command']['action'])
        assert shadow.wait_idle()
        stats = shadow.stats()
        assert primary == json.loads(before)
        assert stats['counters']['potential_new_actions'] == 1
        assert stats['would_actions'] == {'block':1}
        assert 'private.test' not in json.dumps(stats)
        assert 'secret' not in json.dumps(stats)
        assert not shadow.offer(candidate(),CandidateAssessment(.5,['test'],'jev'),'allow')
        assert stats['queue_capacity'] == 64
    finally:
        assert shadow.close()


def test_shadow_queue_is_bounded_and_does_not_wait_for_model():
    gate = threading.Event()
    backend = Backend(gate)
    shadow = observer(backend,queue_size=1,dedup_size=2)
    try:
        assert shadow.offer(candidate('one'),CandidateAssessment(.5,[],'jev'),'allow')
        assert backend.entered.wait(timeout=2)
        assert shadow.offer(candidate('two'),CandidateAssessment(.5,[],'jev'),'allow')
        assert not shadow.offer(candidate('three'),CandidateAssessment(.5,[],'jev'),'allow')
        assert shadow.stats()['counters']['queue_dropped'] == 1
    finally:
        gate.set()
        assert shadow.close()


def test_shadow_failure_clear_skip_and_protected_policy():
    shadow = observer(Backend(fail=True))
    try:
        assert not shadow.offer(candidate(),CandidateAssessment(0,[],'allow'),'allow')
        assert shadow.offer(candidate(),CandidateAssessment(.5,[],'jev'),'allow')
        assert shadow.wait_idle()
        assert shadow.stats()['counters']['model_errors'] == 1
        assert shadow.stats()['last_error_type'] == 'RuntimeError'
        assert 'private error' not in json.dumps(shadow.stats())
    finally:
        shadow.close()
    safe = observer(Backend())
    try:
        item = candidate()
        from dataclasses import replace
        item = replace(item,tag='ARTICLE')
        safe.offer(item,CandidateAssessment(.5,[],'jev'),'allow')
        assert safe.wait_idle()
        assert safe.stats()['would_actions'] == {'allow':1}
    finally:
        safe.close()


def test_shadow_callback_failure_and_closed_offers_are_isolated():
    def fail(*args):
        raise ValueError('review sink failed')
    shadow = observer(Backend(),on_decision=fail)
    assert shadow.offer(candidate(),CandidateAssessment(.5,[],'jev'),'allow')
    assert shadow.wait_idle()
    assert shadow.stats()['counters']['processed'] == 1
    assert shadow.stats()['counters']['callback_errors'] == 1
    assert shadow.stats()['counters']['model_errors'] == 0
    assert shadow.close()
    assert not shadow.offer(candidate('after-close'),CandidateAssessment(.5,[],'jev'),'allow')
