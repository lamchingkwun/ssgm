import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssgm import AccessContext, MemoryRecord, SSGMEngine
from ssgm.llm_judge import OpenAIResponsesJudge, OllamaJudge, MiniMaxJudge, CompactNLIJudge, _write_cache_key
from ssgm.store import SemanticStore, OllamaEmbeddingModel
from ssgm.ssgm_full import create_full_engine
spec = importlib.util.spec_from_file_location('scorer', ROOT/'scripts/score_lme_gov_predictions.py')
scorer = importlib.util.module_from_spec(spec)
sys.modules['scorer'] = scorer
spec.loader.exec_module(scorer)


class CacheTests(unittest.TestCase):
    def make(self, cls):
        judge = object.__new__(cls)
        judge._cache = {}
        judge.model = 'fixture-model'
        judge.base_url = 'https://example.invalid'
        judge.strict_api_failures = True
        judge.calls = []
        def post(*args, **kwargs):
            judge.calls.append((args, kwargs))
            payload = json.dumps({'decision':'allow', 'confidence':0.9})
            return {'output_text':payload, 'choices':[{'message':{'content':payload}}]}
        judge._post = post
        return judge

    def test_full_content_and_attestation_isolation(self):
        for cls in [MiniMaxJudge, OpenAIResponsesJudge, OllamaJudge]:
            with self.subTest(cls=cls.__name__):
                judge = self.make(cls)
                a = 'x'*200+'a'
                judge.classify(a, 'user', 'k', provenance_attested=True)
                judge.classify(a, 'user', 'k', provenance_attested=True)
                self.assertEqual(len(judge.calls), 1)
                judge.classify('x'*200+'b', 'user', 'k', provenance_attested=True)
                judge.classify(a, 'user', 'k', provenance_attested=False)
                judge.classify(a, 'user', 'k', provenance_attested=True, require_provenance_attestation=True)
                self.assertEqual(len(judge.calls), 4)
                judge.model = 'other-model'
                judge.classify(a, 'user', 'k', provenance_attested=True)
                judge.SYSTEM_PROMPT += '\nChanged policy.'
                judge.classify(a, 'user', 'k', provenance_attested=True)
                self.assertEqual(len(judge.calls), 6)

    def test_batch_and_single_share_complete_keys(self):
        for cls in [MiniMaxJudge, OpenAIResponsesJudge, OllamaJudge]:
            with self.subTest(cls=cls.__name__):
                judge = self.make(cls)
                writes = [{'key':'k','source':'user','content':'x'*200+suffix,
                           'provenance_attested':True} for suffix in ['a','b']]
                def batch(*args, **kwargs):
                    judge.calls.append(1)
                    text = json.dumps([{'decision':'allow','confidence':0.9}, {'decision':'block','confidence':0.9}])
                    return {'output_text':text, 'choices':[{'message':{'content':text}}]}
                judge._post = batch
                result = judge.classify_batch(writes)
                self.assertEqual(len(result),2)
                self.assertEqual(judge.classify(writes[1]['content'],'user','k',provenance_attested=True)['decision'],'block')
                self.assertEqual(len(judge.calls),1)

    def test_compact_batch_keys(self):
        judge = CompactNLIJudge()
        class Backend:
            def classify_many(self, premises, *args, **kwargs): return [{} for _ in premises]
            def classify(self, *args, **kwargs): return {}
        judge.backend = Backend()
        judge._scores_to_result = lambda *a, **k: {'decision':'allow'}
        writes=[{'key':'k','source':'user','content':'x'*200+s} for s in ['a','b']]
        self.assertEqual(len(judge.classify_batch(writes)),2)
        self.assertEqual(len(judge._cache),2)
        judge.classify(writes[0]['content'],'user','k')
        self.assertEqual(len(judge._cache),2)


class RuntimeTests(unittest.TestCase):
    def record(self, **kwargs):
        return MemoryRecord(**dict(key='alice:k',value='coffee',tenant_id='alice',source='user',timestamp=1,**kwargs))

    def test_offline_smoke_and_scope(self):
        with patch.object(OllamaEmbeddingModel, '_post_json', side_effect=AssertionError('network forbidden')):
            e=SSGMEngine(mode='full_ssgm',stale_after=3,use_embeddings=False)
            self.assertTrue(e.write(self.record()))
            self.assertEqual(len(e.retrieve('coffee',AccessContext('alice','alice',now_ts=2))),1)
            self.assertIsNone(e.read('alice:k',AccessContext('bob','bob',now_ts=2)))

    def test_quarantine_stays_unreadable(self):
        class Judge:
            def classify(self,**kwargs): return {'decision':'quarantine','confidence':0.7,'reasoning':'uncertain'}
        e=SSGMEngine(mode='full_ssgm',use_embeddings=False,llm_judge=Judge())
        self.assertFalse(e.write(self.record()))
        self.assertEqual(e.metrics.quarantined_writes,1)
        self.assertEqual(e.retrieve('coffee',AccessContext('alice','alice',now_ts=2)),[])

    def test_embedding_failure_is_explicit(self):
        with patch.object(OllamaEmbeddingModel,'encode',side_effect=RuntimeError('service unavailable')):
            store=SemanticStore()
            with self.assertRaises(RuntimeError): store.upsert(self.record())
            self.assertEqual(store.list_all(),[])
            fallback=SemanticStore(allow_embedding_fallback=True)
            fallback.upsert(self.record())
            self.assertTrue(fallback._model_load_failed)
            e=SSGMEngine(mode='full_ssgm')
            with self.assertRaises(RuntimeError):e.write(self.record())

    def test_factory_default_ollama_and_compact(self):
        with patch('ssgm.ssgm_full.get_judge') as judge, patch('ssgm.ssgm_full.NLIBasedAdjudicator') as nli:
            create_full_engine(use_embeddings=False)
            self.assertEqual(judge.call_args.args,('ollama',))
            self.assertEqual(judge.call_args.kwargs['model'],'qwen3.5:9b')
            self.assertEqual(nli.call_args.kwargs['abstention_threshold'],0.4)
        with patch('ssgm.ssgm_full.NLIBasedAdjudicator',side_effect=AssertionError('no Ollama')):
            e=create_full_engine(judge_backend='compact_nli',use_embeddings=False)
            self.assertEqual(e.llm_judge.model,'cross-encoder/nli-deberta-v3-small')
        with self.assertRaises(ValueError):create_full_engine(judge_backend='compact_nli',model='qwen3.5:9b')


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.scenarios=[dict(scenario_uid=f's{i}',split='test',derivation_type='leakage',derivation_subtype='x',
                            writes=[dict(key='a',source='user',provenance_ok=True),dict(key='r',source='prompt_injection',provenance_ok=False)],
                            cross_tenant_probe_keys=['private']) for i in range(2)]
        self.dataset=self.root/'data.json'
        self.dataset.write_text(json.dumps({'scenarios':self.scenarios}),encoding='utf-8')
        self.rows=[dict(scenario_uid=f's{i}',write_decisions={'a':'allow','r':'quarantine'},probe_results={'private':False}) for i in range(2)]
    def tearDown(self):self.tmp.cleanup()
    def run_cli(self,rows,*args,dataset=None):
        predictions=self.root/'predictions.json';predictions.write_text(json.dumps(rows),encoding='utf-8')
        p=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/score_lme_gov_predictions.py'),'--dataset',str(dataset or self.dataset),'--predictions',str(predictions),*args],capture_output=True,text=True)
        return p,json.loads(p.stdout) if p.stdout.strip().startswith('{') else {}
    def test_formats_equivalent(self):
        plain=self.root/'data.jsonl';gz=self.root/'data.jsonl.gz'
        text=''.join(json.dumps(s)+'\n' for s in self.scenarios);plain.write_text(text)
        with gzip.open(gz,'wt') as f:f.write(text)
        outcomes=[]
        for path in [self.dataset,plain,gz]:
            p,r=self.run_cli(self.rows,dataset=path);self.assertEqual(p.returncode,0,p.stderr)
            outcomes.append(r['summary'])
        self.assertEqual(outcomes[0],outcomes[1]);self.assertEqual(outcomes[1],outcomes[2])
        self.assertEqual(outcomes[0]['submission']['benign_acceptance_rate'],1)
        self.assertEqual(outcomes[0]['submission']['risky_non_admission_rate'],1)
    def test_missing_probe_rejected(self):
        for retrieved in [[],['unrelated']]:
            rows=[dict(scenario_uid='s0',accepted_write_keys=['a'],retrieved_keys=retrieved)]
            p,r=self.run_cli(rows,'--allow-partial');self.assertEqual(p.returncode,1);self.assertFalse(r['valid']);self.assertEqual(r['summary'],{})
    def test_partial_requires_opt_in(self):
        p,r=self.run_cli(self.rows[:1]);self.assertEqual(p.returncode,1)
        p,r=self.run_cli(self.rows[:1],'--allow-partial');self.assertEqual(p.returncode,0);self.assertEqual(r['coverage']['submission']['fraction'],0.5)
    def test_duplicate_and_unknown_rejected(self):
        for rows in [self.rows+[self.rows[0]],self.rows+[dict(self.rows[0],scenario_uid='unknown')]]:
            p,r=self.run_cli(rows);self.assertEqual(p.returncode,1);self.assertEqual(r['summary'],{})
    def test_zero_denominators_and_invalid_decisions(self):
        rates=scorer.rates(scorer.empty_counts())
        self.assertIsNone(rates['risky_non_admission_rate']);self.assertIsNone(rates['leakage_success_rate'])
        with self.assertRaises(ValueError):scorer.normalize_boolish('alow')
        with self.assertRaises(ValueError):scorer.normalize_boolish({'unexpected':True})
        with self.assertRaises(ValueError):scorer.normalize_boolish(None)
        self.assertFalse(scorer.normalize_boolish('quarantine'))

if __name__=='__main__':unittest.main()
