import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from proteinevo.config import ROOT,SPLITS
from proteinevo.delivery import workflow as w
from proteinevo.evidence_graph import EvidenceGraph

class ReportWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=w.observations(pd.read_csv(SPLITS/'gb1_initial_labeled.csv'))
        cls.known=cls.source.head(100);cls.available=set(cls.source.variant)-set(cls.known.variant)
        cls.model=w.fit_model(cls.known)
        (ROOT/'runs/test-temp').mkdir(parents=True,exist_ok=True)
    def temp(self):return tempfile.TemporaryDirectory(dir=ROOT/'runs/test-temp')
    def test_invalid_observations_and_candidate_labels_rejected(self):
        with self.assertRaises(ValueError):w.observations(pd.DataFrame({'variant':['V*GV'],'fitness':[1]}))
        with self.assertRaises(ValueError):w.observations(pd.DataFrame({'variant':['VDGV'],'fitness':[float('nan')]}))
        with self.assertRaises(ValueError):w.candidate_ids(self.source,self.known.variant)
    def test_changed_observation_model_cannot_be_reused(self):
        with self.temp() as t:
            changed=self.known.copy();changed.loc[0,'fitness']+=1
            with self.assertRaises(ValueError):w.recommend(changed,self.model,Path(t)/'wrong',available=self.available)
    def test_recommendations_need_no_hidden_labels_and_have_full_sequences(self):
        with self.temp() as t,patch.object(pd,'read_csv',side_effect=AssertionError('Hidden files unavailable')):
            rec,_,_=w.recommend(self.known,self.model,Path(t)/'recommend',available=self.available,candidate_budget=80,batch_size=8)
            self.assertEqual(len(rec),8);self.assertEqual(rec.variant.nunique(),8)
            self.assertTrue(set(rec.variant)<=self.available);self.assertTrue((rec.full_sequence.str.len()==56).all())
            self.assertNotIn('experimental_fitness',rec)
    def test_training_and_evaluation_overlap_rejected(self):
        with self.temp() as t,self.assertRaises(ValueError):w.evaluate_model(self.model,self.known,Path(t)/'overlap')
    def test_llm_failure_not_replaced_by_rule(self):
        with self.temp() as t,patch('proteinevo.active_science.planner.Planner.client',side_effect=RuntimeError('no model')):
            out=Path(t)/'llm'
            with self.assertRaisesRegex(RuntimeError,'no model'):w.recommend(self.known,self.model,out,'tool_llm',self.available,80,8)
            self.assertEqual(json.loads((out/'execution_state.json').read_text())['status'],'failed')
            self.assertFalse((out/'recommendations.csv').exists())
    def test_three_round_feedback_and_no_remeasurement(self):
        with self.temp() as t:
            out=Path(t)/'evolution'
            with patch.object(pd,'read_csv',side_effect=AssertionError('No project label files')):metrics=w.demo(self.source,out)
            rec=pd.read_csv(out/'recommendations.csv')
            self.assertEqual(len(rec),72);self.assertTrue(set(rec.variant)<=set(self.source.variant))
            for method,block in rec.groupby('method'):
                self.assertEqual(block.variant.nunique(),24)
                self.assertEqual([r['observed_n'] for r in metrics if r['method']==method],[77,85,93])
            self.assertEqual(json.loads((out/'execution_state.json').read_text())['real_LLM_calls'],0)
    def test_completed_output_is_not_overwritten(self):
        with self.temp() as t:
            p=Path(t)/'done';p.mkdir();(p/'sentinel.txt').write_text('keep')
            with self.assertRaises(FileExistsError):w.recommend(self.known,self.model,p,available=self.available)
            self.assertEqual((p/'sentinel.txt').read_text(),'keep')
    def test_graph_only_uses_supplied_observations(self):
        with patch.object(pd,'read_csv',side_effect=AssertionError('No oracle')):g=EvidenceGraph(self.known)
        variants={v['variant'] for v in g.nodes.values() if v['type']=='observed_variant'}
        self.assertEqual(variants,set(self.known.variant))
    def test_archived_report_results_match_complete_recommendations(self):
        rec=pd.read_csv(ROOT/'results/evolution/recommendations.csv');summary=pd.read_csv(ROOT/'results/evolution/method_summary.csv').set_index('method')
        self.assertEqual(len(rec),2304)
        for m,block in rec.groupby('method'):
            means=block.groupby('panel_seed').experimental_fitness.mean()
            self.assertAlmostEqual(means.mean(),summary.loc[m,'mean'],places=10)
            self.assertAlmostEqual(means.std(),summary.loc[m,'std'],places=10)
            self.assertEqual(int((block.experimental_fitness>=5).sum()),summary.loc[m,'hits_ge5'])
        for v,rnd,value in [('FWAA',2,8.76200151345),('KWAA',3,0.00928509731509)]:
            row=rec[(rec.panel_seed==45005)&(rec.method=='tool_llm')&(rec['round']==rnd)&(rec.variant==v)]
            self.assertEqual(len(row),1)
            self.assertAlmostEqual(row.experimental_fitness.iloc[0],value,places=3)

if __name__=='__main__':unittest.main()
