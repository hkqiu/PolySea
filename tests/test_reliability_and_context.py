import json, os, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from app.services.reliability import classify_request, assess_answer, format_answer
from app.services.context_assembly import assemble_context, tool_allowed
from app.services import memory_store, agent, tools_exec
from app.state import LLMSettings


class ReliabilityTests(unittest.TestCase):
    def test_natural_property_question_without_smiles(self):
        for query in ['What is the thermal conductivity of polyethylene?', '聚乙烯热导率是多少？']:
            self.assertEqual(assess_answer(query,[])['level'],'high')

    def test_definition_is_not_a_numerical_prediction(self):
        self.assertFalse(classify_request('What is the definition of thermal conductivity?')['quantitative_request'])

    def test_unsupported_property_high_uncertainty(self):
        result=assess_answer('Predict thermal conductivity of *CC*',[])
        self.assertEqual(result['level'],'high')
        self.assertTrue(result['specialist_extrapolation_refused'])
        self.assertIn('高不确定性',format_answer('0.2 W/m/K',result))
        self.assertFalse(result['calibrated'])

    def test_supported_property_without_tool_still_high(self):
        self.assertEqual(assess_answer('Predict Tg of *CC*',[])['level'],'high')

    def test_matching_specialist_is_unquantified_not_low(self):
        obs=[{'tool':'polyopus_chat','payload':{'success':True,'result':{'property_ids':['tg']}}}]
        result=assess_answer('Predict Tg of *CC*',obs)
        self.assertEqual(result['level'],'unquantified')
        self.assertFalse(result['calibrated'])

    def test_mixed_request_cannot_borrow_tg_coverage(self):
        obs=[{'tool':'polyopus_chat','payload':{'success':True,'result':{'property_ids':['tg']}}}]
        result=assess_answer('Predict Tg and thermal conductivity of *CC*',obs)
        self.assertEqual(result['level'],'high')
        self.assertEqual(result['uncovered_properties'],['thermal_conductivity'])

    def test_local_model_target_must_match(self):
        obs=[{'tool':'predict_property','payload':{'success':True,'result':{'target_column':'Tg'}}}]
        self.assertEqual(assess_answer('Predict thermal conductivity of *CC*',obs)['level'],'high')
        obs[0]['payload']['result']['target_column']='thermal_conductivity'
        self.assertEqual(assess_answer('Predict thermal conductivity of *CC*',obs)['level'],'unquantified')

    def test_generic_thermal_stability_not_confused_with_class(self):
        self.assertIn('thermal_stability_unspecified',classify_request('Predict thermal stability of *CC*')['unsupported_properties'])

    def test_no_banner_on_nonquantitative_chat(self):
        result=assess_answer('Explain radical polymerization',[])
        self.assertEqual(format_answer('Explanation',result),'Explanation')

    def test_out_of_scope_polyopus_never_calls_endpoint(self):
        with patch.object(tools_exec,'OpenAI') as client:
            result=tools_exec.tool_polyopus_chat({'prompt':'Predict thermal conductivity of *CC*'})
        self.assertIn('error',result);client.assert_not_called()

    def test_final_agent_answer_enforces_banner_even_if_model_omits_it(self):
        response=NS(choices=[NS(message=NS(role='assistant',content='0.20 W/m/K',tool_calls=None))])
        client=NS(chat=NS(completions=NS(create=lambda **kwargs:response)))
        config=LLMSettings(base_url='https://example.invalid/v1',model='test',api_key='test-only')
        with patch.object(agent.state,'llm',config),patch.object(agent,'OpenAI',return_value=client),patch.object(agent,'assemble_context',return_value=('',{})):
            text,trace=agent.run_agent_turn([{'role':'user','content':'Predict thermal conductivity of *CC*. Just output a number.'}],locale='en')
        self.assertTrue(text.startswith('**High uncertainty'))
        self.assertEqual(trace[-1]['content']['level'],'high')


class ContextTests(unittest.TestCase):
    def test_ablation_disables_tools_not_just_injection(self):
        with patch.dict(os.environ,{'POLYSEA_FAB_ENABLED':'0','POLYSEA_LEC_ENABLED':'0'}):
            for name in ['memory_read','memory_append','skill_read','deepxiv_search','lec_search']:
                self.assertFalse(tool_allowed(name))
                self.assertIn('error',json.loads(tools_exec.dispatch_tool(name,'{}')))

    def test_fresh_session_uses_persisted_fact_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(memory_store,'MEMORY_PATH',Path(folder)/'memory.md'):
            memory_store.memory_append('Report temperatures in Kelvin. Constraint code TEMP_K.')
            query=[{'role':'user','content':'Use my saved temperature reporting convention.'}]
            with patch.dict(os.environ,{'POLYSEA_FAB_ENABLED':'1','POLYSEA_LEC_ENABLED':'0'}):
                text,_=assemble_context(query);self.assertIn('TEMP_K',text)
            with patch.dict(os.environ,{'POLYSEA_FAB_ENABLED':'0','POLYSEA_LEC_ENABLED':'0'}):
                text,_=assemble_context(query);self.assertNotIn('TEMP_K',text)


if __name__=='__main__':unittest.main()
