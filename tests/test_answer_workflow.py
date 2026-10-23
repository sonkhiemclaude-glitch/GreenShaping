import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('answer',Path(__file__).parents[1]/'scripts/validate_higgs_answer_workflow.py')
# scripts are sibling imports in command-line execution.
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_answer_checks_numeric_precision_counts_and_schema_has_no_ground_truth_values():
    ref={'count':14,'weights':[.123456789,.0005]}
    assert m.answer_matches({'count':14,'weights':[.12345679,.0005]},ref)
    for wrong in [{'count':True,'weights':[.123456789,.0005]}, {'count':14.0,'weights':[.123456789,.0005]},
                  {'count':14,'weights':[float('nan'),.0005]}, {'count':14,'weights':[.1,.0005]}]:
        assert not m.answer_matches(wrong,ref)
    schema=m.answer_schema(ref)
    assert '0.123456789' not in str(schema) and 'const' not in str(schema) and 'enum' not in str(schema)


def test_accepted_plan_replaces_frozen_question_without_mutation():
    from greenshaping.testbed.question_bank import Question
    q=Question(1,'record_count','request',[[{'site':0,'params':{}}]])
    calls=[{'site':1,'params':{}}]
    actual=m.replace(q,plan=[calls])
    assert actual.plan==[calls]
    assert q.plan==[[{'site':0,'params':{}}]]
    assert actual.question_id==q.question_id and actual.metadata==q.metadata


def test_decimal_result_presentation_preserves_magnitudes_and_original_matcher():
    import json
    ref={'weights':[-3.133041557853051e-05,-8.92789592141558e-05,1.25e20,1e-12], 'n':1892953}
    text=m.decimal_json(ref)
    assert 'e-' not in text and 'e+' not in text
    assert '-0.00003133041557853051' in text
    assert json.loads(text)==ref
    assert m.answer_matches(json.loads(text),ref)
    wrong={'weights':[-3.133041557853051,-8.92789592141558,1.25e20,1e-12],'n':1892953}
    assert not m.answer_matches(wrong,ref)
