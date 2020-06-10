from nose.tools import assert_list_equal
import numpy as np
import pandas as pd

from ..results import ModelCollection, ModelResults, PatientResults

y_test_fold1 = pd.DataFrame([['a', 1]] * 110, columns=['patient', 'y'])
y_test_fold2 = pd.DataFrame([['b', 0]] * 150, columns=['patient', 'y'])


def test_patient_results_sunny_day():
    patient_a = PatientResults('a', 1, 1, 0)
    patient_a.set_results([1] * 60 + [0] * 50)
    lst, cols = patient_a.to_list()
    assert_list_equal(lst, [
        'a',
        50,
        60,
        60 / 110.0,
        1,
        1,
        0,
        1,
    ])
    assert_list_equal(cols, [
        'patient_id',
        'other_votes',
        'ards_votes',
        'frac_votes',
        'majority_prediction',
        'fold_idx',
        'epoch',
        'ground_truth',
    ])


def test_calc_fold_stats():
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, 1, 0)
    model_collection.add_model(y_test_fold2, pred2, 2, 0)
    model_collection.calc_fold_stats(1, print_results=False)
    results = model_collection.model_results['folds'][1]
    assert results[results.patho == 'ards'].iloc[0].recall == 1
    assert results[results.patho == 'ards'].iloc[0].prec == 1
    assert np.isnan(results[results.patho == 'other'].iloc[0].recall)
    assert results[results.patho == 'other'].iloc[0].spec == 1
    model_collection.calc_fold_stats(2, print_results=False)
    results = model_collection.model_results['folds'][2]
    assert results[results.patho == 'other'].iloc[0].recall == 1
    assert results[results.patho == 'other'].iloc[0].prec == 1
    assert np.isnan(results[results.patho == 'ards'].iloc[0].recall)
    assert results[results.patho == 'ards'].iloc[0].spec == 1


def test_calc_aggregate_stats():
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, 1, 0)
    model_collection.add_model(y_test_fold2, pred2, 2, 0)
    model_collection.calc_aggregate_stats(print_results=False)
    results = model_collection.model_results['aggregate']
    assert results[results.patho == 'ards'].iloc[0].recall == 1
    assert results[results.patho == 'ards'].iloc[0].prec == 1
    assert results[results.patho == 'other'].iloc[0].spec == 1
    assert results[results.patho == 'other'].iloc[0].recall == 1
    assert results[results.patho == 'other'].iloc[0].prec == 1
    assert results[results.patho == 'ards'].iloc[0].spec == 1


def test_get_all_patient_results_dataframe():
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, 1, 0)
    model_collection.add_model(y_test_fold2, pred2, 2, 0)
    expected = pd.DataFrame([
        ['a', 50, 60, 60 / 110.0, 1, 1, 0, 1],
        ['b', 110, 40, 40 / 150.0, 0, 2, 0, 0],
    ], columns=['patient_id', 'other_votes', 'ards_votes', 'frac_votes', 'majority_prediction', 'fold_idx', 'epoch', 'ground_truth'])
    res = model_collection.get_all_patient_results_dataframe()
    assert (expected == res).all().all()


def test_count_predictions():
    model = ModelResults(1, 0)
    pred1 = pd.Series([0] * 50 + [1] * 60)
    model.set_results(y_test_fold1, pred1)
    res, cols = model.count_predictions()
    assert res == [0, 1, 0, 0, 1, 0, 0, 0, 1, 0], res


def test_auc_results():
    patient_results = pd.DataFrame([
        [50, 60, 60 / 110.0, 1, 1, 0, 1],
        [50, 60, 60 / 110.0, 1, 1, 0, 1],
        [50, 60, 60 / 110.0, 1, 1, 0, 1],
        [110, 40, 40 / 150.0, 0, 2, 0, 0],
        [110, 40, 40 / 150.0, 0, 2, 0, 0],
        [110, 40, 40 / 150.0, 0, 2, 0, 0],
        [50, 60, 60 / 110.0, 1, 1, 1, 1],
        [50, 60, 60 / 110.0, 1, 1, 1, 1],
        [50, 60, 60 / 110.0, 1, 1, 1, 1],
        [110, 40, 40 / 150.0, 0, 2, 1, 0],
        [110, 40, 40 / 150.0, 0, 2, 1, 0],
        [110, 40, 40 / 150.0, 0, 2, 1, 0],
    ], columns=['other_votes', 'ards_votes', 'frac_votes', 'majority_prediction', 'fold_idx', 'epoch', 'ground_truth'])
    model_collection = ModelCollection()
    aucs = model_collection.get_auc_results(patient_results)
    assert (aucs == np.array([1, 1])).all()
