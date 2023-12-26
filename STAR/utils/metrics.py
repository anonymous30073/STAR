# how to do when micro averaging in cross validation ask andrew (do micro for each fold and then avg? or "concat" all results and do micro for all?)
import json
from collections import defaultdict

import pytrec_eval
from sklearn.metrics import ndcg_score, roc_auc_score
import numpy as np

from scipy import stats

INTERNAL_USER_ID_FIELD = "internal_user_id"
INTERNAL_ITEM_ID_FIELD = "internal_item_id"

ranking_metrics = [
    "ndcg_cut_5",
    "ndcg_cut_10",
    "P_1",
]


def get_significance(dict1, dict2, level = 0.05):
    list1 = []
    list2 = []
    for k in dict1.keys():
        list1.append(dict1[k])
        list2.append(dict2[k])
    return stats.ttest_rel(list1, list2).pvalue


def calculate_ranking_metrics(gt, pd, relevance_level,
                              given_ranking_metrics=None, calc_pytrec=False,
                              exp_dir=None, anchor_path=None, avg=True):
    if given_ranking_metrics is None:
        given_ranking_metrics = ranking_metrics
    ndcg_metrics = [m for m in given_ranking_metrics if m.startswith("ndcg_")]
    results, results_dict = calculate_ndcg(gt, pd, ndcg_metrics)
    if calc_pytrec:
        gt = {k: {k2: int(v2) for k2, v2 in v.items()} for k, v in gt.items()}
        r2, d2 = calculate_ranking_metrics_pytreceval(gt, pd, relevance_level, given_ranking_metrics)
        for m, v in r2.items():
            if m in results:
                continue
                results[f"pytrec_{m}"] = v
            else:
                results[m] = v
        for m, v in d2.items():
            if m in results_dict:
                continue
                results_dict[f"pytrec_{m}"] = v
            else:
                results_dict[m] = v
    if avg:
        if exp_dir:
            json.dump(results_dict, open(exp_dir + "per_user_dict.js", "w"))
        for m in results:
            assert len(results[m]) == len(gt)
            results[m] = np.array(results[m]).mean(axis=0).tolist()
        if anchor_path:
            anchor_dict = json.load(open(anchor_path + "per_user_dict.js"))
            for m in results_dict.keys():
                results["significance_" + m] = get_significance(anchor_dict[m], results_dict[m]) if anchor_path else 0
    return results


def calculate_ranking_metrics_pytreceval(gt, pd, relevance_level, given_ranking_metrics):
    '''
    :param gt: dict of user -> item -> true score (relevance)
    :param pd: dict of user -> item -> predicted score
    :param relevance_level:
    :param given_ranking_metrics:
    :return: metric scores
    '''
    evaluator = pytrec_eval.RelevanceEvaluator(gt, given_ranking_metrics, relevance_level=int(relevance_level))
    scores = evaluator.evaluate(pd)
    per_qid_score = defaultdict()
    per_qid_score_dict = defaultdict()
    for m in given_ranking_metrics:
        per_qid_score[m] = [scores[qid][m] for qid in gt.keys()]
        per_qid_score_dict[m] = dict((qid, scores[qid][m]) for qid in gt.keys())
    # scores = [[metrics_dict.get(m, -1) for m in given_ranking_metrics] for metrics_dict in per_user_scores.values()]
    # scores = np.array(scores).mean(axis=0).tolist()
    # per_qid_scores = dict(zip(given_ranking_metrics, scores))
    return per_qid_score, per_qid_score_dict


def ndcg(gt, pd, k):
    per_qid_score = []
    per_qid_score_dict = dict()
    for qid in gt.keys():
        user_items = gt[qid].keys()
        true_rel = [[gt[qid][k] for k in user_items]]
        pred = [[pd[qid][k] for k in user_items]]
        temp = ndcg_score(true_rel, pred, k=k)
        per_qid_score.append(temp)
        per_qid_score_dict[qid] = temp
    return per_qid_score, per_qid_score_dict


def calculate_ndcg(gt, pd, given_ranking_metrics):
    '''

    :param gt: dict of user -> item -> true score (relevance)
    :param pd: dict of user -> item -> predicted score
    :param relevance_level:
    :param given_ranking_metrics:
    :return: metric scores
    '''
    per_qid_score = defaultdict()
    per_qid_score_dict = defaultdict()
    for m in given_ranking_metrics:
        if m.startswith("ndcg_cut_"):
            per_qid_score[m], per_qid_score_dict[m] = ndcg(gt, pd, int(m[m.rindex("_")+1:]))
        else:
            raise NotImplementedError("other metrics not implemented")
    return per_qid_score, per_qid_score_dict


def log_results(ground_truth, prediction_scores, internal_user_ids, internal_items_ids,
                external_users, external_items, output_path_ground_truth, output_path_predicted, output_path_log=None):
    # we want to log the results corresponding to external user and item ids
    ex_users = external_users.to_pandas().set_index(INTERNAL_USER_ID_FIELD)
    user_ids = ex_users.loc[internal_user_ids].user_id.values
    ex_items = external_items.to_pandas().set_index(INTERNAL_ITEM_ID_FIELD)
    item_ids = ex_items.loc[internal_items_ids].item_id.values

    gt = {str(u): {} for u in sorted(set(user_ids))}
    pd = {str(u): {} for u in sorted(set(user_ids))}
    for i in range(len(ground_truth)):
        gt[str(user_ids[i])][str(item_ids[i])] = float(ground_truth[i])
        pd[str(user_ids[i])][str(item_ids[i])] = float(prediction_scores[i])
    json.dump({"predicted": pd}, open(output_path_predicted, 'w'))
    json.dump({"ground_truth": gt}, open(output_path_ground_truth, 'w'))
    cnt = 0
    if output_path_log is not None and 'text' in ex_users.columns:
        with open(output_path_log, "w") as f:
            for user_id in gt.keys():
                if cnt == 100:
                    break
                cnt += 1
                f.write(f"user:{user_id} - text:{ex_users[ex_users['user_id'] == user_id]['text'].values[0]}\n\n\n")
                for item_id, pd_score in sorted(pd[user_id].items(), key=lambda x:x[1], reverse=True):
                    f.write(f"item:{item_id} - label:{gt[user_id][item_id]} - score:{pd_score} - text:{ex_items[ex_items['item_id'] == item_id]['text'].values[0]}\n\n")
                f.write("-----------------------------\n")


def calculate_metrics(ground_truth, prediction_scores, users, items, relevance_level, given_ranking_metrics=None):
    # # qid= user1:{ item1:1 } ...
    gt = {str(u): {} for u in set(users)}
    pd = {str(u): {} for u in set(users)}
    # min_not_zero = 1
    for i in range(len(ground_truth)):
        if len(items) == 0:
            gt[str(users[i])][str(i)] = float(ground_truth[i])
            pd[str(users[i])][str(i)] = float(prediction_scores[i])
        else:
            gt[str(users[i])][str(items[i])] = float(ground_truth[i])
            pd[str(users[i])][str(items[i])] = float(prediction_scores[i])
        # if ground_truth[i] != 0 and ground_truth[i] < min_not_zero:
        #     min_not_zero = ground_truth[i]
    return calculate_ranking_metrics_macro_avg_over_qid(gt, pd, relevance_level, given_ranking_metrics, calc_pytrec=False)


def calculate_ranking_metrics_macro_avg_over_qid(gt, pd, relevance_level,
                                                   given_ranking_metrics=None, calc_pytrec=True):
    if given_ranking_metrics is None:
        given_ranking_metrics = ranking_metrics
    ndcg_metrics = [m for m in given_ranking_metrics if m.startswith("ndcg_")]
    results, results_dict = calculate_ndcg(gt, pd, ndcg_metrics)
    if calc_pytrec or True:
        gt = {k: {k2: int(v2) for k2, v2 in v.items()} for k, v in gt.items()}

        r2, d2 = calculate_ranking_metrics_pytreceval(gt, pd, relevance_level, given_ranking_metrics)
        for m, v in r2.items():
            if m in results:
                results[f"pytrec_{m}"] = v
            else:
                results[m] = v
    for m in results:
        results[m] = np.array(results[m]).mean(axis=0).tolist()
    return results


