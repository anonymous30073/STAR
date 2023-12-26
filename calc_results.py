import argparse
import csv
import json
import os.path
import time
from collections import Counter, defaultdict

import transformers
import pandas as pd

from STAR.utils.metrics import calculate_ranking_metrics

relevance_level = 1


def get_metrics(ground_truth, prediction_scores, ranking_metrics=None, calc_pytrec=True, exp_dir=None, anchor_path=None):
    if len(ground_truth) == 0:
        return {}
    start = time.time()
    results = calculate_ranking_metrics(gt=ground_truth, pd=prediction_scores,
                                        relevance_level=relevance_level,
                                        given_ranking_metrics=ranking_metrics,
                                        calc_pytrec=calc_pytrec, exp_dir=exp_dir, anchor_path=anchor_path)
    print(f"ranking metrics in {time.time() - start}")
    return results


def get_results(prediction, ground_truth, ranking_metrics,
                exp_dir=None,
                anchor_path=None, neg_strategy=None):
    ret = []
    ground_truth = {u: v for u, v in ground_truth.items()}
    prediction = {u: v for u, v in prediction.items()}

    total_results = get_metrics(ground_truth=ground_truth,
                                prediction_scores=prediction,
                                ranking_metrics=ranking_metrics, exp_dir=exp_dir + f"/ALL_{neg_strategy}_", anchor_path = anchor_path + f"/ALL_{neg_strategy}_" if anchor_path else None)
    metric_header = sorted(total_results.keys())
    ret.append({"ALL":  {h: total_results[h] for h in metric_header}})
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--pos_gt_path', type=str, default=None, help='path to gt file')
    parser.add_argument('--neg_gt_path', type=str, default=None, help='path to gt file')
    parser.add_argument('--pred_path', type=str, default=None, help='path to pd file')
    parser.add_argument('--out_path', type=str, default=None, help='path to output file')

    parser.add_argument('--anchor_path', type=str, default=None, help='path to baseline to compute significance. if left none, significance is not computed')
    args, _ = parser.parse_known_args()

    ranking_metrics_ = ["ndcg_cut_5", "P_1"]

    if "test_predicted_test_neg_popular_100_" in args.pred_path:
        ng = "popularity"
    elif "test_predicted_test_neg_SB_BM25_100_" in args.pred_path:
        ng = "SB_BM25"
    elif "test_predicted_test_neg_standard_100_" in args.pred_path:
        ng = "standard"
    else:
        raise ValueError("ng not specified")

    prediction_ = json.load(open(args.pred_path))
    if len(prediction_.keys()) == 1 and "predicted" in prediction_:
        prediction_ = prediction_["predicted"]
    pos_file = pd.read_csv(args.pos_gt_path, dtype=str)
    neg_file = pd.read_csv(args.neg_gt_path, dtype=str)
    ground_truth_ = defaultdict(lambda: defaultdict())
    for user_id, item_id in zip(pos_file["user_id"], pos_file["item_id"]):
        ground_truth_[user_id][item_id] = 1
    for user_id, item_id in zip(neg_file["user_id"], neg_file["item_id"]):
        ground_truth_[user_id][item_id] = 0

    results = get_results(prediction_, ground_truth_, ranking_metrics_,
                          exp_dir=os.path.dirname(args.pred_path),
                          anchor_path=args.anchor_path,
                          neg_strategy=ng)

    outfile = args.out_path
    outfile_f = open(outfile, "a")
    outfile_f.write(" ".join(args.pred_path.split("/")[-2:]) + "\n")
    outfile_f.write(repr(results) + "\n")
    print(results[0]["ALL"])
    outfile_f.close()