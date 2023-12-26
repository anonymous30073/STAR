import os

eval_modes = ["standard", "SB_BM25", "popular"]
dss = ["games", "books", "tv"]

anchor = "lsr_distil"

for ds in dss:
    DS = "data/" + ds
    exp_dir = "output/" + ds

    for eval_mode in eval_modes:
        res_file = f"results_{ds}_{eval_mode}.txt"
        open(res_file, "w")
        anchor_folder = None
        for fo in os.listdir(exp_dir):
            if anchor not in fo:
                continue
            else:
                anchor_folder = exp_dir + fo
            for fi in os.listdir(exp_dir + fo):
               if fi.startswith(f"test_predicted_test_neg_{eval_mode}_100_best_model"):
                    print(os.path.join(exp_dir, fo))
                    pred_path = os.path.join(exp_dir, fo, fi)
                    comm = f"python calc_results.py --pos_gt_path {DS}/test.csv --neg_gt_path {DS}/test_neg_{eval_mode}_100.csv  --train_file_path {DS}/train.csv --pred_path {pred_path} --out_path {res_file}"
                    os.system(comm)
                    break

        for fo in os.listdir(exp_dir):
            if anchor in fo:
                continue
            for fi in os.listdir(exp_dir + fo):
               if fi.startswith(f"test_predicted_test_neg_{eval_mode}_100_best_model"):
                    print(os.path.join(exp_dir, fo))
                    pred_path = os.path.join(exp_dir, fo, fi)
                    comm = f"python calc_results.py --pos_gt_path {DS}/test.csv --neg_gt_path {DS}/test_neg_{eval_mode}_100.csv  --train_file_path {DS}/train.csv --pred_path {pred_path} --out_path {res_file} --anchor_path {anchor_folder}"
                    os.system(comm)
                    break

