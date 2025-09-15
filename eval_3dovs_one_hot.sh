for scene in bed bench blue_sofa covered_desk lawn office_desk room snacks sofa table; do
    python eval_3dovs_one_hot.py --data-dir ./data/3DOVS/$scene --checkpoint results/3DOVS/$scene/ckpts/ckpt_29999_rank0.pt  --results-dir ./results/3DOVS_eval/$scene --feature_path ./results/3DOVS_feats/$scene/features_one_hot_3dovs_$scene.pt
done