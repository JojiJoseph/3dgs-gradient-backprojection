for scene in bed bench blue_sofa covered_desk lawn office_desk room snacks sofa table; do
    python backproject.py --data-dir data/3DOVS/$scene --checkpoint results/3DOVS/$scene/ckpts/ckpt_29999_rank0.pt --feature clip-text-3dovs --results-dir results/3DOVS_feats/$scene --tag $scene
done
