echo "exp1"
python grid_search.py -g search_targets/experiments/exp1_baselines.json -l
echo "exp2"
python grid_search.py -g search_targets/experiments/exp2_batch.json -l
echo "exp3"
python grid_search.py -g search_targets/experiments/exp3_ncentroids.json -l
echo "exp4"
python grid_search.py -g search_targets/experiments/exp4_parallelization.json -l
echo "exp5"
python grid_search.py -g search_targets/experiments/exp5_pairing.json -l
echo "exp6"
python grid_search.py -g search_targets/experiments/exp6_sgd_kmeans.json -l
echo "algorithms"
python grid_search.py -g search_targets/algorithms/contrastive.json -l
python grid_search.py -g search_targets/algorithms/ours.json -l
python grid_search.py -g search_targets/algorithms/ours_split.json -l
python grid_search.py -g search_targets/algorithms/pca.json -l
python grid_search.py -g search_targets/algorithms/penultimate.json -l
