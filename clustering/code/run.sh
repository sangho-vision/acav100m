CURRENT=$(dirname "$0")
FEATURES="$CURRENT/../../data/features/shard-000000.pkl"
python $CURRENT/cli.py cluster --feature_path=$FEATURES --out_path="$CURRENT/../../data/clusters" \
  --meta_path="$CURRENT/../../data/videos"
