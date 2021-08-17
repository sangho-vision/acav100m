CURRENT=$(dirname "$0")
CLUSTERS="$CURRENT/../../data/clusters/shard-000000.pkl"
python $CURRENT/cli.py run --shards_path=$CLUSTERS \
  --meta_path="$CURRENT/../../data/videos" \
  --out_path="$CURRENT/../../data/output.csv"
