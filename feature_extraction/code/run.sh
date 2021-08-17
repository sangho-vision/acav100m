CURRENT=$(dirname "$0")
TAR="$CURRENT/../../data/videos/shard-000000.tar"
if [ ! -f $(realpath $TAR) ]; then
  bash $CURRENT/bundle.sh "$CURRENT" "$CURRENT/../../data/videos/clips"
fi
python $CURRENT/cli.py extract --tar_path=$TAR --out_path="$CURRENT/../../data/features"
