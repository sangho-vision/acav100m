CURRENT=$(dirname "$0")
python $CURRENT/run.py --data-path "$CURRENT/../../data/videos/raw"
bash $CURRENT/format.sh $CURRENT/../../data/videos/raw
