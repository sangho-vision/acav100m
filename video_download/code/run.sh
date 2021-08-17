CURRENT=$(dirname "$0")
python $CURRENT/run.py --data-path "$CURRENT/../../data/filtered.tsv" --output-dir "$CURRENT/../../data/videos/raw"
