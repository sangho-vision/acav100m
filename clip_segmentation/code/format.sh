VIDEOS=$1
OUT="$VIDEOS/../clips"
mv "$VIDEOS/clips_diversity_greedy" $OUT
find $OUT -mindepth 2 -type f -exec mv -t $OUT -i '{}' +  # flatten dir
rm -rf $OUT/*/
