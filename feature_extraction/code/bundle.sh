CURRENT=$1
DATADIR=$2
WORKDIR=$PWD
cd $DATADIR
tar -cf "shard-000000.tar" *.mp4
mv "shard-000000.tar" ../
cd $WORKDIR
python $CURRENT/build_metadata.py --clip-path $DATADIR
