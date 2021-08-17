# Clustering with Extracted Features

## Command

1. Run the commandline interface.

```bash
python cli.py cluster --feature_path="<path to feature directory>/shard-{shard_range}.pkl" \
  --out_path='<path to cluster directory> --meta_path=<path to video directory>'
```

## File Structures

### Logs
```
path: {out_path}/{filename}
filename: 'log_{hostname}_{main process pid}_{timestamp}.json'
file structure:
{
 'hostname': hostname,
 'pid': pid,
 'timestamp': timestamp,
 'time': time e.g.) 2020-10-17 -,
 'shards': [shard_list]
}
```
