## Sample training run:
```console
python run.py --mode train --data_dir .\dataset\ --epochs 100 --batch_size 4
```

## Sample query run:
```console
python run.py --mode query --query_image .\dataset\11\images\unnamed_2.png --query_type part --exclude_query_assembly --max_parts_per_assembly 2
```