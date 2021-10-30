# attention preserves L1 norm...

To setup on SCC, first load modules:

```
module load python3/3.8.10
module load pytorch/1.9.0
```

Now, setup virtual environment:

```
[ ! -d "env" ] && virtualenv --system-site-packages env
source env/bin/activate
pip install -r requirements.txt
```