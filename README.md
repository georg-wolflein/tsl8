tsl8
====

*tsl8*, pronounced "tesselate" is a parallelised solution for reading whole slide images (WSIs), splitting them into background patches, and rejecting background tiles.

# Installation
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Running
```
source env/bin/activate
python -m tsl8.extract_tiles
```