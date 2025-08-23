# RAG Dataset

Generated from: `scn_output/scn_text.txt`

Total chunks: 23

## Dataset Statistics

- **Total Chunks**: 23
- **Unique Pages**: 12
- **Page Range**: 2-13
- **Sides Present**: ['RIGHT', 'LEFT']
- **Avg Word Count**: 320.60869565217394
- **Avg Char Count**: 2136.4347826086955
- **Min Word Count**: 108
- **Max Word Count**: 588
- **Total Words**: 7374

## Available Formats

- **JSON**: `scn_output/chunks/chunks.json`
- **JSONL**: `scn_output/chunks/chunks.jsonl`
- **CSV**: `scn_output/chunks/chunks.csv`
- **TXT**: `scn_output/chunks/txt_files`
- **MARKDOWN**: `scn_output/chunks/chunks.md`

## Usage Examples

### Load JSON chunks
```python
import json
with open('scn_output/chunks/chunks.json', 'r') as f:
    chunks = json.load(f)
```

### Load JSONL chunks (streaming)
```python
import json
chunks = []
with open('scn_output/chunks/chunks.jsonl', 'r') as f:
    for line in f:
        chunks.append(json.loads(line))
```
