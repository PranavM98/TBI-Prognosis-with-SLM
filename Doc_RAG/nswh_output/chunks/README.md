# RAG Dataset

Generated from: `nswh_output/nswh_text.txt`

Total chunks: 136

## Dataset Statistics

- **Total Chunks**: 136
- **Unique Pages**: 77
- **Page Range**: 5-117
- **Sides Present**: ['RIGHT', 'LEFT']
- **Avg Word Count**: 281.75
- **Avg Char Count**: 1842.4264705882354
- **Min Word Count**: 14
- **Max Word Count**: 676
- **Total Words**: 38318

## Available Formats

- **JSON**: `nswh_output/chunks/chunks.json`
- **JSONL**: `nswh_output/chunks/chunks.jsonl`
- **CSV**: `nswh_output/chunks/chunks.csv`
- **TXT**: `nswh_output/chunks/txt_files`
- **MARKDOWN**: `nswh_output/chunks/chunks.md`

## Usage Examples

### Load JSON chunks
```python
import json
with open('nswh_output/chunks/chunks.json', 'r') as f:
    chunks = json.load(f)
```

### Load JSONL chunks (streaming)
```python
import json
chunks = []
with open('nswh_output/chunks/chunks.jsonl', 'r') as f:
    for line in f:
        chunks.append(json.loads(line))
```
