# RAG Dataset

Generated from: `btf_output/btf_text.txt`

Total chunks: 15

## Dataset Statistics

- **Total Chunks**: 15
- **Unique Pages**: 8
- **Page Range**: 1-8
- **Sides Present**: ['RIGHT', 'LEFT']
- **Avg Word Count**: 378.93333333333334
- **Avg Char Count**: 2554.2
- **Min Word Count**: 6
- **Max Word Count**: 638
- **Total Words**: 5684

## Available Formats

- **JSON**: `btf_output/chunks/chunks.json`
- **JSONL**: `btf_output/chunks/chunks.jsonl`
- **CSV**: `btf_output/chunks/chunks.csv`
- **TXT**: `btf_output/chunks/txt_files`
- **MARKDOWN**: `btf_output/chunks/chunks.md`

## Usage Examples

### Load JSON chunks
```python
import json
with open('btf_output/chunks/chunks.json', 'r') as f:
    chunks = json.load(f)
```

### Load JSONL chunks (streaming)
```python
import json
chunks = []
with open('btf_output/chunks/chunks.jsonl', 'r') as f:
    for line in f:
        chunks.append(json.loads(line))
```
