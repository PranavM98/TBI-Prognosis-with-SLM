# RAG Dataset

Generated from: `cma_output/cma_text.txt`

Total chunks: 15

## Dataset Statistics

- **Total Chunks**: 15
- **Unique Pages**: 6
- **Page Range**: 1-6
- **Sides Present**: ['RIGHT', 'LEFT', 'MIDDLE']
- **Avg Word Count**: 195.93333333333334
- **Avg Char Count**: 1320.8
- **Min Word Count**: 22
- **Max Word Count**: 349
- **Total Words**: 2939

## Available Formats

- **JSON**: `cma_output/chunks.json`
- **JSONL**: `cma_output/chunks.jsonl`
- **CSV**: `cma_output/chunks.csv`
- **TXT**: `cma_output/txt_files`
- **MARKDOWN**: `cma_output/chunks.md`

## Usage Examples

### Load JSON chunks
```python
import json
with open('cma_output/chunks.json', 'r') as f:
    chunks = json.load(f)
```

### Load JSONL chunks (streaming)
```python
import json
chunks = []
with open('cma_output/chunks.jsonl', 'r') as f:
    for line in f:
        chunks.append(json.loads(line))
```
