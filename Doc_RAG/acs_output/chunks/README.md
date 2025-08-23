# RAG Dataset

Generated from: `acs_output/acs_text.txt`

Total chunks: 135

## Dataset Statistics

- **Total Chunks**: 135
- **Unique Pages**: 79
- **Page Range**: 5-99
- **Sides Present**: ['RIGHT', 'LEFT']
- **Avg Word Count**: 242.13333333333333
- **Avg Char Count**: 1726.3111111111111
- **Min Word Count**: 10
- **Max Word Count**: 476
- **Total Words**: 32688

## Available Formats

- **JSON**: `acs_output/chunks/chunks.json`
- **JSONL**: `acs_output/chunks/chunks.jsonl`
- **CSV**: `acs_output/chunks/chunks.csv`
- **TXT**: `acs_output/chunks/txt_files`
- **MARKDOWN**: `acs_output/chunks/chunks.md`

## Usage Examples

### Load JSON chunks
```python
import json
with open('acs_output/chunks/chunks.json', 'r') as f:
    chunks = json.load(f)
```

### Load JSONL chunks (streaming)
```python
import json
chunks = []
with open('acs_output/chunks/chunks.jsonl', 'r') as f:
    for line in f:
        chunks.append(json.loads(line))
```
