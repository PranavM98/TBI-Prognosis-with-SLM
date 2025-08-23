# RAG Dataset

Generated from: `nhice_output/nhice_text.txt`

Total chunks: 340

## Dataset Statistics

- **Total Chunks**: 340
- **Unique Pages**: 192
- **Page Range**: 12-203
- **Sides Present**: ['RIGHT', 'LEFT']
- **Avg Word Count**: 217.27941176470588
- **Avg Char Count**: 1425.15
- **Min Word Count**: 13
- **Max Word Count**: 784
- **Total Words**: 73875

## Available Formats

- **JSON**: `nhice_output/chunks/chunks.json`
- **JSONL**: `nhice_output/chunks/chunks.jsonl`
- **CSV**: `nhice_output/chunks/chunks.csv`
- **TXT**: `nhice_output/chunks/txt_files`
- **MARKDOWN**: `nhice_output/chunks/chunks.md`

## Usage Examples

### Load JSON chunks
```python
import json
with open('nhice_output/chunks/chunks.json', 'r') as f:
    chunks = json.load(f)
```

### Load JSONL chunks (streaming)
```python
import json
chunks = []
with open('nhice_output/chunks/chunks.jsonl', 'r') as f:
    for line in f:
        chunks.append(json.loads(line))
```
