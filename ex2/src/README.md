# Assignment 2 - ASR

Audio segmentation and ASR assignment for Advanced Topics in Audio Processing using Deep Learning.

## Directory Structure

```
Samples/
├── Raw/                 # Original unsegmented recordings (48kHz)
│   ├── Adam.wav
│   ├── Gal.wav
│   └── ...
└── Segmented/           # Segmented files (16kHz)
    ├── Adam/
    │   ├── segment_00.wav   # Rename to 0.wav, 1.wav, ..., 9.wav, random.wav
    │   └── ...
    └── ...
```

## Scripts

### play_segments.py
Play all segments to verify segmentation quality.

```bash
python3 play_segments.py
```

## Segmentation Parameters

| Speaker | top_db | min_gap | extend_ms |
|---------|--------|---------|-----------|
| Gal     | 25     | 0.3s    | 50ms      |
| Hagar   | 25     | 0.3s    | 50ms      |
| Roy    | 25     | 0.3s    | 50ms      |
| Ofir    | 25     | 0.3s    | 50ms      |
| Adam    | 25     | 0.3s    | 150ms     |
| Nirit   | 20     | 0.15s   | 50ms      |
| Ido     | 25     | 0.1s    | 50ms      |
| Shir    | 30     | 0.1s    | 50ms      |
| Inbar   | 25     | 0.15s   | 150ms     |

## Speaker Split

| Role | Speakers |
|------|----------|
| Class Representative (Reference DB) | Gal |
| Training (2M + 2F) | Adam, Ido, Hagar, Inbar |
| Evaluation (2M + 2F) | Roy, Ofir, Nirit, Shir |

## File Naming

Segments are named `segment_00.wav` to `segment_10.wav` and correspond to:
- `segment_00.wav` - `segment_09.wav`: digits 0-9
- `segment_10.wav`: random word

## Next Steps

1. Extract Mel Spectrograms (25ms window, 10ms hop, 80 filter banks)
2. Implement DTW algorithm
3. Implement CTC forward algorithm
