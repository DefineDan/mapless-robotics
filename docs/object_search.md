# Object search

Pass `--object-query` to `auto_explore_navigate` to search for an object automatically.

```bash
python -m mapless_robotics.auto_explore_navigate --index 10 --depth-source habitat \
  --object-query "a place to sit"
```

## Query translation (Qwen)
[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) translates the request once at
startup into short detector labels (e.g. `chair`, `sofa`, `bench`). It runs on CPU by
default (`--query-device cuda` to change that) and is unloaded before exploration starts.
The labels are printed and shown in Rerun. Check them, since the model can misread a request.

`--no-qwen` skips this step and passes the query directly to OWLv2, e.g. `--object-query "bed" --no-qwen`.

## Detection (OWLv2)
[OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2) runs on the RGB captures
in its own worker, so it never delays frontier updates. It keeps one capture in flight and
skips new ones while busy.

| Option | Default | Notes |
|---|---|---|
| `--object-model` | `google/owlv2-base-patch16-ensemble` | Model ID or local directory |
| `--object-device` | `auto` | CUDA if available (half precision), else CPU |
| `--object-image-size` | native | Try `512` on CPU. Smaller inputs can miss small objects |
| `--object-threshold` | 0.2 | Uncalibrated detection score |
| `--object-confirmations` | 3 | Consecutive captures that must agree |
| `--object-consistency-radius` | 0.75 m | How close those detections must be |
| `--object-stand-off` | 1.0 m | Approach distance from the object surface |

Models are cached in `models/huggingface/`, which needs internet on the first run.

## Confirmation and approach
- A detection is placed in 3D using the median depth at the box center. Missing depth
  or objects beyond 5 m can't confirm a target.
- Once confirmed, the object takes priority over frontiers. The robot drives to a point
  `--object-stand-off` from the object, faces it, checks with a fresh detection, and pauses.
- `Enter` starts LoTIS retrieval. `o` dismisses the object and resumes exploration.
  A blocked approach stops until a path appears or you dismiss the object.

Rerun shows **Object detections** (boxes), a yellow object marker with a green approach
point in the graph, and an **Object search** status panel. Results are saved in
`route.json` under `object_search`.
