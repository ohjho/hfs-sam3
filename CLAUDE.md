# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Hugging Face Space (Gradio SDK) that runs **SAM3 video segmentation on ZeroGPU**. Users give a video + text prompt; the model tracks/segments matching objects across frames. Deployed at `huggingface.co/spaces/GF-John/sam3`.

## Run / develop

```bash
uv sync                 # install deps from pyproject.toml into .venv
uv run python app.py    # launch the Gradio app locally (serves UI + MCP server + /docs)
```

`ffmpeg_extractor.py` is also a Typer CLI — useful for debugging frame extraction independently of the model:

```bash
uv run python ffmpeg_extractor.py get-video-metadata <video>
uv run python ffmpeg_extractor.py extract-frames <video> --fps 8 --output-dir ./frames
```

There are no tests or linters configured.

## Deployment (read before touching deps or CI)

Push to `main` → `.github/workflows/deploy_to_hf_space.yaml` **force-pushes** the repo to the HF Space (requires `HF_TOKEN` secret; `FORCE_PUSH` toggles `-f`). Treat the HF Space as a mirror of `main` with this repo as sole source.

**Dependency source of truth is `requirements.txt`, not `pyproject.toml`.** The CI only auto-generates `requirements.txt` from `pyproject.toml` when `requirements.txt` is *absent*; since it's committed, CI uses it verbatim and `pyproject.toml` is effectively ignored for deployment. The two are intentionally out of sync. `requirements.txt` pins `transformers==5.5.3` — the first released version that ships the SAM3 classes (`Sam3VideoModel`, `Sam3Model`, `Sam3TrackerModel`, etc.); earlier it installed from git because SAM3 was unreleased, but pinning to a release avoids surprise breakage from moving `main`. Do **not** add `kernels` — a newer `kernels` release breaks transformers' `hub_kernels.py` import (`LayerRepository` now requires a `revision`/`version`, which transformers doesn't pass), crashing the Space at startup; transformers degrades fine without it. If you add a runtime dependency, edit `requirements.txt`.

The HF Space header config lives in `README.md` frontmatter (`sdk: gradio`, `app_file: app.py`, etc.).

## Architecture

`app.py` is the only deployed entrypoint. Flow of `video_inference()`:

1. **Frame extraction** (`ffmpeg_extractor.extract_frames`) — uses ffmpeg piping raw RGB to stdout rather than OpenCV; downscales via `max_short_edge` and can burn in timestamp/frame-number overlays (disabled in the app path). Extraction runs at `calc_effective_fps(vid_fps, sample_fps, every_x)` — the optional `sample_fps`/`every_x` params downsample the video before tracking, so all output frame indices are positions in the *sampled* sequence (documented in the MCP docstrings).
2. **Model inference** — `Sam3VideoProcessor.init_video_session` → `add_text_prompt` → iterate `Sam3VideoModel.propagate_in_video_iterator`, post-processing each frame to masks + `object_ids`. The session keeps frame preprocessing/storage on `video_load_device` ("cpu" default) and inference state on CPU — transformers otherwise defaults everything to `inference_device`, and normalizing/storing a long video on the ZeroGPU slice OOMs; `get_frame` streams each frame to the GPU on access.
3. **Output** — two modes from the same function via `annotation_mode`:
   - default → list of per-object-per-frame detection dicts: normalized `x/y/w/h` bbox (from `visualizer.mask_to_xyxy`), `track_id`, and `mask_b64` (1-bit PNG base64, via `toolbox.mask_encoding.b64_mask_encode`). See `example_output.json`.
   - `annotation_mode=True` → renders mask overlays per frame (`apply_mask_overlay`, colored by `object_id`) and muxes back to mp4 (`frames_to_vid`).

The Gradio UI (two tabs: "Video-Object Tracking" → JSON, "Video Annotation" → video) wraps these as separate `api_name`s.

**ZeroGPU GPU allocation:** `@spaces.GPU(duration=...)` takes a *callable* (`calc_timeout_duration`) that introspects `video_inference`'s signature and binds the incoming args — this is how the per-request GPU lease length is set dynamically. Keep that signature/binding in sync if you change `video_inference`'s parameters. An explicit `timeout_duration` is a hard override; otherwise the duration is estimated in the main process (unbilled) from the ffprobe'd video length and the downsampling params (`GPU_DURATION_*` constants: per-frame cost, overhead, 30–300s clamp, 120s fallback when the probe fails). Tune `GPU_DURATION_PER_FRAME_S` from Space logs — an underestimate kills the task mid-run, an overestimate only raises the quota gate.

Model + processor load **once at import time** into module globals (`VID_MODEL`, `VID_PROCESSOR`); on failure they're set to `None` and every request asserts against that. `DTYPE` is `bfloat16` on capable CUDA, else `float16`, else CPU.

## Not part of the deployed app

- `example.py` — standalone reference demo with extra tabs (image segmentation, click-to-segment) using `Sam3Model`/`Sam3TrackerModel`. Not imported by `app.py`; kept as a richer example. Note its `apply_mask_overlay` is an older copy of the one in `app.py`.
- `visualizer.py` — only `mask_to_xyxy` is used by `app.py`; the rest (`annotate_detections`, `annotate_masks`) references helpers (`im_draw_bbox`, `mcolors`) that are **not imported**, so those functions are currently dead/broken.
