# Import helpers for mask encoding and bbox extraction
import inspect
import json
import math
import sys
import tempfile

import cv2
import gradio as gr
import matplotlib
import numpy as np
import spaces
import torch
from loguru import logger
from PIL import Image
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
    Sam3VideoModel,
    Sam3VideoProcessor,
)

# Import ffmpeg_extractor helpers
from ffmpeg_extractor import extract_frames, get_video_metadata

# import local helpers
from toolbox.mask_encoding import b64_mask_decode, b64_mask_encode
from visualizer import mask_to_xyxy

logger.remove()
logger.add(
    sys.stderr,
    format="<d>{time:YYYY-MM-DD ddd HH:mm:ss}</d> | <lvl>{level}</lvl> | <lvl>{message}</lvl>",
)

# Set target DEVICE and DTYPE
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE}, dtype: {DTYPE}")
logger.info("Loading Models and Processors...")
try:
    VID_MODEL = Sam3VideoModel.from_pretrained("facebook/sam3").to(DEVICE, dtype=DTYPE)
    VID_PROCESSOR = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    logger.success("Models and Processors Loaded!")
except Exception as e:
    logger.error(f"❌ CRITICAL ERROR LOADING VIDEO MODELS: {e}")
    VID_MODEL = None
    VID_PROCESSOR = None

try:
    # Text-prompt image segmentation (concept segmentation)
    IMG_MODEL = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
    IMG_PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")
    # Visual-prompt image segmentation (points/boxes, SAM2-style)
    TRK_MODEL = Sam3TrackerModel.from_pretrained("facebook/sam3").to(DEVICE)
    TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    logger.success("Image Models and Processors Loaded!")
except Exception as e:
    logger.error(f"❌ CRITICAL ERROR LOADING IMAGE MODELS: {e}")
    IMG_MODEL = None
    IMG_PROCESSOR = None
    TRK_MODEL = None
    TRK_PROCESSOR = None

try:
    # Visual-prompt video tracking (mask prompts, SAM2-style PVS); same checkpoint —
    # the class pulls the tracker weights out and drops the detector head.
    TRK_VID_MODEL = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(
        DEVICE, dtype=DTYPE
    )
    TRK_VID_PROCESSOR = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    logger.success("Tracker Video Model and Processor Loaded!")
except Exception as e:
    logger.error(f"❌ CRITICAL ERROR LOADING TRACKER VIDEO MODEL: {e}")
    TRK_VID_MODEL = None
    TRK_VID_PROCESSOR = None


def apply_mask_overlay(base_image, mask_data, object_ids=None, opacity=0.5):
    """Draws segmentation masks on top of an image, using object IDs for coloring."""
    if isinstance(base_image, np.ndarray):
        base_image = Image.fromarray(base_image)
    base_image = base_image.convert("RGBA")

    if mask_data is None or len(mask_data) == 0:
        return base_image.convert("RGB")

    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.cpu().numpy()
    mask_data = mask_data.astype(np.uint8)

    # Handle dimensions
    if mask_data.ndim == 4:
        mask_data = mask_data[0]
    if mask_data.ndim == 3 and mask_data.shape[0] == 1:
        mask_data = mask_data[0]

    num_masks = mask_data.shape[0] if mask_data.ndim == 3 else 1
    if mask_data.ndim == 2:
        mask_data = [mask_data]
        num_masks = 1

    # Use object_ids for coloring if provided, else fallback to index
    if object_ids is not None and len(object_ids) == num_masks:
        # Use a fixed color map and assign color based on object_id
        try:
            color_map = matplotlib.colormaps["rainbow"]
        except AttributeError:
            import matplotlib.cm as cm

            color_map = cm.get_cmap("rainbow")
        # Normalize object_ids to a color index (e.g., mod by 256)
        unique_ids = sorted(set(object_ids))
        id_to_color_idx = {oid: i for i, oid in enumerate(unique_ids)}
        rgb_colors = [
            tuple(
                int(c * 255)
                for c in color_map(id_to_color_idx[oid] / max(len(unique_ids), 1))[:3]
            )
            for oid in object_ids
        ]
    else:
        try:
            color_map = matplotlib.colormaps["rainbow"].resampled(max(num_masks, 1))
        except AttributeError:
            import matplotlib.cm as cm

            color_map = cm.get_cmap("rainbow").resampled(max(num_masks, 1))
        rgb_colors = [
            tuple(int(c * 255) for c in color_map(i)[:3]) for i in range(num_masks)
        ]

    composite_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))

    for i, single_mask in enumerate(mask_data):
        mask_bitmap = Image.fromarray((single_mask * 255).astype(np.uint8))
        if mask_bitmap.size != base_image.size:
            mask_bitmap = mask_bitmap.resize(base_image.size, resample=Image.NEAREST)

        fill_color = rgb_colors[i]
        color_fill = Image.new("RGBA", base_image.size, fill_color + (0,))
        mask_alpha = mask_bitmap.point(lambda v: int(v * opacity) if v > 0 else 0)
        color_fill.putalpha(mask_alpha)
        composite_layer = Image.alpha_composite(composite_layer, color_fill)

    return Image.alpha_composite(base_image, composite_layer).convert("RGB")


def frames_to_vid(pil_frames, output_path: str, vid_fps: int, vid_w: int, vid_h: int):
    assert len(pil_frames) > 0, f"Number of frames must be greater than 0"
    assert isinstance(pil_frames, list), f"pil_frames must be a list"
    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), vid_fps, (vid_w, vid_h)
    )
    for f in pil_frames:
        video_writer.write(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
    video_writer.release()
    return output_path


def calc_effective_fps(vid_fps: float, sample_fps: float = None, every_x: int = None):
    """FPS the video is actually sampled at after the optional downsampling filters."""
    # empty/0 gr.Number must not activate the filters
    sample_fps = sample_fps or None
    every_x = int(every_x) if every_x else None
    if sample_fps:
        return min(float(sample_fps), vid_fps)
    if every_x:
        return vid_fps / every_x
    return vid_fps


# Rough GPU seconds per propagated frame for SAM3, incl. CPU->GPU frame streaming and mask
# post-processing. Tune from Space logs: an underestimate kills the task mid-run (wasting
# the whole billed window), an overestimate only raises the quota gate / queue penalty for
# callers.
GPU_DURATION_PER_FRAME_S = 0.5
GPU_DURATION_OVERHEAD_S = 20  # ffmpeg extraction + session preprocessing
GPU_DURATION_MIN_S = 30
GPU_DURATION_MAX_S = 300  # videos estimated above this should be downsampled instead
GPU_DURATION_FALLBACK_S = 120  # metadata probe failed


def _estimate_gpu_duration(
    input_video, timeout_duration=None, sample_fps=None, every_x=None, extra_steps=0
):
    """Estimate the GPU lease from the probed video length and downsampling; runs in the
    main process before the GPU is requested (probe time is not billed)."""
    if timeout_duration:  # explicit user value is a hard override
        return int(timeout_duration)
    try:
        video_path = (
            input_video
            if isinstance(input_video, str)
            else (input_video or {}).get("name")
        )
        vmeta = get_video_metadata(video_path, bverbose=False) if video_path else None
    except Exception as e:
        logger.warning(f"video probe failed ({e}); falling back to fixed GPU duration")
        vmeta = None
    if not vmeta or not vmeta.get("fps") or not vmeta.get("duration"):
        return GPU_DURATION_FALLBACK_S
    effective_fps = calc_effective_fps(
        vmeta["fps"], sample_fps=sample_fps, every_x=every_x
    )
    steps = math.ceil(vmeta["duration"] * effective_fps) + extra_steps
    est = GPU_DURATION_OVERHEAD_S + steps * GPU_DURATION_PER_FRAME_S
    duration = max(GPU_DURATION_MIN_S, min(GPU_DURATION_MAX_S, math.ceil(est)))
    logger.info(f"requesting {duration}s of GPU time for ~{steps} propagation steps")
    return duration


def calc_timeout_duration(input_video, *args, **kwargs):
    # spaces calls this with video_inference's args — keep the binding in sync with its
    # signature.
    sig = inspect.signature(video_inference)
    bound = sig.bind(input_video, *args, **kwargs)
    bound.apply_defaults()
    return _estimate_gpu_duration(
        input_video,
        timeout_duration=bound.arguments.get("timeout_duration"),
        sample_fps=bound.arguments.get("sample_fps"),
        every_x=bound.arguments.get("every_x"),
    )


def calc_visual_timeout_duration(input_video, *args, **kwargs):
    # spaces calls this with video_visual_inference's args — keep the binding in sync with
    # its signature.
    sig = inspect.signature(video_visual_inference)
    bound = sig.bind(input_video, *args, **kwargs)
    bound.apply_defaults()
    return _estimate_gpu_duration(
        input_video,
        timeout_duration=bound.arguments.get("timeout_duration"),
        sample_fps=bound.arguments.get("sample_fps"),
        every_x=bound.arguments.get("every_x"),
        # a nonzero ref frame adds the reverse pass (ref..0) on top of the forward one
        extra_steps=int(bound.arguments.get("ref_frame_idx") or 0),
    )


# Our Inference Function
@spaces.GPU(duration=calc_timeout_duration)
def video_inference(
    input_video,
    prompt: str,
    timeout_duration: int = None,
    sample_fps: float = None,
    every_x: int = None,
    video_load_device: str = "cpu",
    annotation_mode: bool = False,
) -> list[dict] | str:
    """Track and segment objects across a video with SAM3 using a natural-language text prompt, returning per-object-per-frame detections (or an annotated video).

    The prompt is a concept to find (e.g. "player in white", "red car"); every matching instance is tracked across all frames with a stable track_id. The video can optionally be downsampled before tracking with sample_fps or every_x; the "frame" field of every detection is a 0-based position in the SAMPLED frame sequence, NOT an original video frame number, so with every_x=5 the detection "frame": 2 corresponds to original frame 10. This tool has two output shapes selected by annotation_mode. When annotation_mode is false (default) it returns a JSON list of detection objects, one per tracked object per frame. Each detection has: "frame" (integer sampled-frame index as defined above); "track_id" (integer, stable across frames for one object); "x", "y", "w", "h" (floats); "conf" (float, always 1 for video); and "mask_b64" (string). The bounding box ("x","y","w","h") is NORMALIZED 0-1: (x, y) is the top-left corner and (w, h) is the box size, each divided by the video width/height -- i.e. the albumentations "coco" layout [x_min, y_min, width, height] but normalized to 0-1 rather than absolute pixels, and NOT the "pascal_voc" [x_min, y_min, x_max, y_max] layout. Bounding-box formats are documented at https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats . "mask_b64" is a base64-encoded 1-bit PNG of the binary segmentation mask at the frame resolution; decode it with PIL, e.g. numpy.array(Image.open(io.BytesIO(base64.b64decode(mask_b64)))) to get a 0/255 mask. When annotation_mode is true it instead returns a filesystem path to an mp4 video with the colored mask overlays burned in.

    Args:
        input_video: The input video to segment (a file path, or an uploaded-file object with a "name" key).
        prompt: Natural-language description of the object(s) to track/segment, e.g. "player in white".
        timeout_duration: Max GPU lease in seconds for this request, used as a hard override; leave empty to auto-estimate it from the video length and downsampling settings.
        sample_fps: Sample the video down to roughly this many frames per second before tracking (clamped to the source fps; takes precedence over every_x); leave empty to keep every frame.
        every_x: Keep only every Nth frame of the video before tracking, e.g. 5 keeps original frames 0, 5, 10, ... (ignored when sample_fps is set); leave empty to keep every frame.
        video_load_device: Device the video frames are preprocessed and stored on: "cpu" (default) streams frames to the GPU one at a time and keeps GPU memory low; "cuda" holds the whole preprocessed video in GPU memory, which is faster per frame but runs out of GPU memory on long videos.
        annotation_mode: If false (default) return the JSON detections list; if true return a path to an annotated mp4 with mask overlays.
    """
    assert type(VID_MODEL) != type(None) and type(VID_PROCESSOR) != type(
        None
    ), "Video Models failed to load on startup."
    assert input_video and prompt, "Missing video or prompt."

    # Gradio passes a dict with 'name' key for uploaded files
    video_path = (
        input_video if isinstance(input_video, str) else input_video.get("name", None)
    )
    assert video_path, "Invalid video input."

    # Use FFmpeg-based helpers for metadata and frame extraction
    vmeta = get_video_metadata(video_path, bverbose=False)
    assert vmeta, "Failed to extract video metadata."
    vid_fps = vmeta["fps"]
    vid_w = vmeta["width"]
    vid_h = vmeta["height"]
    effective_fps = calc_effective_fps(vid_fps, sample_fps=sample_fps, every_x=every_x)

    # Extract frames as PIL Images (no timestamp/frame_num overlays)
    pil_frames = extract_frames(
        video_path,
        fps=effective_fps,
        max_short_edge=min(vid_w, vid_h),
        write_timestamp=False,
        write_frame_num=False,
        output_dir=None,
    )
    assert len(pil_frames) > 0, "No frames found in video."

    # Convert PIL Images to numpy arrays (RGB)
    video_frames = [np.array(frame.convert("RGB")) for frame in pil_frames]

    session = VID_PROCESSOR.init_video_session(
        video=video_frames,
        inference_device=DEVICE,
        # Preprocessing/storing the full video on cuda (the transformers default when only
        # inference_device is set) OOMs on long videos — normalizing hundreds of frames
        # needs tens of GB transiently. Keep frames on video_load_device ("cpu" by
        # default); get_frame streams each one to inference_device on access.
        processing_device=video_load_device,
        video_storage_device=video_load_device,
        # Per-frame outputs accumulate over the whole video — park them in RAM as well.
        inference_state_device="cpu",
        dtype=DTYPE,
    )
    session = VID_PROCESSOR.add_text_prompt(inference_session=session, text=prompt)
    temp_out_path = tempfile.mktemp(suffix=".mp4")

    detections = []
    annotated_frames = []
    for model_out in VID_MODEL.propagate_in_video_iterator(
        inference_session=session, max_frame_num_to_track=len(video_frames)
    ):
        post_processed = VID_PROCESSOR.postprocess_outputs(session, model_out)
        f_idx = model_out.frame_idx
        original_pil = Image.fromarray(video_frames[f_idx])
        if "masks" in post_processed:
            detected_masks = post_processed["masks"]
            object_ids = post_processed["object_ids"]
            object_ids = [int(oid) for oid in object_ids]
            if detected_masks.ndim == 4:
                detected_masks = detected_masks.squeeze(1)

            for i, mask in enumerate(detected_masks):
                mask = mask.cpu().numpy()
                mask_bin = (mask > 0.0).astype(np.uint8)
                xyxy = mask_to_xyxy(mask_bin)
                if not xyxy:
                    continue
                x0, y0, x1, y1 = xyxy
                det = {
                    "frame": f_idx,
                    "track_id": int(object_ids[i]),
                    "x": x0 / vid_w,
                    "y": y0 / vid_h,
                    "w": (x1 - x0) / vid_w,
                    "h": (y1 - y0) / vid_h,
                    "conf": 1,
                    "mask_b64": b64_mask_encode(mask_bin).decode("ascii"),
                }
                detections.append(det)

        if annotation_mode:
            final_frame = (
                apply_mask_overlay(original_pil, detected_masks, object_ids=object_ids)
                if "masks" in post_processed
                else original_pil
            )
            annotated_frames.append(final_frame)

    return (
        frames_to_vid(
            annotated_frames,
            output_path=temp_out_path,
            # mux at the sampled fps so a downsampled video keeps real-time duration
            vid_fps=effective_fps,
            vid_h=vid_h,
            vid_w=vid_w,
        )
        if annotation_mode
        else detections
    )


def video_annotation(
    input_video,
    prompt: str,
    timeout_duration: int = None,
    sample_fps: float = None,
    every_x: int = None,
    video_load_device: str = "cpu",
) -> str:
    """Track and segment objects across a video with SAM3 using a natural-language text prompt, and return an annotated video with the segmentation masks overlaid.

    The prompt is a concept to find (e.g. "player in white", "red car"); every matching instance is tracked across all frames and rendered as a colored mask overlay (color keyed by object). The video can optionally be downsampled before tracking with sample_fps or every_x; the returned video then contains only the sampled frames, muxed at the sampled frame rate so it keeps the original real-time duration. This is the annotated-video counterpart of video_inference: it returns a filesystem path to an mp4 with the mask overlays burned in, rather than JSON detections. Use video_inference instead if you need the structured per-frame bounding boxes and base64 masks.

    Args:
        input_video: The input video to segment (a file path, or an uploaded-file object with a "name" key).
        prompt: Natural-language description of the object(s) to track/segment, e.g. "player in white".
        timeout_duration: Max GPU lease in seconds for this request, used as a hard override; leave empty to auto-estimate it from the video length and downsampling settings.
        sample_fps: Sample the video down to roughly this many frames per second before tracking (clamped to the source fps; takes precedence over every_x); leave empty to keep every frame.
        every_x: Keep only every Nth frame of the video before tracking, e.g. 5 keeps original frames 0, 5, 10, ... (ignored when sample_fps is set); leave empty to keep every frame.
        video_load_device: Device the video frames are preprocessed and stored on: "cpu" (default) streams frames to the GPU one at a time and keeps GPU memory low; "cuda" holds the whole preprocessed video in GPU memory, which is faster per frame but runs out of GPU memory on long videos.
    """
    return video_inference(
        input_video,
        prompt,
        timeout_duration=timeout_duration,
        sample_fps=sample_fps,
        every_x=every_x,
        video_load_device=video_load_device,
        annotation_mode=True,
    )


@spaces.GPU(duration=calc_visual_timeout_duration)
def video_visual_inference(
    input_video,
    masks: str | list,
    drop_masks: bool = False,
    ref_frame_idx: int = 0,
    timeout_duration: int = None,
    sample_fps: float = None,
    every_x: int = None,
    video_load_device: str = "cpu",
) -> list[dict]:
    """SAM3 video segmentation with visual prompts: track objects through a video given their segmentation masks on a reference frame; a drop-in match for SAM2's process_video.

    Seed the objects to track by supplying their masks on the reference frame (ref_frame_idx); SAM3 then propagates each object forward -- and, when ref_frame_idx is nonzero, also backward -- through every frame. Unlike video_inference (which finds objects from a text prompt), this tool tracks exactly the objects whose masks you provide, e.g. masks obtained from image_visual_inference or image_text_inference on the reference frame. The video can optionally be downsampled before tracking with sample_fps or every_x; all frame indices (the "frame" field of every detection AND the ref_frame_idx input) are 0-based positions in the SAMPLED frame sequence, NOT original video frame numbers, so with every_x=5 the detection "frame": 2 corresponds to original frame 10. Returns a JSON list of per-frame detections, one entry per tracked object per frame in which it appears (frames where an object is absent are skipped). Each detection is a dict with: "frame" (integer sampled-frame index as defined above); "track_id" (integer object id matching the position of the seed mask in the masks input, stable across frames); "x", "y", "w", "h" (the object's bounding box as top-left-x, top-left-y, width, height, each NORMALIZED to 0.0-1.0 by dividing by the frame width or height -- this is the albumentations "coco" layout [x_min, y_min, width, height] but normalized to 0-1 rather than absolute pixels, and it is NOT [x_min, y_min, x_max, y_max]; box formats documented at https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats ); "conf" (always 1); and, unless drop_masks is true, "mask_b64" (a base64-encoded 1-bit PNG string the same width and height as the video frame, 1 inside the object and 0 elsewhere).

    Args:
        input_video: The input video to segment (a file path, or an uploaded-file object with a "name" key).
        masks: JSON list of base64-encoded 1-bit PNG masks for the reference frame, one per object to track, e.g. ["b'iVBORw0KGgo...'", ...]; the b'...' literal wrapper is accepted and stripped.
        drop_masks: When true, omit the "mask_b64" field from every detection so only bounding-box information is returned.
        ref_frame_idx: Index of the frame the provided masks correspond to, counted in the SAMPLED frame sequence when sample_fps or every_x is set; a nonzero value triggers bidirectional tracking (forward and backward from this frame).
        timeout_duration: Max GPU lease in seconds for this request, used as a hard override; leave empty to auto-estimate it from the video length, downsampling settings, and ref_frame_idx.
        sample_fps: Sample the video down to roughly this many frames per second before tracking (clamped to the source fps; takes precedence over every_x); leave empty to keep every frame.
        every_x: Keep only every Nth frame of the video before tracking, e.g. 5 keeps original frames 0, 5, 10, ... (ignored when sample_fps is set); leave empty to keep every frame.
        video_load_device: Device the video frames are preprocessed and stored on: "cpu" (default) streams frames to the GPU one at a time and keeps GPU memory low; "cuda" holds the whole preprocessed video in GPU memory, which is faster per frame but runs out of GPU memory on long videos.
    """
    assert TRK_VID_MODEL is not None and TRK_VID_PROCESSOR is not None, (
        "Tracker video model failed to load on startup."
    )
    assert input_video and masks, "Missing video or masks."
    video_path = (
        input_video if isinstance(input_video, str) else input_video.get("name", None)
    )
    assert video_path, "Invalid video input."

    masks = json.loads(masks) if isinstance(masks, str) else masks
    masks = [
        m[2:-1].encode() if m.startswith("b'") and m.endswith("'") else m for m in masks
    ]  # expect the b'' literal to be included
    masks = [b64_mask_decode(m).astype(bool) for m in masks]

    vmeta = get_video_metadata(video_path, bverbose=False)
    assert vmeta, "Failed to extract video metadata."
    vid_fps = vmeta["fps"]
    vid_w = vmeta["width"]
    vid_h = vmeta["height"]
    effective_fps = calc_effective_fps(vid_fps, sample_fps=sample_fps, every_x=every_x)

    pil_frames = extract_frames(
        video_path,
        fps=effective_fps,
        max_short_edge=min(vid_w, vid_h),
        write_timestamp=False,
        write_frame_num=False,
        output_dir=None,
    )
    assert len(pil_frames) > 0, "No frames found in video."
    ref_frame_idx = int(ref_frame_idx or 0)
    assert 0 <= ref_frame_idx < len(pil_frames), (
        f"ref_frame_idx {ref_frame_idx} out of range for "
        f"{len(pil_frames)} sampled frames."
    )
    video_frames = [np.array(frame.convert("RGB")) for frame in pil_frames]

    session = TRK_VID_PROCESSOR.init_video_session(
        video=video_frames,
        inference_device=DEVICE,
        # same OOM guard as video_inference: keep frames + per-frame state off the GPU
        processing_device=video_load_device,
        video_storage_device=video_load_device,
        inference_state_device="cpu",
        dtype=DTYPE,
    )
    # Seed every object in a single call — add_inputs_to_inference_session overwrites
    # session.obj_with_new_inputs on each call, so seeding masks one-per-call would leave
    # only the last object registered as "new".
    TRK_VID_PROCESSOR.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=ref_frame_idx,
        obj_ids=list(range(len(masks))),
        input_masks=masks,
    )

    def frame_detections(model_out):
        pp_masks = TRK_VID_PROCESSOR.post_process_masks(
            [model_out.pred_masks],
            original_sizes=[[session.video_height, session.video_width]],
            binarize=False,
        )[0]  # (num_objects, 1, H, W)
        dets = []
        for i, obj_id in enumerate(session.obj_ids):
            mask_bin = (pp_masks[i, 0].cpu().numpy() > 0.0).astype(np.uint8)
            xyxy = mask_to_xyxy(mask_bin)
            if not xyxy:
                continue
            x0, y0, x1, y1 = xyxy
            det = {
                "frame": model_out.frame_idx,
                "track_id": int(obj_id),
                "x": x0 / vid_w,
                "y": y0 / vid_h,
                "w": (x1 - x0) / vid_w,
                "h": (y1 - y0) / vid_h,
                "conf": 1,
            }
            if not drop_masks:
                det["mask_b64"] = b64_mask_encode(mask_bin).decode("ascii")
            dets.append(det)
        return dets

    detections = []
    for model_out in TRK_VID_MODEL.propagate_in_video_iterator(
        inference_session=session,
        start_frame_idx=ref_frame_idx,
        max_frame_num_to_track=len(video_frames),
    ):
        detections.extend(frame_detections(model_out))
    if ref_frame_idx > 0:
        # backward pass ref..0; the ref frame itself is already covered by the forward pass
        for model_out in TRK_VID_MODEL.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=ref_frame_idx, reverse=True
        ):
            if model_out.frame_idx == ref_frame_idx:
                continue
            detections.extend(frame_detections(model_out))
    detections.sort(key=lambda d: (d["frame"], d["track_id"]))
    return detections


def image_visual_inference(
    im: Image.Image,
    variant=None,
    bboxes=None,
    points=None,
    point_labels=None,
) -> list[str]:
    """Segment objects in a single image with SAM3 using visual prompts (bounding boxes and/or points); a drop-in match for SAM2's process_image.

    Provide at least one of bboxes or points. Each bounding box produces its own mask; all points together describe one object and produce a single mask. The return value is a JSON list of base64-encoded 1-bit-PNG mask strings, ordered as: one mask per bounding box (in the order the boxes were given), then, if points were provided, one final mask for the points prompt. Each string decodes to a binary segmentation mask at the input image resolution; decode it with PIL, e.g. numpy.array(Image.open(io.BytesIO(base64.b64decode(mask_b64)))) to get a 0/255 mask. Box pixel coordinates use (x0, y0) = top-left and (x1, y1) = bottom-right corners of the ORIGINAL input image (albumentations "pascal_voc" [x_min, y_min, x_max, y_max], absolute pixels); point coordinates (x, y) are absolute pixels in the same image. Coordinate formats are documented at https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats .

    Args:
        im: The RGB image to segment (a PIL Image).
        variant: Accepted for SAM2 API parity and ignored; SAM3 has a single model.
        bboxes: Bounding-box prompts as a list of dicts (or a JSON string), each dict {"x0","y0","x1","y1"} in absolute pixels; one output mask per box.
        points: Point prompts as a list of dicts (or a JSON string), each dict {"x","y"} in absolute pixels; all points together define a single object/mask.
        point_labels: List of ints (or a JSON string) parallel to points, 1=foreground and 0=background; required whenever points is given.
    """
    assert TRK_MODEL is not None and TRK_PROCESSOR is not None, (
        "Image tracker model failed to load on startup."
    )

    # input validation (mirrors SAM2 process_image)
    has_bboxes = bboxes is not None and bboxes != ""
    has_points = points is not None and points != ""
    has_point_labels = point_labels is not None and point_labels != ""
    assert has_bboxes or has_points, "either bboxes or points must be provided."
    if has_points:
        assert has_point_labels, "point_labels is required if points are provided."

    bboxes = json.loads(bboxes) if isinstance(bboxes, str) and has_bboxes else bboxes
    points = json.loads(points) if isinstance(points, str) and has_points else points
    point_labels = (
        json.loads(point_labels)
        if isinstance(point_labels, str) and has_point_labels
        else point_labels
    )
    if has_points:
        assert len(points) == len(point_labels), (
            f"{len(points)} points provided but there are {len(point_labels)} labels."
        )

    # Build transformers prompt inputs (same nesting as SAM2):
    #   input_boxes:  (image, num_boxes, 4)            -> one object per box
    #   input_points: (image, num_objects, num_points, 2)
    #   input_labels: (image, num_objects, num_points)
    proc_kwargs = {}
    if has_bboxes:
        proc_kwargs["input_boxes"] = [
            [[b["x0"], b["y0"], b["x1"], b["y1"]] for b in bboxes]
        ]
    if has_points:
        proc_kwargs["input_points"] = [[[[p["x"], p["y"]] for p in points]]]
        proc_kwargs["input_labels"] = [[list(point_labels)]]

    return _gpu_image_visual_inference(im, proc_kwargs)


@spaces.GPU(duration=20)
def _gpu_image_visual_inference(im: Image.Image, proc_kwargs: dict) -> list[str]:
    inputs = TRK_PROCESSOR(
        images=im.convert("RGB"), return_tensors="pt", **proc_kwargs
    ).to(DEVICE)
    with torch.no_grad():
        outputs = TRK_MODEL(**inputs, multimask_output=False)

    # post_process_masks upscales to original size and binarizes;
    # [0] -> (num_objects, num_masks=1, h, w)
    masks = TRK_PROCESSOR.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]
    output_masks = [np.asarray(mask).squeeze().astype(np.uint8) for mask in masks]
    return [b64_mask_encode(m).decode("ascii") for m in output_masks]


def image_text_inference(
    im: Image.Image, prompt: str, conf_threshold: float = 0.5
) -> list[dict]:
    """Segment every instance of a concept in a single image with SAM3 using a natural-language text prompt (concept segmentation).

    The prompt is a concept to find (e.g. "player in white", "red car"); each matching instance becomes one detection. Returns a JSON list of detection objects, each with: "object_id" (integer index of the instance within this image); "x", "y", "w", "h" (floats); "conf" (float 0-1 detection score); and "mask_b64" (string). The bounding box ("x","y","w","h") is NORMALIZED 0-1: (x, y) is the top-left corner and (w, h) is the box size, each divided by the image width/height -- i.e. the albumentations "coco" layout [x_min, y_min, width, height] but normalized to 0-1 rather than absolute pixels, and NOT the "pascal_voc" [x_min, y_min, x_max, y_max] layout. Bounding-box formats are documented at https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats . "mask_b64" is a base64-encoded 1-bit PNG of the binary segmentation mask at the input image resolution; decode it with PIL, e.g. numpy.array(Image.open(io.BytesIO(base64.b64decode(mask_b64)))) to get a 0/255 mask.

    Args:
        im: The RGB image to segment (a PIL Image).
        prompt: Natural-language concept to segment, e.g. "player in white"; all matching instances are returned.
        conf_threshold: Minimum detection score from 0.0 to 1.0; instances scoring below this are discarded.
    """
    assert IMG_MODEL is not None and IMG_PROCESSOR is not None, (
        "Image text model failed to load on startup."
    )
    assert im is not None and prompt, "Missing image or prompt."

    pil_image = im.convert("RGB")
    return _gpu_image_text_inference(pil_image, prompt, conf_threshold)


@spaces.GPU(duration=20)
def _gpu_image_text_inference(
    pil_image: Image.Image, prompt: str, conf_threshold: float
) -> list[dict]:
    inputs = IMG_PROCESSOR(
        images=pil_image, text=prompt, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = IMG_MODEL(**inputs)

    results = IMG_PROCESSOR.post_process_instance_segmentation(
        outputs,
        threshold=conf_threshold,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    img_w, img_h = pil_image.size
    detections = []
    for i, (mask, score) in enumerate(zip(results["masks"], results["scores"])):
        mask_bin = (mask.cpu().numpy() > 0).astype(np.uint8)
        xyxy = mask_to_xyxy(mask_bin)
        if not xyxy:
            continue
        x0, y0, x1, y1 = xyxy
        detections.append(
            {
                "object_id": i,
                "x": x0 / img_w,
                "y": y0 / img_h,
                "w": (x1 - x0) / img_w,
                "h": (y1 - y0) / img_h,
                "conf": float(score),
                "mask_b64": b64_mask_encode(mask_bin).decode("ascii"),
            }
        )
    return detections


# the Gradio App
with gr.Blocks() as app:
    with gr.Tab("Video-Object Tracking"):
        gr.Interface(
            fn=video_inference,
            inputs=[
                gr.Video(label="Input Video"),
                gr.Textbox(
                    label="Prompt",
                    lines=3,
                    info="Describe the Object(s) you would like to track/ segmentate",
                    value="",
                ),
                gr.Number(
                    label="Timeout Override (seconds)",
                    info="max GPU lease; leave empty to auto-estimate from video length and downsampling",
                    value=None,
                    precision=0,
                ),
                gr.Number(
                    label="Sample FPS",
                    info="downsample the video to this many frames per second before tracking; empty = every frame",
                    value=None,
                ),
                gr.Number(
                    label="Every Xth Frame",
                    info="keep only every Nth frame before tracking; ignored when Sample FPS is set",
                    value=None,
                    precision=0,
                ),
                gr.Dropdown(
                    label="Video Load Device",
                    info="where preprocessed frames are stored; cpu streams them to the GPU per frame",
                    choices=["cpu", "cuda"],
                    value="cpu",
                ),
            ],
            outputs=gr.JSON(label="Output JSON"),
            title="SAM3 Video Segmentation",
            description="Segment Objects in Video using Text Prompts",
            api_name="video_inference",
        )
    with gr.Tab("Video Annotation"):
        gr.Interface(
            fn=video_annotation,
            inputs=[
                gr.Video(label="Input Video"),
                gr.Textbox(
                    label="Prompt",
                    lines=3,
                    info="Describe the Object(s) you would like to track/ segmentate",
                    value="",
                ),
                gr.Number(
                    label="Timeout Override (seconds)",
                    info="max GPU lease; leave empty to auto-estimate from video length and downsampling",
                    value=None,
                    precision=0,
                ),
                gr.Number(
                    label="Sample FPS",
                    info="downsample the video to this many frames per second before tracking; empty = every frame",
                    value=None,
                ),
                gr.Number(
                    label="Every Xth Frame",
                    info="keep only every Nth frame before tracking; ignored when Sample FPS is set",
                    value=None,
                    precision=0,
                ),
                gr.Dropdown(
                    label="Video Load Device",
                    info="where preprocessed frames are stored; cpu streams them to the GPU per frame",
                    choices=["cpu", "cuda"],
                    value="cpu",
                ),
            ],
            outputs=gr.Video(label="Processed Video"),
            title="SAM3 Video Segmentation",
            description="Segment Objects in Video using Text Prompts",
            api_name="video_annotation",
        )
    with gr.Tab("Video Visual Tracking"):
        gr.Interface(
            fn=video_visual_inference,
            inputs=[
                gr.Video(label="Input Video"),
                gr.Textbox(
                    label="Masks for Objects of Interest in the Reference Frame",
                    value=None,
                    lines=5,
                    placeholder="""
                    JSON list of base64 encoded masks, e.g.: ["b'iVBORw0KGgoAAAANSUhEUgAABDgAAAeAAQAAAAADGtqnAAAXz...'",...]
                    """,
                ),
                gr.Checkbox(
                    label="Drop Masks",
                    info="remove base64 encoded masks from result JSON",
                    value=True,
                ),
                gr.Number(
                    label="Reference Frame Index",
                    info="frame index for the provided object masks",
                    value=0,
                    precision=0,
                ),
                gr.Number(
                    label="Timeout Override (seconds)",
                    info="max GPU lease; leave empty to auto-estimate from video length and downsampling",
                    value=None,
                    precision=0,
                ),
                gr.Number(
                    label="Sample FPS",
                    info="downsample the video to this many frames per second before tracking; empty = every frame",
                    value=None,
                ),
                gr.Number(
                    label="Every Xth Frame",
                    info="keep only every Nth frame before tracking; ignored when Sample FPS is set",
                    value=None,
                    precision=0,
                ),
                gr.Dropdown(
                    label="Video Load Device",
                    info="where preprocessed frames are stored; cpu streams them to the GPU per frame",
                    choices=["cpu", "cuda"],
                    value="cpu",
                ),
            ],
            outputs=gr.JSON(label="Output JSON"),
            title="SAM3 Video Tracking (Visual Prompts)",
            description="Track Objects in Video from their Masks on a Reference Frame (SAM2-style)",
            api_name="video_visual_inference",
        )
    with gr.Tab("Image Visual Segmentation"):
        gr.Interface(
            fn=image_visual_inference,
            inputs=[
                gr.Image(label="Input Image", type="pil"),
                gr.Dropdown(
                    label="Model Variant",
                    choices=["sam3"],
                    value="sam3",
                    info="Kept for SAM2 API parity; ignored (SAM3 has a single model)",
                ),
                gr.Textbox(
                    label="Bounding Boxes",
                    value=None,
                    lines=5,
                    placeholder='JSON list of dicts: [{"x0":..., "y0":..., "x1":..., "y1":...}, ...]',
                ),
                gr.Textbox(
                    label="Points",
                    lines=3,
                    placeholder='JSON list of dicts: [{"x":..., "y":...}, ...]',
                ),
                gr.Textbox(
                    label="Points' Labels",
                    placeholder="JSON list of ints, e.g. [1, 0] (1=foreground, 0=background)",
                ),
            ],
            outputs=gr.JSON(label="Output JSON"),
            title="SAM3 Image Segmentation (Visual Prompts)",
            description="Segment Objects in an Image using Points and/or Bounding Boxes",
            api_name="image_visual_inference",
        )
    with gr.Tab("Image Text Segmentation"):
        gr.Interface(
            fn=image_text_inference,
            inputs=[
                gr.Image(label="Input Image", type="pil"),
                gr.Textbox(
                    label="Prompt",
                    lines=2,
                    info="Concept to segment, e.g. 'player in white'",
                    value="",
                ),
                gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05, label="Confidence Threshold"
                ),
            ],
            outputs=gr.JSON(label="Output JSON"),
            title="SAM3 Image Segmentation (Text Prompt)",
            description="Segment all instances of a concept in an Image using a Text Prompt",
            api_name="image_text_inference",
        )
app.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
