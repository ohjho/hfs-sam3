# Import helpers for mask encoding and bbox extraction
import inspect
import json
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
    Sam3VideoModel,
    Sam3VideoProcessor,
)

# Import ffmpeg_extractor helpers
from ffmpeg_extractor import extract_frames, get_video_metadata

# import local helpers
from toolbox.mask_encoding import b64_mask_encode
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


def calc_timeout_duration(vid_file, *args, **kwargs):
    sig = inspect.signature(video_inference)
    bound = sig.bind(vid_file, *args, **kwargs)
    bound.apply_defaults()
    return bound.arguments.get("timeout_duration", 60)


# Our Inference Function
@spaces.GPU(duration=calc_timeout_duration)
def video_inference(
    input_video,
    prompt: str,
    timeout_duration: int = 60,
    annotation_mode: bool = False,
):
    """
    Segments objects in a video using a text prompt.
    Returns a list of detection dicts (one per object per frame) and output video path/status.
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

    # Extract frames as PIL Images (no timestamp/frame_num overlays)
    pil_frames = extract_frames(
        video_path,
        fps=int(vid_fps),
        max_short_edge=min(vid_w, vid_h),
        write_timestamp=False,
        write_frame_num=False,
        output_dir=None,
    )
    assert len(pil_frames) > 0, "No frames found in video."

    # Convert PIL Images to numpy arrays (RGB)
    video_frames = [np.array(frame.convert("RGB")) for frame in pil_frames]

    session = VID_PROCESSOR.init_video_session(
        video=video_frames, inference_device=DEVICE, dtype=DTYPE
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
            vid_fps=vid_fps,
            vid_h=vid_h,
            vid_w=vid_w,
        )
        if annotation_mode
        else detections
    )


def video_annotation(input_video, prompt: str, timeout_duration: int = 60):
    return video_inference(
        input_video, prompt, timeout_duration=timeout_duration, annotation_mode=True
    )


@spaces.GPU
def image_visual_inference(
    im: Image.Image,
    variant=None,
    bboxes=None,
    points=None,
    point_labels=None,
):
    """
    SAM3 image segmentation with visual prompts (points and/or boxes).

    Drop-in match for SAM2's `process_image`:

    Args:
        im: Pillow Image
        variant: accepted for SAM2 API parity; ignored (SAM3 has a single model)
        bboxes: bounding boxes to segment, as a list of dicts (or JSON string):
            [{"x0":..., "y0":..., "x1":..., "y1":...}, ...] -- one mask per box
        points: points to segment, as a list of dicts (or JSON string):
            [{"x":..., "y":...}, ...] -- all points describe a single object/mask
        point_labels: list of ints (or JSON string), 1=foreground, 0=background
    Returns:
        list: a list of base64-encoded mask strings (one per box, then one for points)
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


@spaces.GPU
def image_text_inference(im: Image.Image, prompt: str, conf_threshold: float = 0.5):
    """
    SAM3 image segmentation with a text prompt (concept segmentation).

    Args:
        im: Pillow Image
        prompt: concept to segment, e.g. "player in white"
        conf_threshold: score threshold for kept instances
    Returns:
        list: detection dicts [{"track_id":..., "x":..., "y":..., "w":..., "h":...,
              "conf":..., "mask_b64":...}, ...] with normalized x/y/w/h
    """
    assert IMG_MODEL is not None and IMG_PROCESSOR is not None, (
        "Image text model failed to load on startup."
    )
    assert im is not None and prompt, "Missing image or prompt."

    pil_image = im.convert("RGB")
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
                "track_id": i,
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
                gr.Radio([60, 120, 180, 240], value=60, label="Timeout (seconds)"),
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
                gr.Radio([60, 120, 180, 240], value=60, label="Timeout (seconds)"),
            ],
            outputs=gr.Video(label="Processed Video"),
            title="SAM3 Video Segmentation",
            description="Segment Objects in Video using Text Prompts",
            api_name="video_annotation",
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
