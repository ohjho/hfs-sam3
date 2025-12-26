# Import helpers for mask encoding and bbox extraction
import sys
import tempfile
from ast import Return

import cv2
import gradio as gr
import matplotlib
import numpy as np
import spaces
import torch
from loguru import logger
from PIL import Image
from transformers import (
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


# Our Inference Function
@spaces.GPU(duration=120)
def video_inference(input_video, prompt: str, annotation_mode: bool = False):
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
            final_frame = apply_mask_overlay(
                original_pil, detected_masks, object_ids=object_ids
            )
        else:
            final_frame = original_pil
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


def video_annotation(input_video, prompt: str):
    return video_inference(input_video, prompt, annotation_mode=True)


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
            ],
            outputs=gr.Video(label="Processed Video"),
            title="SAM3 Video Segmentation",
            description="Segment Objects in Video using Text Prompts",
            api_name="video_annotation",
        )
app.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
