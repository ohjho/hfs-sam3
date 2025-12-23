import ffmpeg, typer, os, sys, json
from loguru import logger
from PIL import Image
from tqdm import tqdm

logger.remove()
logger.add(
    sys.stderr,
    format="<d>{time:YYYY-MM-DD ddd HH:mm:ss}</d> | <lvl>{level}</lvl> | <lvl>{message}</lvl>",
)
app = typer.Typer(pretty_exceptions_show_locals=False)


def parse_frame_name(fname: str):
    """return a tuple of frame_type and frame_index"""
    fn, fext = os.path.splitext(os.path.basename(fname))
    frame_type, frame_index = fn.split("_")
    return frame_type, int(frame_index)


def get_fps_ffmpeg(video_path: str):
    probe = ffmpeg.probe(video_path)
    # Find the first video stream
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    if video_stream is None:
        raise ValueError("No video stream found")
    # Frame rate is given as a string fraction, e.g., '30000/1001'
    r_frame_rate = video_stream["r_frame_rate"]
    num, denom = map(int, r_frame_rate.split("/"))
    return num / denom


@app.command()
def get_video_metadata(video_path: str, bverbose: bool = True):
    """
    Extract comprehensive metadata from a video file.

    Args:
        video_path (str): Path to the video file

    Returns:
        dict: Dictionary containing video metadata including:
            - width, height: Video dimensions
            - duration: Video duration in seconds
            - fps: Frames per second
            - codec: Video codec name
            - bitrate: Video bitrate
            - format_name: Container format
            - file_size: File size in bytes
    """
    try:
        probe = ffmpeg.probe(video_path)

        # Find the first video stream
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )

        if video_stream is None:
            raise ValueError("No video stream found")

        # Extract basic video properties
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        duration = float(video_stream.get("duration", 0))

        # Calculate FPS
        r_frame_rate = video_stream.get("r_frame_rate", "0/1")
        num, denom = map(int, r_frame_rate.split("/"))
        fps = num / denom if denom != 0 else 0

        # Get codec and bitrate
        codec = video_stream.get("codec_name", "unknown")
        bitrate = (
            int(video_stream.get("bit_rate", 0)) if video_stream.get("bit_rate") else 0
        )

        # Get format information
        format_info = probe.get("format", {})
        format_name = format_info.get("format_name", "unknown")
        file_size = int(format_info.get("size", 0))

        # Get audio stream info if available
        audio_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "audio"),
            None,
        )

        audio_codec = audio_stream.get("codec_name", "none") if audio_stream else "none"
        audio_bitrate = (
            int(audio_stream.get("bit_rate", 0))
            if audio_stream and audio_stream.get("bit_rate")
            else 0
        )

        metadata = {
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "video_codec": codec,
            "video_bitrate": bitrate,
            "audio_codec": audio_codec,
            "audio_bitrate": audio_bitrate,
            "format_name": format_name,
            "file_size": file_size,
            "total_streams": len(probe["streams"]),
        }
        if bverbose:
            logger.info(f"Video metadata extracted: {json.dumps(metadata, indent=4)}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to extract video metadata: {e}")
        return None


@app.command()
def extract_frames(
    input_path: str,
    fps: int = 8,
    max_short_edge: int = 1080,
    write_timestamp: bool = True,
    write_frame_num: bool = True,
    output_dir: str = None,
):
    """
    Extract frames from a video file using FFmpeg.

    Args:
        input_path (str): Path to the input video file.
        fps (int): Frames per second to extract.
        max_short_edge (int): Maximum length of the shorter edge of the extracted frames.
        write_timestamp (bool): Whether to write the timestamp of each frame.
        write_frame_num (bool): Whether to write the frame number of each frame.
        output_dir (str): Directory to save the extracted frames.

    Returns:
        List of PIL Images
    """
    if output_dir:
        assert os.path.isdir(
            output_dir
        ), f"Output directory {output_dir} does not exist"

    # Probe video to get width, height, and duration
    vmeta = get_video_metadata(input_path, bverbose=False)
    org_w, org_h = vmeta["width"], vmeta["height"]
    max_short_edge = int(max_short_edge) if max_short_edge else min(org_w, org_h)
    long_edge = int((max(org_h, org_w) / min(org_h, org_w)) * max_short_edge)
    long_edge += 0 if long_edge % 2 == 0 else 1
    duration = vmeta["duration"]
    org_fps = vmeta["fps"]
    if fps > org_fps:
        logger.debug(
            f"requested fps({fps}) exceeded source fps({org_fps}): fps will be capped to source fps({org_fps})"
        )
        fps = org_fps

    # Calculate total frames to extract based on fps and duration
    total_frames = int(duration * fps)

    # add scale filter only if needed
    add_scale_filter = max_short_edge < min(org_w, org_h)
    w = max_short_edge if org_w < org_h else long_edge
    h = max_short_edge if org_w > org_h else long_edge
    logger.debug(f"Video dimensions: {org_w}x{org_h}")
    if add_scale_filter:
        logger.debug(f"\tscaling video to {w}x{h}")

    # Set drawtext filter text
    drawtext_filter_text = (
        r"text='Timestamp\:%{pts\:hms} \|Frame Number\: %{frame_num}'"
        if write_frame_num
        else r"text='Timestamp\:%{pts\:hms}'"
    )

    # Setup the ffmpeg filter chain
    drawtext_filter = (
        f",drawtext={drawtext_filter_text}: x=(w-tw)/2: y=h-(2*lh): fontcolor=white: fontsize=20: box=1: boxcolor=0x00000099: boxborderw=5"
        if write_timestamp
        else ""
    )
    scale_filter = (
        # f",scale='if(lt(iw, ih), {max_short_edge}, -2)':'if(lt(ih, iw), {max_short_edge}, -2)'"
        f",scale='{w}:{h}'"
        if add_scale_filter
        else ""
    )
    filter_chain = f"fps={fps}{drawtext_filter}{scale_filter}"

    # Run ffmpeg process with output as rawvideo piped to stdout
    process = (
        ffmpeg.input(input_path)
        .output("pipe:", vf=filter_chain, format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    logger.info(f"running ffmpeg with filter:\n{filter_chain}")

    frame_size = (
        long_edge * max_short_edge * 3 if add_scale_filter else org_w * org_h * 3
    )  # 3 bytes per pixel (RGB)
    frames = []

    # Use a for loop with known total frames count to read frames
    for _ in tqdm(range(total_frames), desc="Extracting frames with FFMPEG"):
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes or len(in_bytes) < frame_size:
            break
        frame = Image.frombytes(
            "RGB", (w, h) if add_scale_filter else (org_w, org_h), in_bytes
        )
        frames.append(frame)

    process.stdout.close()
    process.wait()

    if output_dir:
        vname, _ = os.path.splitext(os.path.basename(input_path))
        for i, im in enumerate(tqdm(frames, desc=f"Saving frames to {output_dir}")):
            output_path = os.path.join(output_dir, f"{vname}_{i}.jpg")
            im.save(output_path)

    return frames


def extract_specific_frames(
    input_path: str,
    timestamps_or_frames: list,
    max_short_edge: int = 1080,
):
    """
    Extract specific frames from a video file using FFmpeg at given timestamps or frame numbers.

    Args:
        input_path (str): Path to the input video file.
        timestamps_or_frames (list): List of timestamps (in seconds) or frame numbers to extract.
        max_short_edge (int): Maximum length of the shorter edge of the extracted frames.
        write_timestamp (bool): Whether to write the timestamp of each frame.
        write_frame_num (bool): Whether to write the frame number of each frame.
        use_timestamps (bool): If True, treat input list as timestamps. If False, treat as frame numbers.

    Returns:
        List of PIL Images corresponding to the specified timestamps/frames
    """
    # Probe video to get width, height, and duration
    vmeta = get_video_metadata(input_path, bverbose=False)
    org_w, org_h = vmeta["width"], vmeta["height"]
    max_short_edge = int(max_short_edge) if max_short_edge else min(org_w, org_h)
    long_edge = int((max(org_h, org_w) / min(org_h, org_w)) * max_short_edge)
    long_edge += 0 if long_edge % 2 == 0 else 1
    duration = vmeta["duration"]
    org_fps = vmeta["fps"]

    # add scale filter only if needed
    add_scale_filter = max_short_edge < min(org_w, org_h)
    w = max_short_edge if org_w < org_h else long_edge
    h = max_short_edge if org_w > org_h else long_edge
    logger.debug(f"Video dimensions: {org_w}x{org_h}")
    if add_scale_filter:
        logger.debug(f"\tscaling video to {w}x{h}")
    scale_filter = f",scale='{w}:{h}'" if add_scale_filter else ""

    frames = []

    for target in tqdm(timestamps_or_frames, desc="Extracting specific frames"):
        try:
            # Convert frame number to timestamp if needed
            use_timestamps = isinstance(target, float)
            if use_timestamps:
                seek_time = float(target)
                if seek_time > duration:
                    logger.warning(
                        f"Timestamp {seek_time}s exceeds video duration {duration}s, skipping"
                    )
                    continue
            else:
                # Convert frame number to timestamp
                seek_time = float(target) / org_fps
                if seek_time > duration:
                    logger.warning(f"Frame {target} exceeds video duration, skipping")
                    continue

            filter_chain = f"fps={org_fps}{scale_filter}"

            # Extract single frame at specific timestamp
            logger.debug(f"Extracting frame at {seek_time}s")
            process = (
                ffmpeg.input(input_path, ss=seek_time)
                .output(
                    "pipe:",
                    vf=filter_chain,
                    format="rawvideo",
                    pix_fmt="rgb24",
                    frames=1,
                )
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            frame_size = (
                w * h * 3 if add_scale_filter else org_w * org_h * 3
            )  # 3 bytes per pixel (RGB)

            in_bytes = process.stdout.read(frame_size)
            if in_bytes and len(in_bytes) >= frame_size:
                frame = Image.frombytes(
                    "RGB", (w, h) if add_scale_filter else (org_w, org_h), in_bytes
                )
                frames.append(frame)
            else:
                logger.warning(
                    f"Failed to extract frame at {'timestamp' if use_timestamps else 'frame'} {target}"
                )
                frames.append(
                    None
                )  # Add None for failed extractions to maintain list alignment

            process.stdout.close()
            process.wait()

        except Exception as e:
            logger.error(
                f"Error extracting frame at {'timestamp' if use_timestamps else 'frame'} {target}: {e}"
            )
            frames.append(None)  # Add None for failed extractions

    # Filter out None values if desired (or keep them for alignment)
    logger.info(
        f"Successfully extracted {len([f for f in frames if f is not None])} out of {len(timestamps_or_frames)} requested frames"
    )

    return frames


@app.command()
def extract_audio(
    video_path: str,
    output_dir: str = "/tmp/miro/clip_cognition/audio/",
    overwrite: bool = False,
):
    """extracting audio of a video file into m4a without re-encoding
    ref: https://www.baeldung.com/linux/ffmpeg-audio-from-video#1-extracting-audio-without-re-encoding
    """
    # only return audio if its available
    vmeta = get_video_metadata(video_path, bverbose=False)
    if vmeta.get("audio_codec") == "none":
        logger.error(f"No audio found in {video_path}")
        return None

    # Create output directory if it doesn't exist
    output_dir = output_dir if output_dir else os.path.dirname(video_path)
    vname, vext = os.path.splitext(os.path.basename(video_path))
    output_dir = os.path.join(output_dir, vname)
    output_fname = os.path.join(output_dir, vname + ".mp3")
    if os.path.isfile(output_fname):
        if overwrite:
            os.remove(output_fname)
            logger.warning(f"removed existing data: {output_fname}")
        else:
            logger.error(f"overwrite is false and data already exists!")
            return None
    os.makedirs(output_dir, exist_ok=True)

    # Construct the ffmpeg-python pipeline
    stream = ffmpeg.input(video_path)
    config_dict = {"map": "0:a", "acodec": "mp3"}  # "copy"}
    stream = ffmpeg.output(stream, output_fname, **config_dict)

    # Execute the ffmpeg command
    try:
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        logger.success(f"audio extracted to {output_fname}")
        return output_fname
    except ffmpeg.Error as e:
        logger.error(f"Error executing FFmpeg command: {e.stderr.decode()}")
    return None


if __name__ == "__main__":
    app()
