import os
import base64
import mimetypes
from PIL import Image
import io
from transformers.video_utils import VideoMetadata


def encode_pil_to_jpeg_data_url(pil_image):
    from io import BytesIO
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def sample_video_frames_to_data_urls(video_path_local, fps=1, nframe=0, nframe_max=-1):
    """
    Sample frames from a video and return base64-encoded data URLs along with metadata.

    Args:
        video_path_local: Path to the video file
        fps: Target frames per second for sampling (if > 0, uses fps-based sampling)
        nframe: Number of frames to sample (used if fps <= 0)
        nframe_max: Maximum number of frames to sample

    Returns:
        tuple: (frame_data_urls, metadata)
        - frame_data_urls: List of base64-encoded frame images
        - metadata: VideoMetadata dataclass containing info about the sampled frames:
            - total_num_frames: Number of sampled frames
            - fps: Effective frame rate of the sampled frames
            - duration: Duration covered by the sampled frames (in seconds)
            - video_backend: Backend used for video processing ('decord')
    """
    import numpy as np
    from PIL import Image
    import decord

    vid = decord.VideoReader(video_path_local)
    total_frames = len(vid)
    video_fps = vid.get_avg_fps()
    total_duration = total_frames / max(1e-6, video_fps)

    if fps > 0:
        required_frames = int(total_duration * fps)
        desired_frames = max(1, required_frames)
        if nframe_max > 0 and desired_frames > nframe_max:
            desired_frames = nframe_max
        if desired_frames >= total_frames:
            indices = list(range(total_frames))
        elif desired_frames == 1:
            indices = [0]  # Always use first frame for single frame sampling
        else:
            # Generate evenly spaced indices and ensure uniqueness
            raw_indices = np.linspace(0, total_frames - 1, desired_frames)
            indices = list(np.unique(np.round(raw_indices).astype(int)))
    else:
        desired_frames = max(1, int(nframe) if nframe and nframe > 0 else 8)
        if nframe_max > 0 and desired_frames > nframe_max:
            desired_frames = nframe_max
        if desired_frames >= total_frames:
            indices = list(range(total_frames))
        elif desired_frames == 1:
            indices = [0]  # Always use first frame for single frame sampling
        else:
            # Generate evenly spaced indices and ensure uniqueness
            raw_indices = np.linspace(0, total_frames - 1, desired_frames)
            indices = list(np.unique(np.round(raw_indices).astype(int)))

    images = [Image.fromarray(vid[i].asnumpy()) for i in indices]
    frame_urls = [encode_pil_to_jpeg_data_url(im) for im in images]

    sampled_num_frames = len(indices)

    # Pass source fps and source frame indices so the processor can compute
    # timestamps with vLLM's formula: int(source_frame_idx) * int(1000/source_fps) / 1000
    metadata = VideoMetadata(
        total_num_frames=sampled_num_frames,
        fps=video_fps,
        frames_indices=[int(i) for i in indices],
        duration=total_duration,
        video_backend=None,
    )

    return frame_urls, metadata


def maybe_path_or_url_to_data_urls(path_or_url, fps=1, nframe=0, nframe_max=-1):
    """
    Convert a path or URL to data URLs, handling videos, images, and remote files.

    Args:
        path_or_url: Path or URL to the media file
        fps: Target frames per second for video sampling (if > 0, uses fps-based sampling)
        nframe: Number of frames to sample from video (used if fps <= 0)
        nframe_max: Maximum number of frames to sample

    Returns:
        tuple: (data_urls, metadata)
        - data_urls: List of base64-encoded data URLs
        - metadata: VideoMetadata dataclass with video metadata or None for images
    """
    val = str(path_or_url or "")
    low = val.lower()

    # Handle data URLs
    if low.startswith("data:"):
        if low.startswith("data:video/mp4"):
            header, _, b64part = val.partition(",")
            if not b64part:
                return [val], None
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            try:
                tmp.write(base64.b64decode(b64part))
                tmp.flush(); tmp.close()
                return sample_video_frames_to_data_urls(tmp.name, fps=fps, nframe=nframe, nframe_max=nframe_max)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
        return [val], None

    # Remote URL
    if low.startswith("http://") or low.startswith("https://"):
        if low.endswith(".mp4"):
            try:
                import tempfile, urllib.request
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpf:
                    urllib.request.urlretrieve(val, tmpf.name)
                    local_path = tmpf.name
                result = sample_video_frames_to_data_urls(local_path, fps=fps, nframe=nframe, nframe_max=nframe_max)
                try:
                    os.unlink(local_path)
                except Exception:
                    pass
                return result
            except Exception:
                return [val], None
        return [val], None

    # Local path
    if os.path.exists(val):
        mime, _ = mimetypes.guess_type(val)
        if mime and mime.startswith("image/"):
            with open(val, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return [f"data:{mime};base64,{b64}"], None
        if mime == "video/mp4" or (mime is None and val.endswith(".mp4")):
            return sample_video_frames_to_data_urls(val, fps=fps, nframe=nframe, nframe_max=nframe_max)
        # Fallback: treat as binary image
        with open(val, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return [f"data:image/jpeg;base64,{b64}"], None

    return [val], None


def pil_image_from_base64(b64_str: str) -> Image.Image:
    # Handle data URLs like "data:image/png;base64,...."
    if b64_str.startswith('data:'):
        b64_str = b64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))
