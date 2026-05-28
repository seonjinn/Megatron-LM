#!/usr/bin/env python3
"""
Streamlit app to visualize Q/A samples from YAML config datasets.
Simple viewer that shows images and Q/A pairs without curation status.
"""

import streamlit as st
import json
import os
import re
import random
import socket
import subprocess
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple

# Default YAML config path
DEFAULT_YAML_CONFIG = "/lustre/fsw/portfolios/llmservice/users/matthieul/eagle_recipe_online_packing/final_recipe/eagle_sft_v13.52.no.text.2x.commercial.yaml"
DEFAULT_JSONL_BASE = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data"

#DEFAULT_YAML_CONFIG = "/lustre/fsw/portfolios/llmservice/users/matthieul/eagle_recipe_online_packing/final_recipe/pretraining_ocr_guilin_102325.yaml"
#DEFAULT_JSONL_BASE = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets"
# Bounding box colors for multiple boxes
BBOX_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
]


def parse_bboxes_from_text(text: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse bounding boxes from text.

    Supports formats:
    - <box>[x1, y1, x2, y2]</box>
    - [x1, y1, x2, y2] (standalone in certain contexts)

    Coordinates are assumed to be normalized 0-1000.
    Returns list of (x1, y1, x2, y2) tuples.
    """
    bboxes = []

    # Pattern for <box>[x1, y1, x2, y2]</box>
    box_pattern = r'<box>\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*</box>'
    for match in re.finditer(box_pattern, text):
        x1, y1, x2, y2 = map(int, match.groups())
        bboxes.append((x1, y1, x2, y2))

    # Pattern for standalone [x1, y1, x2, y2] in question context
    bracket_pattern = r'(?:inside|within|from|at)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    for match in re.finditer(bracket_pattern, text, re.IGNORECASE):
        x1, y1, x2, y2 = map(int, match.groups())
        if (x1, y1, x2, y2) not in bboxes:
            bboxes.append((x1, y1, x2, y2))

    return bboxes


def draw_bboxes_on_image(img: Image.Image, bboxes: List[Tuple[int, int, int, int]],
                          norm_scale: int = 1000, line_width: int = 3) -> Image.Image:
    """Draw bounding boxes on an image."""
    if not bboxes:
        return img

    img_with_boxes = img.copy().convert('RGB')
    draw = ImageDraw.Draw(img_with_boxes)

    img_width, img_height = img_with_boxes.size

    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        px1 = int(x1 * img_width / norm_scale)
        py1 = int(y1 * img_height / norm_scale)
        px2 = int(x2 * img_width / norm_scale)
        py2 = int(y2 * img_height / norm_scale)

        color = BBOX_COLORS[idx % len(BBOX_COLORS)]

        for i in range(line_width):
            draw.rectangle(
                [px1 - i, py1 - i, px2 + i, py2 + i],
                outline=color
            )

        label = f"Q{idx + 1}" if len(bboxes) > 1 else "BOX"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        label_bbox = draw.textbbox((px1, py1 - 20), label, font=font)
        draw.rectangle(label_bbox, fill=color)
        draw.text((px1, py1 - 20), label, fill=(255, 255, 255), font=font)

    return img_with_boxes


# Page config
st.set_page_config(
    page_title="Q/A Dataset Viewer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap');

    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .qa-box {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #00b4db;
    }

    .question-label {
        color: #00b4db;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .answer-label {
        color: #38ef7d;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .stats-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #0f3460;
    }

    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .stats-label {
        color: #a0a0a0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)


def build_jsonl_path(rel_path: str, jsonl_base: str) -> str:
    """
    Build the full JSONL path from a relative path.

    Handles paths like 'playground/commercial_sft_jsonl/...' by stripping
    the 'playground/' prefix and joining with the base directory.
    """
    if rel_path.startswith('playground/'):
        jsonl_path = os.path.join(jsonl_base, rel_path[len('playground/'):]) + '.jsonl'
    else:
        jsonl_path = os.path.join(jsonl_base, rel_path) + '.jsonl'
    return jsonl_path


def load_yaml_config(yaml_path: str, jsonl_base: str = DEFAULT_JSONL_BASE) -> dict:
    """Load the YAML config and return dataset entries with their paths and media sources."""
    datasets = []
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        splits = config.get('splits', {})
        train = splits.get('train', {})
        blend_entries = train.get('blend_epochized', [])

        for entry in blend_entries:
            rel_path = entry.get('path', '')
            aux = entry.get('aux', {})
            media_source = aux.get('media_source', '')
            subflavors = entry.get('subflavors', {})
            name = subflavors.get('name', '')

            # Build full JSONL path from relative path
            jsonl_path = build_jsonl_path(rel_path, jsonl_base) if rel_path else ''

            # Strip 'filesystem://' prefix from media_source
            if media_source.startswith('filesystem:///'):
                media_source = media_source[len('filesystem://'):]
            elif media_source.startswith('filesystem://'):
                media_source = media_source[len('filesystem://'):]

            if jsonl_path and media_source:
                datasets.append({
                    'path': jsonl_path,
                    'media_source': media_source,
                    'name': name or os.path.basename(rel_path)
                })

    except Exception as e:
        st.error(f"Could not load YAML config: {e}")

    return datasets


def load_jsonl_data(file_path: str) -> list:
    """Load and parse a JSONL file."""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        st.error(f"Could not load file: {e}")
    return data


def extract_qa_pairs(entry: dict) -> list:
    """Extract all question/answer pairs from an entry."""
    qa_pairs = []
    conversations = entry.get('conversations', [])

    i = 0
    while i < len(conversations):
        conv = conversations[i]
        if conv.get('from') == 'human':
            question = conv.get('value', '').replace('<image>', '').strip()
            if i + 1 < len(conversations) and conversations[i + 1].get('from') == 'gpt':
                answer = conversations[i + 1].get('value', '')
                qa_pairs.append((question, answer))
                i += 2
            else:
                i += 1
        else:
            i += 1

    return qa_pairs if qa_pairs else []


def get_host_ip():
    """Get the host IP address."""
    try:
        result = subprocess.run(['hostname', '-i'], capture_output=True, text=True)
        ip = result.stdout.strip().split()[0]
        return ip
    except:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"


def render_qa_pair(qa_index: int, question: str, answer: str):
    """Render a single Q/A pair."""
    st.markdown(f"#### Q/A #{qa_index + 1}")

    st.markdown('<div class="question-label">Question</div>', unsafe_allow_html=True)
    st.markdown(f"> {question}")
    st.markdown("")

    st.markdown('<div class="answer-label">Answer</div>', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown("---")


def main():
    # Show server URL info at startup
    if 'server_url_shown' not in st.session_state:
        host_ip = get_host_ip()
        port = os.environ.get('STREAMLIT_SERVER_PORT', '8501')
        st.session_state.server_url = f"http://{host_ip}:{port}"
        st.session_state.server_url_shown = True

    # Display server URL in sidebar
    with st.sidebar:
        st.markdown("### 🌐 Server Info")
        st.code(st.session_state.server_url, language=None)
        st.markdown("---")

    # Header
    st.markdown('<h1 class="main-header">📚 Q/A Dataset Viewer</h1>', unsafe_allow_html=True)
    st.markdown("Browse images and Q/A pairs from YAML config datasets")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Settings")

        # YAML config file
        yaml_config_path = st.text_input(
            "YAML Config Path",
            value=DEFAULT_YAML_CONFIG,
            help="Path to the YAML config file containing dataset paths"
        )

        # Load YAML config
        if 'datasets' not in st.session_state or st.session_state.get('yaml_path') != yaml_config_path:
            st.session_state.datasets = load_yaml_config(yaml_config_path)
            st.session_state.yaml_path = yaml_config_path

        datasets = st.session_state.datasets

        if datasets:
            st.success(f"✓ Found {len(datasets)} datasets")

            # Dataset selector
            dataset_names = [d['name'] for d in datasets]
            selected_dataset_idx = st.selectbox(
                "Select Dataset",
                range(len(dataset_names)),
                format_func=lambda i: dataset_names[i]
            )

            selected_dataset = datasets[selected_dataset_idx]
            jsonl_path = selected_dataset['path']
            image_root = selected_dataset['media_source']

            st.markdown("---")
            st.markdown("**JSONL Path:**")
            st.code(jsonl_path, language=None)
            st.markdown("**Image Root:**")
            st.code(image_root, language=None)
        else:
            st.warning("No datasets found in YAML config")
            jsonl_path = st.text_input("JSONL file path", value="")
            image_root = st.text_input("Image root directory", value="")

        st.markdown("---")

        # Load data button
        if st.button("🔄 Load Data", use_container_width=True):
            st.session_state.reload = True

    # Check if file exists
    if not jsonl_path or not os.path.exists(jsonl_path):
        st.warning(f"JSONL file not found: {jsonl_path}")
        st.info("Please select a valid dataset from the YAML config.")
        return

    # Load data
    current_file = st.session_state.get('current_file')
    file_changed = current_file != jsonl_path

    if 'data' not in st.session_state or st.session_state.get('reload', False) or file_changed:
        with st.spinner("Loading data..."):
            st.session_state.data = load_jsonl_data(jsonl_path)
            st.session_state.current_file = jsonl_path
            st.session_state.reload = False

    data = st.session_state.data

    if not data:
        st.warning("No data found in the file.")
        return

    # Calculate stats
    total_entries = len(data)
    total_qa_pairs = sum(len(extract_qa_pairs(d)) for d in data)

    # Stats cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number" style="color: #00b4db;">{total_entries}</div>
            <div class="stats-label">Total Entries</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number" style="color: #38ef7d;">{total_qa_pairs}</div>
            <div class="stats-label">Total Q/A Pairs</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_qa = total_qa_pairs / total_entries if total_entries else 0
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number" style="color: #a78bfa;">{avg_qa:.1f}</div>
            <div class="stats-label">Avg Q/A per Entry</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Random sampling buttons
    st.markdown("### 🎲 Random Sampling")
    rand_col1, rand_col2 = st.columns(2)

    with rand_col1:
        if st.button("🎲 Random Sample", use_container_width=True, disabled=len(data) == 0):
            if data:
                random_idx = random.randint(0, len(data) - 1)
                st.session_state.selected_entry = data[random_idx]
                st.session_state.selected_idx = random_idx
                st.rerun()

    with rand_col2:
        if st.button("🔄 Clear Selection", use_container_width=True):
            if 'selected_entry' in st.session_state:
                del st.session_state.selected_entry
            st.rerun()

    st.markdown("---")

    # Check if we have a randomly selected entry
    if 'selected_entry' in st.session_state:
        entry = st.session_state.selected_entry
        qa_pairs = extract_qa_pairs(entry)
        image_rel_path = entry.get('image', '')
        image_path = os.path.join(image_root, image_rel_path)
        num_qa = len(qa_pairs)

        st.info(f"📌 Showing randomly selected sample #{st.session_state.get('selected_idx', '?')} ({num_qa} Q/A pairs). Click 'Clear Selection' to browse.")

        # Q/A selector for multi-question entries
        selected_qa_idx = 0
        if num_qa > 1:
            qa_options = []
            for idx, (q, a) in enumerate(qa_pairs):
                q_preview = q[:50] + "..." if len(q) > 50 else q
                qa_options.append(f"Q/A #{idx + 1}: {q_preview}")

            selected_qa_idx = st.selectbox(
                "Select Q/A pair to view:",
                range(len(qa_options)),
                format_func=lambda i: qa_options[i],
                key="random_qa_selector"
            )

        # Get selected Q/A
        if qa_pairs:
            question, answer = qa_pairs[selected_qa_idx]
        else:
            question, answer = "", ""

        # Main content
        col_img, col_qa = st.columns([1, 1])

        with col_img:
            st.markdown("### 🖼️ Image")

            if not image_rel_path:
                st.warning("No image path in this entry")
            elif os.path.isdir(image_path):
                st.warning(f"Path is a directory (video dataset?): {image_path}")
            elif os.path.exists(image_path):
                try:
                    img = Image.open(image_path)

                    # Extract and draw bounding boxes
                    bboxes = parse_bboxes_from_text(question)
                    if bboxes:
                        img = draw_bboxes_on_image(img, bboxes)
                        st.caption(f"📍 {len(bboxes)} bounding box(es) detected")

                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                st.warning(f"Image not found: {image_path}")

            with st.expander("📍 Image path"):
                st.code(image_rel_path, language=None)

        with col_qa:
            st.markdown(f"### 💬 Q/A #{selected_qa_idx + 1} of {num_qa}")
            if question or answer:
                render_qa_pair(selected_qa_idx, question, answer)
            else:
                st.info("No Q/A pairs found in this entry.")

        with st.expander("📄 Raw JSON data"):
            st.json(entry)

        return

    # Browse mode
    st.markdown("### 📋 Browse Mode")

    if data:
        sample_idx = st.slider(
            "Sample index",
            0, len(data) - 1, 0,
            help="Navigate through samples"
        )
    else:
        st.warning("No samples to browse.")
        return

    # Display sample
    entry = data[sample_idx]
    qa_pairs = extract_qa_pairs(entry)
    image_rel_path = entry.get('image', '')
    image_path = os.path.join(image_root, image_rel_path)
    num_qa = len(qa_pairs)

    # Q/A selector for multi-question entries
    selected_qa_idx = 0
    if num_qa > 1:
        qa_options = []
        for idx, (q, a) in enumerate(qa_pairs):
            q_preview = q[:50] + "..." if len(q) > 50 else q
            qa_options.append(f"Q/A #{idx + 1}: {q_preview}")

        selected_qa_idx = st.selectbox(
            "Select Q/A pair to view:",
            range(len(qa_options)),
            format_func=lambda i: qa_options[i],
            key=f"browse_qa_selector_{sample_idx}"
        )

    # Get selected Q/A
    if qa_pairs:
        question, answer = qa_pairs[selected_qa_idx]
    else:
        question, answer = "", ""

    # Main content
    col_img, col_qa = st.columns([1, 1])

    with col_img:
        st.markdown("### 🖼️ Image")

        if not image_rel_path:
            st.warning("No image path in this entry")
        elif os.path.isdir(image_path):
            st.warning(f"Path is a directory (video dataset?): {image_path}")
        elif os.path.exists(image_path):
            try:
                img = Image.open(image_path)

                # Extract and draw bounding boxes
                bboxes = parse_bboxes_from_text(question)
                if bboxes:
                    img = draw_bboxes_on_image(img, bboxes)
                    st.caption(f"📍 {len(bboxes)} bounding box(es) detected")

                st.image(img, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.warning(f"Image not found: {image_path}")

        with st.expander("📍 Image path"):
            st.code(image_rel_path, language=None)

    with col_qa:
        st.markdown(f"### 💬 Q/A #{selected_qa_idx + 1} of {num_qa}")
        if question or answer:
            render_qa_pair(selected_qa_idx, question, answer)
        else:
            st.info("No Q/A pairs found in this entry.")

    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if sample_idx > 0:
            if st.button("⬅️ Previous", use_container_width=True):
                st.rerun()

    with col3:
        if sample_idx < len(data) - 1:
            if st.button("Next ➡️", use_container_width=True):
                st.rerun()

    # Raw JSON
    with st.expander("📄 Raw JSON data"):
        st.json(entry)


if __name__ == "__main__":
    host_ip = get_host_ip()
    port = os.environ.get('STREAMLIT_SERVER_PORT', '8501')
    print(f"\n{'='*60}")
    print(f"  📚 Q/A Dataset Viewer")
    print(f"  📍 Access URL: http://{host_ip}:{port}")
    print(f"{'='*60}\n")
    main()
