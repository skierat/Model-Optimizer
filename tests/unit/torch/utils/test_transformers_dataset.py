# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Transformer-backed dataset collators."""

from types import SimpleNamespace

from modelopt.torch.utils.plugins import transformers_dataset
from modelopt.torch.utils.plugins.transformers_dataset import (
    ShardedDataset,
    VisionLanguageDataCollator,
)


def test_vlm_video_processor_uses_image_pixel_bounds(monkeypatch):
    """Qwen3-VL video limits inherit launcher image limits unless explicitly overridden."""
    monkeypatch.delenv("VLM_VIDEO_MIN_PIXELS", raising=False)
    monkeypatch.delenv("VLM_VIDEO_MAX_PIXELS", raising=False)
    video_processor = SimpleNamespace(
        size={"shortest_edge": 4096, "longest_edge": 25165824}
    )
    processor = SimpleNamespace(video_processor=video_processor)

    VisionLanguageDataCollator._configure_video_processor(
        processor,
        {"min_pixels": 50176, "max_pixels": 802816},
    )

    assert video_processor.size == {"shortest_edge": 50176, "longest_edge": 802816}

    monkeypatch.setenv("VLM_VIDEO_MAX_PIXELS", "2097152")
    VisionLanguageDataCollator._configure_video_processor(
        processor,
        {"min_pixels": 50176, "max_pixels": 802816},
    )

    assert video_processor.size == {"shortest_edge": 50176, "longest_edge": 2097152}


def test_jsonl_fallback_preserves_heterogeneous_records(monkeypatch, tmp_path):
    """The mixed PAI/video and VQA/image rows bypass Arrow only after it rejects them."""
    data_path = tmp_path / "pai-vqa.jsonl"
    data_path.write_text(
        "{\"messages\":[{\"content\":[{\"type\":\"video\",\"video\":\"a.mp4\"}]}]}\n"
        "{\"messages\":[{\"content\":[{\"type\":\"image\",\"image\":\"b.jpg\"}]}]}\n",
        encoding="utf-8",
    )

    def fail_arrow_load(*args, **kwargs):
        raise RuntimeError("incompatible Arrow schema")

    monkeypatch.setattr(transformers_dataset, "load_dataset", fail_arrow_load)
    dataset = ShardedDataset("json", data_files=str(data_path))

    assert len(dataset) == 2
    assert dataset[0]["messages"][0]["content"][0]["video"] == "a.mp4"
    assert dataset[1]["messages"][0]["content"][0]["image"] == "b.jpg"
