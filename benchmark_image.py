"""Benchmark: Docling OCR vs Qwen3-VL for image processing."""

import asyncio
import base64
import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

# Setup before imports that trigger logging
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["RAPIDOCR_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

import httpx

TEST_URLS = [
    "https://preview.redd.it/newelle-1-2-released-v0-4e7vn39lt2eg1.png?width=1080&crop=smart&auto=webp&s=2441a9eba115c7ea02c7fc4941aecfc20ed1d9df",
    "https://live.staticflickr.com/65535/55047985189_9ccb7e9df9_o.jpg",
]

VISION_PROMPT = "Describe this image in detail. Extract any text you see."
OUTPUT_PREVIEW_LENGTH = 400


async def download_image(url: str) -> tuple[bytes, float]:
    """Download image and return bytes with elapsed time."""
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    return resp.content, time.perf_counter() - start


async def benchmark_docling(image_bytes: bytes, suffix: str) -> tuple[float, str]:
    """Benchmark Docling OCR approach."""
    from core import document

    tmp_path = Path(tempfile.gettempdir()) / f"bench_test{suffix}"
    tmp_path.write_bytes(image_bytes)

    converter = document._get_converter()
    if converter is None:
        return 0.0, "Docling not installed"

    start = time.perf_counter()
    result = converter.convert(str(tmp_path))
    content = result.document.export_to_markdown()
    return time.perf_counter() - start, content


async def benchmark_qwen_vl(image_bytes: bytes) -> tuple[float, str]:
    """Benchmark Qwen3-VL vision approach."""
    from core import config, summarizer

    if not config.SUMMARIZE_ENABLED:
        return 0.0, "Summarization disabled in config"

    img_b64 = base64.b64encode(image_bytes).decode()

    start = time.perf_counter()
    result = await summarizer.summarize_with_vision("", img_b64, VISION_PROMPT)
    return time.perf_counter() - start, result


def truncate(text: str, max_len: int = OUTPUT_PREVIEW_LENGTH) -> str:
    """Truncate text with ellipsis if needed."""
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def print_separator(char: str = "=", width: int = 70) -> None:
    """Print a separator line."""
    print(char * width)


async def run_benchmark(url: str, index: int) -> dict:
    """Run benchmark for a single URL."""
    print(f"\n{'─' * 70}")
    print(f"Image {index}: {url[:70]}...")

    image_bytes, dl_time = await download_image(url)
    suffix = ".png" if ".png" in url else ".jpg"
    print(f"Downloaded: {len(image_bytes)/1024:.1f} KB in {dl_time:.2f}s")

    print("\n[Docling OCR - CPU]")
    docling_time, docling_result = await benchmark_docling(image_bytes, suffix)
    print(f"Time: {docling_time:.2f}s")
    print(f"Output ({len(docling_result)} chars):")
    print(truncate(docling_result))

    print("\n[Qwen3-VL - GPU]")
    qwen_time, qwen_result = await benchmark_qwen_vl(image_bytes)
    print(f"Time: {qwen_time:.2f}s")
    print(f"Output ({len(qwen_result)} chars):")
    print(truncate(qwen_result))

    winner = "Docling" if docling_time < qwen_time else "Qwen3-VL"
    return {"docling_time": docling_time, "qwen_time": qwen_time, "winner": winner}


def print_summary(results: list[dict]) -> None:
    """Print benchmark summary table."""
    print_separator()
    print("SUMMARY")
    print_separator()
    print(f"{'Image':<10} {'Docling':<12} {'Qwen3-VL':<12} {'Winner':<12}")
    print("-" * 46)

    for i, r in enumerate(results):
        print(f"Image {i+1:<4} {r['docling_time']:.2f}s{'':<6} {r['qwen_time']:.2f}s{'':<6} {r['winner']}")

    avg_docling = sum(r['docling_time'] for r in results) / len(results)
    avg_qwen = sum(r['qwen_time'] for r in results) / len(results)
    avg_winner = "Docling" if avg_docling < avg_qwen else "Qwen3-VL"

    print("-" * 46)
    print(f"{'Average':<10} {avg_docling:.2f}s{'':<6} {avg_qwen:.2f}s{'':<6} {avg_winner}")


async def main() -> None:
    """Run the benchmark."""
    print_separator()
    print("IMAGE PROCESSING BENCHMARK: Docling OCR vs Qwen3-VL")
    print_separator()

    results = []
    for i, url in enumerate(TEST_URLS, start=1):
        result = await run_benchmark(url, i)
        results.append(result)

    print()
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
