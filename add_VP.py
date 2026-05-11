"""
ViKey Stage 1: render numeric visual prompts as text on cached frames.

For every image in every sub-directory under ``--root_dir``, this script writes
a copy with a frame-index caption (``"frame #NN"``) burned into one corner.
The caption is rendered as **red text on a semi-transparent white background
box** -- the "background" variant of ViKey's visual prompt. Use
``add_VP_outline.py`` instead for the stroked-text variant.

The output directory tree mirrors ``--root_dir`` exactly; downstream stages
(``run_<bench>.py``) point ``--cache-dir`` at this output.

Example
-------
    python add_VP.py \\
        --root_dir /path/to/cached_frames \\
        --output_dir /path/to/cached_frames_VP \\
        --auto-font-div 10
"""

import argparse
import os
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


IMAGE_EXTENSIONS: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')


# -- Helpers -----------------------------------------------------------------
def parse_color(color_str: str) -> Tuple[int, ...]:
    """Parse a comma-separated ``r,g,b`` or ``r,g,b,a`` string into an int tuple.

    Raises ``ValueError`` if the string is malformed or has the wrong arity.
    """
    try:
        parts = tuple(int(x.strip()) for x in color_str.split(','))
        if len(parts) not in (3, 4):
            raise ValueError(
                f"Color must be 3 (R,G,B) or 4 (R,G,B,A) values: {color_str}"
            )
        return parts
    except ValueError as e:
        raise ValueError(
            f"Color parsing error: {e}. Format must be 'r,g,b' or 'r,g,b,a'"
        )


def _load_font(font_path: Optional[str], font_size: int) -> ImageFont.ImageFont:
    """Return a PIL font, falling back to the built-in default on failure.

    ``font_path`` may point to a ``.ttf`` / ``.otf`` file. If it is ``None`` or
    cannot be opened, ``ImageFont.load_default(size=...)`` is used (with a
    secondary fallback for older Pillow versions that do not accept ``size``).
    """
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=font_size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


# -- Single-image rendering --------------------------------------------------
def add_text_to_image(
    image_path: str,
    text: str,
    output_path: Optional[str] = None,
    position: str = 'top-left',
    font_size: Optional[int] = None,
    auto_font_div: int = 15,
    text_color: Tuple[int, int, int] = (255, 0, 0),
    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 200),
    font_path: Optional[str] = None,
    verbose: bool = True,
) -> Optional[str]:
    """Render ``text`` on ``image_path`` at ``position`` and save the result.

    The text is drawn over a semi-transparent rectangle so it stays legible
    regardless of the underlying frame content. Font size is auto-derived from
    the image's shorter side (``min(width, height) // auto_font_div``) when
    ``font_size`` is ``None``.

    Returns the output path on success, or ``None`` if rendering failed.
    """
    try:
        img = Image.open(image_path)

        if font_size is None:
            font_size = min(img.width, img.height) // auto_font_div

        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        txt_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)
        font = _load_font(font_path, font_size)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        padding = 10

        # Anchor the text in the requested corner with `padding` px of margin.
        if position == 'top-right':
            x = img.width - text_width - padding * 2
            y = padding
        elif position == 'bottom-left':
            x = padding
            y = img.height - text_height - padding * 2
        elif position == 'bottom-right':
            x = img.width - text_width - padding * 2
            y = img.height - text_height - padding * 2
        else:  # 'top-left' and any unknown value
            x = padding
            y = padding

        # Semi-transparent backing rectangle for legibility.
        bg_bbox = [
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding,
        ]
        draw.rectangle(bg_bbox, fill=bg_color)
        draw.text((x, y), text, font=font, fill=text_color)

        result = Image.alpha_composite(img, txt_layer).convert('RGB')

        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_labeled{ext}"

        result.save(output_path)
        if verbose:
            print(f"Saved: {output_path}")
        return output_path

    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        return None


def add_filename_to_image(
    image_path: str,
    output_path: Optional[str] = None,
    include_parent: bool = False,
) -> Optional[str]:
    """Convenience wrapper that uses the file's basename as the caption."""
    if include_parent:
        parts = image_path.split('/')
        text = '/'.join(parts[-2:]) if len(parts) >= 2 else os.path.basename(image_path)
    else:
        text = os.path.basename(image_path)
    return add_text_to_image(image_path, text, output_path)


# -- Batch rendering ---------------------------------------------------------
def batch_add_text_to_images(
    image_dir: str,
    text_dict: Optional[dict] = None,
    output_dir: Optional[str] = None,
    use_filename: bool = True,
    **kwargs,
) -> List[str]:
    """Recursively process every image under ``image_dir``.

    The caption for each image is taken from ``text_dict[basename]`` if
    provided; otherwise the image's basename is used when ``use_filename`` is
    ``True``. Images without a resolvable caption are skipped.
    """
    results: List[str] = []
    image_files: List[str] = []

    for root, _, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, file))

    print(f"Total {len(image_files)} image files found.")

    for img_path in image_files:
        if text_dict and os.path.basename(img_path) in text_dict:
            text = text_dict[os.path.basename(img_path)]
        elif use_filename:
            text = os.path.basename(img_path)
        else:
            continue

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            rel_path = os.path.relpath(img_path, image_dir)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        else:
            out_path = None

        result = add_text_to_image(img_path, text, out_path, **kwargs)
        if result:
            results.append(result)

    print(f"\nTotal {len(results)} images processed!")
    return results


def batch_add_frame_numbers(
    root_dir: str,
    output_dir: Optional[str] = None,
    font_size: Optional[int] = None,
    auto_font_div: int = 15,
    text_color: Tuple[int, int, int] = (255, 0, 0),
    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 200),
    position: str = 'top-left',
    skip_existing: bool = True,
    font_path: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """Label every sub-directory of ``root_dir`` as a per-video frame folder.

    For each immediate sub-directory of ``root_dir``:
      * frames are listed and sorted lexicographically,
      * the i-th frame is captioned ``"frame #ii"`` (1-based, zero-padded),
      * the captioned image is written to the mirrored sub-directory under
        ``output_dir``.

    When ``skip_existing`` is true, frames whose output file already exists
    are left untouched (resume-friendly).
    """
    results: List[str] = []

    if output_dir is None:
        font_suffix = 'auto' if font_size is None else str(font_size)
        output_dir = root_dir.rstrip('/') + f'_{font_suffix}15_1'

    subdirs: List[str] = sorted(
        item for item in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, item))
    )
    print(f"Found {len(subdirs)} subdirectories.")

    total_processed = 0
    total_skipped = 0

    for subdir in tqdm(subdirs, desc="Progress", ncols=100):
        subdir_path = os.path.join(root_dir, subdir)

        image_files = sorted(
            f for f in os.listdir(subdir_path)
            if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
        )

        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)

        dir_processed = 0
        dir_skipped = 0

        for idx, filename in enumerate(image_files, 1):
            input_path = os.path.join(subdir_path, filename)
            output_path = os.path.join(output_subdir, filename)

            if skip_existing and os.path.exists(output_path):
                dir_skipped += 1
                total_skipped += 1
                continue

            text = f"frame #{idx:02d}"
            result = add_text_to_image(
                input_path,
                text,
                output_path,
                font_size=font_size,
                auto_font_div=auto_font_div,
                text_color=text_color,
                bg_color=bg_color,
                position=position,
                font_path=font_path,
                verbose=False,
                **kwargs,
            )
            if result:
                results.append(result)
                dir_processed += 1
                total_processed += 1

        if dir_skipped > 0:
            tqdm.write(f"  [{subdir}] Processed: {dir_processed}, Skipped: {dir_skipped}")

    print("\n" + "=" * 80)
    print(f"Total: {total_processed} processed, {total_skipped} skipped")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    return results


# -- CLI ---------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Add numeric "frame #NN" captions (background-box style) to images.'
    )
    parser.add_argument('--root_dir', type=str, default="",
                        help='Input directory containing per-video sub-directories of frames.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <root_dir>_auto15_1).')
    parser.add_argument('--font_size', type=int, default=None,
                        help='Font size in pixels. Default: auto from min(width, height) // auto_font_div.')
    parser.add_argument('--auto-font-div', type=int, default=15, dest='auto_font_div',
                        help='Divisor used to derive auto font size (default: 15).')
    parser.add_argument('--position', type=str, default='top-left',
                        choices=['top-left', 'top-right', 'bottom-left', 'bottom-right'],
                        help='Caption corner (default: top-left).')
    parser.add_argument('--font', type=str, default=None, dest='font_path',
                        help='Path to a .ttf/.otf font file. Default: PIL built-in font.')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip frames whose output file already exists (default: True).')
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                        help='Re-render existing output files.')
    parser.add_argument('--text_color', type=str, default=None,
                        help='Text color "r,g,b" (default: "255,0,0" -- red).')
    parser.add_argument('--background_color', type=str, default=None,
                        help='Background color "r,g,b,a" (default: "255,255,255,200" -- white, ~78% opaque).')
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    text_color: Tuple[int, int, int] = (255, 0, 0)
    if args.text_color:
        parsed = parse_color(args.text_color)
        if len(parsed) != 3:
            raise ValueError("--text_color must be in R,G,B format (3 values)")
        text_color = parsed  # type: ignore[assignment]

    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 200)
    if args.background_color:
        parsed = parse_color(args.background_color)
        bg_color = (parsed if len(parsed) == 4 else (*parsed, 200))  # type: ignore[assignment]

    results = batch_add_frame_numbers(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        font_size=args.font_size,
        auto_font_div=args.auto_font_div,
        text_color=text_color,
        bg_color=bg_color,
        position=args.position,
        skip_existing=args.skip_existing,
        font_path=args.font_path,
    )
    print(f"\nTotal {len(results)} files created")
