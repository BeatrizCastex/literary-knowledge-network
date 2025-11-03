#!/usr/bin/env python3
"""Generate word clouds for the 10 largest literary communities."""

from collections.abc import Iterable
from pathlib import Path
from typing import Iterable as TypingIterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from PIL import ImageDraw
from wordcloud import WordCloud


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
WORK_PATHS: TypingIterable[Path] = (
    DATA_DIR / "processed/enriched/work_final.parquet",
    DATA_DIR / "processed/enriched/work_final.csv",
)
COMMUNITY_PATHS: TypingIterable[Path] = (
    DATA_DIR / "output/communities/work_communities.parquet",
    DATA_DIR / "output/communities/work_communities.csv",
)


def load_frame(candidates: Iterable[Path]) -> pd.DataFrame:
    for path in candidates:
        if path.suffix == ".parquet" and path.exists():
            return pd.read_parquet(path)
        if path.suffix == ".csv" and path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"None of the candidate files exist: {', '.join(str(p) for p in candidates)}")


def flatten_keywords(values: pd.Series) -> List[str]:
    tokens: List[str] = []
    for value in values.dropna():
        if isinstance(value, str):
            tokens.extend(value.replace("[", "").replace("]", "").replace(",", " ").split())
        elif isinstance(value, Iterable):
            for item in value:
                if isinstance(item, str):
                    tokens.extend(item.split())
                elif item:
                    tokens.append(str(item))
        elif value:
            tokens.append(str(value))
    return tokens


def resolve_font_path() -> str:
    candidates: List[Optional[str]] = []
    try:
        candidates.append(font_manager.findfont("DejaVu Sans", fallback_to_default=True))
    except Exception:
        candidates.append(None)

    mpl_font = Path(matplotlib.__file__).resolve().parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
    candidates.append(str(mpl_font))
    candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    candidates.append("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and path.suffix.lower() in {".ttf", ".ttc"}:
            return str(path)

    raise RuntimeError(
        "Could not locate a TrueType font for word cloud rendering. "
        "Install DejaVu Sans (ttf) or update the script with a valid font_path."
    )


def main() -> None:
    works = load_frame(WORK_PATHS)
    communities = load_frame(COMMUNITY_PATHS)

    merged = communities.merge(works, on="work_id", how="left")

    cluster_sizes = (
        merged.groupby("cluster_id")["work_id"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )
    top_clusters = cluster_sizes.index.tolist()

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()

    font_path = resolve_font_path()
    print(f"[info] Using font: {font_path}")

    for i, cid in enumerate(top_clusters):
        subset = merged[merged["cluster_id"] == cid]
        keywords = flatten_keywords(subset.get("keywords", pd.Series(dtype=object)))
        if not keywords:
            axes[i].text(0.5, 0.5, "No keywords", ha="center", va="center")
            axes[i].axis("off")
            axes[i].set_title(f"Cluster {cid}", fontsize=11)
            continue

        wc = WordCloud(
            width=400,
            height=300,
            background_color="white",
            colormap="plasma",
            collocations=False,
            font_path=font_path,
        ).generate(" ".join(keywords))

        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Cluster {cid}  (n={len(subset)})", fontsize=11)
        axes[i].axis("off")

    fig.suptitle(
        "Figure 2 – Keyword Centroids of the 10 Largest Literary Communities",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    out_dir = DATA_DIR / "output/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "figure2_keyword_centroids.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Word clouds saved to: {output_path}")


if __name__ == "__main__":
    main()
_ORIGINAL_TEXTBBOX = ImageDraw.ImageDraw.textbbox


def safe_textbbox(self: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font=None, anchor=None, *args, **kwargs):
    try:
        return _ORIGINAL_TEXTBBOX(self, xy, text, font=font, anchor=anchor, *args, **kwargs)
    except ValueError:
        width, height = self.textsize(text, font=font)
        x, y = xy
        return x, y, x + width, y + height


ImageDraw.ImageDraw.textbbox = safe_textbbox  # type: ignore[assignment]
ImageDraw.Draw.textbbox = safe_textbbox  # type: ignore[attr-defined]
ImageDraw.textbbox = safe_textbbox  # type: ignore[assignment]
