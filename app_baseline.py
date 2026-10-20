"""Streamlit entry point for the naive comparison model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import main  # noqa: E402

NAIVE_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/naive_best.pt"


if __name__ == "__main__":
    main(
        default_checkpoint=NAIVE_CHECKPOINT,
        model_label="Naive 모델 · 공정성 정규화 없음",
        page_title="Naive 영화 추천 비교",
        model_notice=(
            "이 모델은 공정성 정규화를 적용하지 않은 비교용 모델입니다. "
            "추천 입력에는 공정성 정규화 후보와 동일하게 좋아한 영화만 사용합니다."
        ),
        expected_model_type="multvae",
        audit_role="naive",
    )
