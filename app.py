"""Korean two-screen Streamlit UI for gender-free movie recommendations.

The website intentionally collects only a small set of liked movies.  Gender
is neither requested nor sent to inference.  A few compatibility helpers in
this module isolate the UI from small representation differences in the
``fairrec.inference`` API; model and data code stay unaware of Streamlit.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fairrec.inference import MovieRecommender  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/fair_best.pt"
DEFAULT_DATA = PROJECT_ROOT / "data/processed/movielens_1m.npz"
DEFAULT_AUDIT = PROJECT_ROOT / "artifacts/metrics/comparison.json"
MIN_LIKES = 3
MAX_LIKES = 20
TOP_K = 10

GENRE_LABELS = {
    "Action": "액션",
    "Adventure": "모험",
    "Animation": "애니메이션",
    "Children's": "어린이",
    "Comedy": "코미디",
    "Crime": "범죄",
    "Documentary": "다큐멘터리",
    "Drama": "드라마",
    "Fantasy": "판타지",
    "Film-Noir": "필름 누아르",
    "Horror": "공포",
    "Musical": "뮤지컬",
    "Mystery": "미스터리",
    "Romance": "로맨스",
    "Sci-Fi": "SF",
    "Thriller": "스릴러",
    "War": "전쟁",
    "Western": "서부극",
}


def _read_field(value: Any, *names: str, default: Any = None) -> Any:
    """Read a field from either a dataclass-like object or a mapping."""

    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _normalise_genres(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split("|") if part.strip())
    return tuple(str(part) for part in value)


def _display_title(title: str) -> str:
    """Move MovieLens suffix articles to the front for natural display."""

    match = re.match(r"^(.*), (The|A|An)$", title.strip())
    return f"{match.group(2)} {match.group(1)}" if match else title.strip()


def _movie_view(value: Any, *, with_score: bool = False) -> dict[str, Any]:
    """Convert public inference records into one stable, UI-only shape."""

    movie_id = int(_read_field(value, "movie_id", "id"))
    title = _display_title(str(_read_field(value, "title", "display_title", default="")))
    year_value = _read_field(value, "year")
    year = int(year_value) if year_value not in (None, -1, 0, "") else None
    genres = _normalise_genres(_read_field(value, "genres", default=()))
    output: dict[str, Any] = {
        "movie_id": movie_id,
        "title": title,
        "year": year,
        "genres": genres,
    }
    if with_score:
        score = _read_field(value, "score", "predicted_score", default=0.0)
        output["score"] = float(score)
    return output


def _engine_movies(engine: MovieRecommender) -> list[dict[str, Any]]:
    """Return the local catalog while keeping API assumptions in one place."""

    records = getattr(engine, "movies", None)
    if callable(records):
        records = records()
    if records is None:  # Backward-compatible fallback for an older prototype.
        records = getattr(engine, "catalog")
    movies = [_movie_view(record) for record in records]
    return sorted(movies, key=lambda movie: (movie["title"].casefold(), movie["movie_id"]))


def _make_engine(checkpoint_path: str, data_path: str) -> MovieRecommender:
    """Construct the engine through its public checkpoint API.

    The released API supports ``MovieRecommender(checkpoint, data)``.  The
    development implementation also exposes ``from_checkpoint``; supporting it
    here keeps that construction detail out of every UI call site.
    """

    factory = getattr(MovieRecommender, "from_checkpoint", None)
    if callable(factory):
        try:
            return factory(checkpoint_path, data_path, device="cpu")
        except TypeError:
            return factory(checkpoint_path, data_path)
    return MovieRecommender(checkpoint_path, data_path)


def _recommend(
    engine: MovieRecommender,
    liked_movie_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Call gender-free inference and normalise recommendation records."""

    liked = [int(movie_id) for movie_id in liked_movie_ids]
    try:
        rows = engine.recommend(liked, k=TOP_K, exclude_movie_ids=liked)
    except TypeError:
        # Compatibility with the direct-constructor API spelling.
        rows = engine.recommend(liked, top_k=TOP_K)
    liked_set = set(liked)
    recommendations = [
        _movie_view(row, with_score=True)
        for row in rows
        if int(_read_field(row, "movie_id", "id")) not in liked_set
    ]
    return recommendations[:TOP_K]


@st.cache_resource(show_spinner="추천 모델을 불러오는 중이에요…")
def load_engine(checkpoint_path: str, data_path: str) -> MovieRecommender:
    """Cache immutable model/catalog state; choices never enter this cache."""

    return _make_engine(checkpoint_path, data_path)


@st.cache_data(show_spinner=False)
def load_offline_audit(path: str) -> Any:
    """Load a static aggregate report; it is never computed from this visitor."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reset_session() -> None:
    """Erase all transient values for this browser session."""

    for key in list(st.session_state):
        del st.session_state[key]


def go_to_setup() -> None:
    """Return to onboarding while retaining the current liked movies."""

    st.session_state["movie_selection"] = list(
        st.session_state.get("saved_movie_selection", [])
    )
    st.session_state["ui_page"] = "setup"


def inject_styles() -> None:
    """Apply a compact cinema-inspired visual system without remote assets."""

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        .hero {
            padding: 1.45rem 1.55rem;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(99, 102, 241, .28);
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(49, 46, 129, .30), rgba(15, 23, 42, .12));
        }
        .hero h1 {
            margin: 0 0 .4rem 0;
            font-size: clamp(1.85rem, 4vw, 2.65rem);
        }
        .hero p { margin: 0; opacity: .84; font-size: 1.02rem; }
        .steps {
            display: flex;
            align-items: center;
            gap: .65rem;
            margin: .25rem 0 1.1rem 0;
            font-size: .9rem;
        }
        .step {
            padding: .38rem .74rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, .14);
            color: rgb(148, 163, 184);
        }
        .step.active { color: white; background: rgb(79, 70, 229); font-weight: 700; }
        .step-line { width: 32px; height: 1px; background: rgba(148, 163, 184, .42); }
        .model-badge {
            display: inline-block;
            margin: 0 0 .85rem 0;
            padding: .32rem .7rem;
            border-radius: 999px;
            color: rgb(199, 210, 254);
            background: rgba(99, 102, 241, .14);
            border: 1px solid rgba(99, 102, 241, .3);
            font-size: .81rem;
            font-weight: 700;
        }
        .genre-chip {
            display: inline-block;
            margin: .12rem .28rem .12rem 0;
            padding: .2rem .52rem;
            border-radius: 999px;
            background: rgba(99, 102, 241, .12);
            border: 1px solid rgba(99, 102, 241, .25);
            font-size: .78rem;
        }
        .rank {
            color: rgb(129, 140, 248);
            font-weight: 800;
            letter-spacing: .045em;
            margin-bottom: .2rem;
        }
        .selected-title { font-weight: 700; margin-bottom: .1rem; }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_steps(current: int) -> None:
    first_class = "step active" if current == 1 else "step"
    second_class = "step active" if current == 2 else "step"
    st.markdown(
        f"""
        <div class="steps" aria-label="추천 진행 단계">
          <span class="{first_class}">1&nbsp; 좋아한 영화</span>
          <span class="step-line"></span>
          <span class="{second_class}">2&nbsp; 추천 결과</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_identity(model_label: str, model_notice: str | None) -> None:
    st.markdown(
        f'<span class="model-badge">{html.escape(model_label)}</span>',
        unsafe_allow_html=True,
    )
    if model_notice:
        st.info(model_notice)


def movie_label(movie: Mapping[str, Any]) -> str:
    year = f" ({movie['year']})" if movie.get("year") else ""
    genres = " · ".join(
        GENRE_LABELS.get(str(genre), str(genre)) for genre in movie["genres"][:3]
    )
    suffix = f" — {genres}" if genres else ""
    return f"{movie['title']}{year}{suffix}"


def genre_chips(genres: Sequence[str]) -> str:
    return " ".join(
        f'<span class="genre-chip">{html.escape(GENRE_LABELS.get(genre, genre))}</span>'
        for genre in genres
    )


def render_audit(
    audit_path: Path,
    *,
    audit_role: str,
    checkpoint_sha256: str,
) -> None:
    """Render an aggregate audit only when it matches the active checkpoint."""

    if not audit_path.is_file():
        return
    with st.expander("오프라인 전체 테스트 공정성 감사 보기"):
        st.caption(
            "아래 값은 미리 분리한 MovieLens 테스트 집단에서 계산한 aggregate 결과입니다. "
            "현재 방문자의 정보로 계산하거나 개인의 속성을 추론하지 않습니다."
        )
        try:
            audit = load_offline_audit(str(audit_path.resolve()))
        except (OSError, ValueError, json.JSONDecodeError):
            st.warning("저장된 감사 결과를 읽지 못했어요.")
            return
        deployment = audit.get("deployment", {}) if isinstance(audit, Mapping) else {}
        expected = deployment.get(audit_role, {}) if isinstance(deployment, Mapping) else {}
        recorded_hash = expected.get("checkpoint_sha256") if isinstance(expected, Mapping) else None
        if recorded_hash != checkpoint_sha256:
            st.warning(
                "이 감사 파일은 현재 로드한 체크포인트와 일치하지 않아 표시하지 않습니다."
            )
            return
        st.json(audit, expanded=False)


def render_setup(
    engine: MovieRecommender,
    model_label: str,
    model_notice: str | None,
) -> None:
    """Render step one: select at least three locally catalogued liked movies."""

    render_steps(1)
    render_model_identity(model_label, model_notice)
    st.markdown(
        """
        <div class="hero">
          <h1>🎬 좋아한 영화를 골라주세요</h1>
          <p>재미있게 본 영화 3편 이상을 알려주면, 아직 고르지 않은 영화 중 취향에 맞는 10편을 찾아드려요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 영화 제목으로 검색하고 선택하세요")
    st.caption(
        "MovieLens 1M 로컬 카탈로그 안에서만 검색합니다. 별점이나 성별 정보는 필요하지 않아요."
    )
    movies = _engine_movies(engine)
    catalog_by_id = {int(movie["movie_id"]): movie for movie in movies}
    options = list(catalog_by_id)
    selected_ids = st.multiselect(
        f"좋아한 영화 선택 (최소 {MIN_LIKES}편)",
        options=options,
        format_func=lambda movie_id: movie_label(catalog_by_id[int(movie_id)]),
        max_selections=MAX_LIKES,
        placeholder="예: Toy Story, The Matrix, Titanic…",
        key="movie_selection",
    )

    if selected_ids:
        st.markdown("#### 선택한 영화")
        columns = st.columns(2)
        for position, movie_id in enumerate(selected_ids):
            movie = catalog_by_id[int(movie_id)]
            with columns[position % 2]:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="selected-title">✓ {html.escape(movie["title"])}</div>',
                        unsafe_allow_html=True,
                    )
                    year_text = str(movie["year"]) if movie.get("year") else "연도 정보 없음"
                    st.caption(year_text)
                    st.markdown(genre_chips(movie["genres"]), unsafe_allow_html=True)
    else:
        st.info("검색창에서 좋아한 영화를 선택하면 이곳에 바로 표시돼요.")

    selected_count = len(selected_ids)
    st.progress(
        min(selected_count / MIN_LIKES, 1.0),
        text=f"선택 완료 {selected_count}/{MIN_LIKES}편",
    )
    ready = selected_count >= MIN_LIKES
    if not ready:
        st.caption(f"추천을 시작하려면 {MIN_LIKES - selected_count}편을 더 선택해 주세요.")
    else:
        st.success("취향 입력 완료! 이제 영화를 추천받을 수 있어요.")

    action_left, action_right = st.columns([4, 1])
    with action_left:
        recommend_clicked = st.button(
            "내 취향 분석하고 추천받기 →",
            type="primary",
            width="stretch",
            disabled=not ready,
        )
    with action_right:
        st.button("처음부터", on_click=reset_session, width="stretch")

    if recommend_clicked:
        with st.spinner("좋아한 영화의 패턴을 분석하고 있어요…"):
            try:
                recommendations = _recommend(engine, selected_ids)
            except Exception:
                st.error("추천을 만드는 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요.")
                return
        if len(recommendations) != TOP_K:
            st.error("추천 결과가 충분하지 않아요. 다른 영화를 추가한 뒤 다시 시도해 주세요.")
            return
        st.session_state["saved_movie_selection"] = [int(value) for value in selected_ids]
        st.session_state["recommendation_rows"] = recommendations
        st.session_state["recommendation_like_count"] = selected_count
        st.session_state["ui_page"] = "results"
        st.rerun()


def render_results(
    model_label: str,
    model_notice: str | None,
    audit_path: Path,
    audit_role: str,
    checkpoint_sha256: str,
) -> None:
    """Render step two as ten scannable, gender-free recommendation cards."""

    recommendations = st.session_state.get("recommendation_rows")
    if not recommendations:
        st.session_state["ui_page"] = "setup"
        st.rerun()

    render_steps(2)
    render_model_identity(model_label, model_notice)
    st.markdown(
        """
        <div class="hero">
          <h1>🍿 당신을 위한 추천 10편</h1>
          <p>좋아한 영화는 제외하고, 선택한 작품들과 취향이 잘 맞는 순서로 골랐어요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    liked_count = int(st.session_state.get("recommendation_like_count", 0))
    st.caption(
        f"좋아한 영화 {liked_count}편만 반영한 세션 전용 추천입니다. "
        "성별을 입력받거나 개인의 성별을 추론하지 않습니다."
    )

    action_left, action_right, spacer = st.columns([1.3, 1, 3])
    with action_left:
        st.button("← 선택 수정하기", on_click=go_to_setup, width="stretch")
    with action_right:
        st.button("처음부터", on_click=reset_session, width="stretch")
    with spacer:
        st.empty()

    cards = st.columns(2)
    for position, recommendation in enumerate(recommendations, start=1):
        with cards[(position - 1) % 2]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="rank">TOP {position}</div>', unsafe_allow_html=True
                )
                st.markdown(f"### {recommendation['title']}")
                year_text = (
                    str(recommendation["year"])
                    if recommendation.get("year")
                    else "연도 정보 없음"
                )
                st.caption(year_text)
                st.markdown(
                    genre_chips(recommendation["genres"]), unsafe_allow_html=True
                )
                st.metric("추천 점수", f"{float(recommendation['score']):.3f}")

    render_audit(
        audit_path,
        audit_role=audit_role,
        checkpoint_sha256=checkpoint_sha256,
    )


def main(
    *,
    default_checkpoint: Path = DEFAULT_CHECKPOINT,
    model_label: str = "공정성 정규화 후보 · SoftGBS + GRL",
    page_title: str = "성별 공정성 감사 영화 추천",
    model_notice: str | None = None,
    expected_model_type: str = "adversarial_multvae",
    audit_role: str = "fair",
) -> None:
    """Run the shared website with one explicitly selected trained model."""

    st.set_page_config(page_title=page_title, page_icon="🎬", layout="wide")
    inject_styles()

    checkpoint = Path(os.environ.get("FAIRREC_CHECKPOINT", str(default_checkpoint)))
    data_path = Path(os.environ.get("FAIRREC_DATA", str(DEFAULT_DATA)))
    audit_path = Path(os.environ.get("FAIRREC_AUDIT", str(DEFAULT_AUDIT)))
    active_model_label = os.environ.get("FAIRREC_MODEL_LABEL", model_label)

    if not checkpoint.is_file():
        st.error(f"학습된 모델을 찾을 수 없어요: {checkpoint}")
        st.stop()
    if not data_path.is_file():
        st.error(f"MovieLens 가공 데이터를 찾을 수 없어요: {data_path}")
        st.stop()
    try:
        engine = load_engine(str(checkpoint.resolve()), str(data_path.resolve()))
    except Exception as exc:
        st.error(f"추천 모델을 불러오지 못했어요: {exc}")
        st.stop()
    if getattr(engine, "model_type", None) != expected_model_type:
        st.error(
            "화면과 체크포인트 유형이 일치하지 않습니다: "
            f"expected={expected_model_type}, actual={getattr(engine, 'model_type', None)}"
        )
        st.stop()

    st.session_state["active_checkpoint"] = str(checkpoint.resolve())
    st.session_state["active_model_label"] = active_model_label
    st.session_state["active_audit"] = str(audit_path.resolve())
    st.session_state["active_model_type"] = engine.model_type
    st.session_state["active_checkpoint_sha256"] = engine.checkpoint_sha256
    st.session_state.setdefault("ui_page", "setup")

    if st.session_state["ui_page"] == "results":
        render_results(
            active_model_label,
            model_notice,
            audit_path,
            audit_role,
            engine.checkpoint_sha256,
        )
    else:
        render_setup(engine, active_model_label, model_notice)

    st.divider()
    st.caption(
        "🔒 선택한 영화는 이 브라우저 세션 안에서만 사용되며 저장되지 않아요. "
        "이 서비스는 추천 입력으로 성별을 요청하거나 사용하지 않습니다."
    )


if __name__ == "__main__":
    main()
