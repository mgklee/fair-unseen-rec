#!/usr/bin/env python3
"""Exercise the complete gender-free, two-screen Streamlit flow headlessly."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINIMUM_LIKES = 3
EXPECTED_RECOMMENDATIONS = 10


def button_by_label(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def visible_text(app: AppTest) -> str:
    elements: Iterable[object] = [
        *app.markdown,
        *app.caption,
        *app.info,
        *app.warning,
        *app.success,
    ]
    return " ".join(str(getattr(element, "value", "")) for element in elements)


def assert_no_gender_widget(app: AppTest) -> None:
    """Reject demographic inputs while allowing explanatory fairness copy."""

    forbidden_types = ("selectbox", "radio", "checkbox", "toggle")
    for widget_type in forbidden_types:
        widgets = list(getattr(app, widget_type))
        if widgets:
            raise AssertionError(f"Unexpected {widget_type} input: {widgets}")
    for widget in [*app.multiselect, *app.text_input, *app.number_input]:
        label = str(getattr(widget, "label", "")).casefold()
        if "성별" in label or "gender" in label:
            raise AssertionError(f"Gender input leaked into the UI: {label!r}")


def run_smoke(
    app_path: Path,
    *,
    expected_checkpoint: Path | None = None,
    expected_model_label: str | None = None,
    expected_notice: str | None = None,
) -> int:
    """Exercise onboarding, results, edit, and reset for one entry point."""

    app = AppTest.from_file(app_path, default_timeout=60).run()
    if app.exception:
        raise RuntimeError(f"App startup exceptions: {app.exception}")
    assert_no_gender_widget(app)

    if expected_checkpoint is not None:
        actual_checkpoint = Path(app.session_state["active_checkpoint"]).resolve()
        if actual_checkpoint != expected_checkpoint.resolve():
            raise AssertionError(
                f"Expected checkpoint {expected_checkpoint}, got {actual_checkpoint}"
            )
    if expected_model_label is not None:
        actual_label = app.session_state["active_model_label"]
        if actual_label != expected_model_label:
            raise AssertionError(
                f"Expected model label {expected_model_label!r}, got {actual_label!r}"
            )
        if expected_model_label not in visible_text(app):
            raise AssertionError("The active model identity was not visible")
    if expected_notice is not None and expected_notice not in visible_text(app):
        raise AssertionError("The model notice was not visible on onboarding")

    if len(app.multiselect) != 1:
        raise AssertionError("Onboarding must expose one searchable movie selector")
    recommend = button_by_label(app, "내 취향 분석하고 추천받기 →")
    if not recommend.disabled:
        raise AssertionError("Recommendation must be disabled before three likes")

    options = list(app.multiselect[0].options[:MINIMUM_LIKES])
    if len(options) != MINIMUM_LIKES:
        raise AssertionError("The local catalog did not expose three selectable movies")
    app.multiselect[0].set_value(options).run()
    if app.exception:
        raise RuntimeError(f"Movie selection exceptions: {app.exception}")
    assert_no_gender_widget(app)
    recommend = button_by_label(app, "내 취향 분석하고 추천받기 →")
    if recommend.disabled:
        raise AssertionError("Recommendation stayed disabled after three likes")

    recommend.click().run()
    if app.exception:
        raise RuntimeError(f"Recommendation exceptions: {app.exception}")
    assert_no_gender_widget(app)
    recommendations = app.session_state["recommendation_rows"]
    if app.session_state["ui_page"] != "results":
        raise AssertionError("The recommendation action did not open results")
    if len(recommendations) != EXPECTED_RECOMMENDATIONS:
        raise AssertionError("The results screen did not contain ten recommendations")
    selected_ids = set(app.session_state["saved_movie_selection"])
    recommended_ids = {int(row["movie_id"]) for row in recommendations}
    if selected_ids.intersection(recommended_ids):
        raise AssertionError("A liked movie leaked into recommendations")

    results_text = visible_text(app)
    if "당신을 위한 추천 10편" not in results_text:
        raise AssertionError("The second-screen heading was not rendered")
    if "개인의 성별을 추론하지 않습니다" not in results_text:
        raise AssertionError("The gender-free inference disclosure was not rendered")
    if len(app.metric) != EXPECTED_RECOMMENDATIONS or "genre-chip" not in results_text:
        raise AssertionError("Recommendation cards, scores, or genres were not rendered")
    if expected_notice is not None and expected_notice not in results_text:
        raise AssertionError("The model notice was not visible on results")

    button_by_label(app, "← 선택 수정하기").click().run()
    if app.exception or app.session_state["ui_page"] != "setup":
        raise RuntimeError("The edit action did not return to onboarding")
    assert_no_gender_widget(app)
    if len(app.multiselect[0].value) != MINIMUM_LIKES:
        raise AssertionError("The edit action did not preserve liked movies")

    button_by_label(app, "처음부터").click().run()
    if app.exception:
        raise RuntimeError(f"Reset exceptions: {app.exception}")
    assert_no_gender_widget(app)
    if app.multiselect[0].value:
        raise AssertionError("Reset did not clear liked movies")
    if not button_by_label(app, "내 취향 분석하고 추천받기 →").disabled:
        raise AssertionError("Recommendation must be disabled again after reset")

    print(
        "Streamlit gender-free two-screen smoke test passed with "
        f"{MINIMUM_LIKES} likes and {EXPECTED_RECOMMENDATIONS} recommendations "
        f"({app_path.name})."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=PROJECT_ROOT / "app.py")
    parser.add_argument("--expected-checkpoint", type=Path)
    parser.add_argument("--expected-model-label")
    parser.add_argument("--expected-notice")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    selected_app = arguments.app
    if not selected_app.is_absolute():
        selected_app = PROJECT_ROOT / selected_app
    raise SystemExit(
        run_smoke(
            selected_app,
            expected_checkpoint=arguments.expected_checkpoint,
            expected_model_label=arguments.expected_model_label,
            expected_notice=arguments.expected_notice,
        )
    )
