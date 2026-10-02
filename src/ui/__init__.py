from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui.app import build_app, launch_ui
    from src.ui.widget import load_saved_tier_models, render_inference_widget


def __getattr__(name: str):
    if name in ("build_app", "launch_ui"):
        from src.ui.app import build_app, launch_ui
        return {"build_app": build_app, "launch_ui": launch_ui}[name]
    if name in ("load_saved_tier_models", "render_inference_widget"):
        from src.ui.widget import load_saved_tier_models, render_inference_widget
        return {
            "load_saved_tier_models": load_saved_tier_models,
            "render_inference_widget": render_inference_widget,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_app",
    "launch_ui",
    "load_saved_tier_models",
    "render_inference_widget",
]
