__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from .api import app as _app

        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
