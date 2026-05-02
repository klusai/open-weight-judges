"""Open-weight multi-judge panel evaluation framework."""

__all__ = ["Judge", "JudgeConfig", "JudgeCallResult"]


def __getattr__(name):
    if name in __all__:
        from judges.judge import Judge, JudgeConfig, JudgeCallResult
        return {"Judge": Judge, "JudgeConfig": JudgeConfig, "JudgeCallResult": JudgeCallResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
