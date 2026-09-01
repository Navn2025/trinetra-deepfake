"""Compatibility shim: fairseq's last PyPI release (0.12.2, ~2022) predates
Python 3.11's stricter dataclasses rules. Dozens of files across fairseq
(config classes, and essentially every model/task/criterion module, since
fairseq eagerly imports all of them on `import fairseq`) declare a dataclass
field like `x: SomeConfig = SomeConfig()` or `x: SomeConfig =
field(default=SomeConfig())` -- both are a mutable default, which Python
3.11 now rejects unless expressed as `field(default_factory=SomeConfig)`.

Patching each file by hand doesn't scale (this is not a one-off, it's the
same pattern repeated throughout the whole package). This monkeypatches
`dataclasses.dataclass` to rewrite that pattern automatically wherever it's
found, immediately before `import fairseq` triggers it. Must be imported
before any fairseq import.
"""
import copy
import dataclasses

_real_dataclass = dataclasses.dataclass
_MISSING = dataclasses.MISSING
_patched = False


def _default_factory_field(default_value):
    return dataclasses.field(default_factory=lambda dv=default_value: copy.deepcopy(dv))


def _patched_dataclass(cls=None, **kwargs):
    def wrap(cls):
        for name in list(getattr(cls, "__annotations__", {}).keys()):
            if name not in cls.__dict__:
                continue
            val = cls.__dict__[name]
            if isinstance(val, dataclasses.Field):
                if (
                    val.default is not _MISSING
                    and val.default_factory is _MISSING
                    and dataclasses.is_dataclass(val.default)
                    and not isinstance(val.default, type)
                ):
                    setattr(cls, name, _default_factory_field(val.default))
            elif dataclasses.is_dataclass(val) and not isinstance(val, type):
                setattr(cls, name, _default_factory_field(val))
        return _real_dataclass(cls, **kwargs)

    if cls is None:
        return wrap
    return wrap(cls)


def apply():
    global _patched
    if not _patched:
        dataclasses.dataclass = _patched_dataclass
        _patched = True
