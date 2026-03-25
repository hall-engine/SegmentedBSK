from Basilisk.architecture import messaging # ensure recorders() work without first importing messaging
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version('bsk')
except Exception:
    __version__ = '0.0.0'
