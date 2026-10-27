"""Independent structural solver adapters; no thermal/EM coupling."""

from .calculix import CalculixRunner, RunResult, parse_dat, parse_frd, parse_sta

__all__ = ["CalculixRunner", "RunResult", "parse_dat", "parse_frd", "parse_sta"]
