"""Utility helpers for simulating trade recommendations based on model forecasts."""

from .simulator import generate_trade_report, serialise_trade_report

__all__ = ["generate_trade_report", "serialise_trade_report"]
