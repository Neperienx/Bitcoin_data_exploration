"""Simple trade bot simulation helpers.

The module analyses a sequence of forecast candles produced by the models
and generates a list of trade recommendations (buy/sell/hold).  If historical
"ground truth" candles are provided the simulator also evaluates the
profit-and-loss that would have resulted from following the plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, MutableMapping, Optional

Number = float | int


@dataclass
class TradeBotState:
    """Internal state of the simulated bot."""

    cash: float = 100.0
    position: float = 0.0
    entry_price: Optional[float] = None

    def clone(self) -> "TradeBotState":
        return TradeBotState(self.cash, self.position, self.entry_price)


def _coerce_float(value: Number | None, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalise_timestamp(value) -> dict:
    timestamp: Optional[int] = None
    datetime_str: Optional[str] = None
    if isinstance(value, MutableMapping):
        timestamp = _coerce_float(value.get("timestamp"), 0.0)
        raw_dt = value.get("datetime")
    else:
        timestamp = _coerce_float(getattr(value, "timestamp", 0.0), 0.0)
        raw_dt = getattr(value, "datetime", None)
    if raw_dt is None and isinstance(value, MutableMapping):
        raw_dt = value.get("datetime")
    if hasattr(raw_dt, "isoformat"):
        datetime_str = raw_dt.isoformat()
    elif raw_dt is not None:
        datetime_str = str(raw_dt)
    return {
        "timestamp": int(round(timestamp)),
        "datetime": datetime_str,
    }


def _normalise_candle(candle: MutableMapping[str, Number]) -> dict:
    timestamp_info = _normalise_timestamp(candle)
    return {
        **timestamp_info,
        "open": _coerce_float(candle.get("open")),
        "close": _coerce_float(candle.get("close"), candle.get("open", 0.0)),
        "probability": _coerce_float(candle.get("probability"), 0.5),
        "expected_return": _coerce_float(candle.get("expected_return")),
    }


def _summarise_orders(orders: List[dict]) -> str:
    if not orders:
        return "Bot would stay in cash for the forecast horizon."
    actions = []
    for order in orders:
        action = order.get("action", "hold").upper()
        when = order.get("datetime") or f"step {order.get('step', 0) + 1}"
        actions.append(f"{action} @ {when}")
    return " · ".join(actions)


def _evaluate_against_truth(
    orders: List[dict],
    ground_truth: List[MutableMapping[str, Number]],
    initial_cash: float,
) -> dict:
    if not ground_truth:
        return {"available": False}

    cash = float(initial_cash)
    position = 0.0
    avg_entry = 0.0
    realized_pl = 0.0

    for order in orders:
        step = int(order.get("step", -1))
        if step < 0 or step >= len(ground_truth):
            continue
        candle = ground_truth[step]
        price = _coerce_float(candle.get("open")) or _coerce_float(
            candle.get("close")
        )
        if price <= 0:
            continue

        if order.get("action") == "buy" and cash > 0:
            units = cash / price
            spend = units * price
            cash -= spend
            position += units
            avg_entry = price
        elif order.get("action") == "sell" and position > 0:
            proceeds = position * price
            cost = position * avg_entry
            cash += proceeds
            realized_pl += proceeds - cost
            position = 0.0
            avg_entry = 0.0

    last_index = len(ground_truth) - 1
    if orders:
        last_index = min(last_index, max(int(o.get("step", 0)) for o in orders))
    last_index = max(0, last_index)
    last_candle = ground_truth[last_index]
    final_price = _coerce_float(last_candle.get("close") or last_candle.get("open"))
    unrealized = position * (final_price - avg_entry)
    total_value = cash + position * final_price

    return {
        "available": True,
        "realized_pl": realized_pl,
        "unrealized_pl": unrealized,
        "total_value": total_value,
        "final_cash": cash,
        "final_position": position,
        "final_price": final_price,
    }


def generate_trade_report(
    forecast: Iterable[MutableMapping[str, Number]],
    *,
    ground_truth: Optional[Iterable[MutableMapping[str, Number]]] = None,
    initial_cash: float = 100.0,
    simulation_steps: Optional[int] = None,
) -> dict:
    """Generate trade recommendations and evaluate them against ground truth."""

    forecast_rows = [_normalise_candle(c) for c in forecast or []]
    total_steps = len(forecast_rows)
    steps = total_steps
    if simulation_steps is not None and total_steps:
        try:
            requested = int(simulation_steps)
        except (TypeError, ValueError):
            requested = total_steps
        if requested > 0:
            steps = max(1, min(requested, total_steps))
        else:
            steps = total_steps
    simulation_rows = forecast_rows[:steps]
    truth_rows = [
        {
            **_normalise_timestamp(c),
            "open": _coerce_float(c.get("open")),
            "close": _coerce_float(c.get("close"), c.get("open", 0.0)),
        }
        for c in (ground_truth or [])
    ]
    if simulation_rows:
        truth_rows = truth_rows[: len(simulation_rows)]

    state = TradeBotState(cash=float(initial_cash))
    decisions: List[dict] = []
    orders: List[dict] = []

    for idx, candle in enumerate(simulation_rows):
        price = candle["open"] or candle["close"]
        if price <= 0:
            price = candle["close"] or 0.0
        probability = candle["probability"]
        expected_return = candle["expected_return"]
        action = "hold"
        reason = "No strong directional signal."

        if state.position <= 0 and expected_return > 0:
            if probability >= 0.52 and price > 0 and state.cash > 0:
                units = state.cash / price
                state.position = units
                state.cash = 0.0
                state.entry_price = price
                action = "buy"
                reason = "Positive expected return with bullish probability."
                orders.append(
                    {
                        "step": idx,
                        "action": action,
                        "price": price,
                        "probability": probability,
                        "expected_return": expected_return,
                        **_normalise_timestamp(candle),
                    }
                )
            else:
                reason = "Signal not strong enough to enter position."
        elif state.position > 0 and expected_return < 0:
            if probability <= 0.48:
                proceeds = state.position * price
                state.cash += proceeds
                state.position = 0.0
                state.entry_price = None
                action = "sell"
                reason = "Negative expected return suggests closing the position."
                orders.append(
                    {
                        "step": idx,
                        "action": action,
                        "price": price,
                        "probability": probability,
                        "expected_return": expected_return,
                        **_normalise_timestamp(candle),
                    }
                )
            else:
                reason = "Downside signal weak; hold existing position."
        elif state.position > 0:
            reason = "Holding existing position."
        else:
            reason = "Staying in cash."

        decisions.append(
            {
                "step": idx,
                "action": action,
                "price": price,
                "probability": probability,
                "expected_return": expected_return,
                "cash_after": state.cash,
                "position_after": state.position,
                "position_value": state.position * price,
                "reason": reason,
                **_normalise_timestamp(candle),
            }
        )

    last_price = simulation_rows[-1]["close"] if simulation_rows else 0.0
    final_equity = state.cash + state.position * last_price
    final_state = {
        "cash": state.cash,
        "position": state.position,
        "entry_price": state.entry_price,
        "mark_price": last_price,
        "total_value": final_equity,
    }

    evaluation = _evaluate_against_truth(orders, truth_rows, float(initial_cash))

    return {
        "initial_cash": float(initial_cash),
        "decisions": decisions,
        "orders": orders,
        "final_state": final_state,
        "evaluation": evaluation,
        "summary": _summarise_orders(orders),
        "simulation_steps": len(simulation_rows),
    }


def serialise_trade_report(report: Optional[dict]) -> Optional[dict]:
    if report is None:
        return None

    def serialise_number(value: Optional[Number]) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    try:
        sim_steps = int(serialise_number(report.get("simulation_steps")))
    except (TypeError, ValueError):
        sim_steps = 0

    serialised = {
        "initial_cash": serialise_number(report.get("initial_cash")),
        "summary": report.get("summary"),
        "simulation_steps": sim_steps,
        "final_state": {
            "cash": serialise_number(report.get("final_state", {}).get("cash")),
            "position": serialise_number(report.get("final_state", {}).get("position")),
            "entry_price": serialise_number(
                report.get("final_state", {}).get("entry_price")
            ),
            "mark_price": serialise_number(
                report.get("final_state", {}).get("mark_price")
            ),
            "total_value": serialise_number(
                report.get("final_state", {}).get("total_value")
            ),
        },
        "evaluation": None,
        "orders": [],
        "decisions": [],
    }

    evaluation = report.get("evaluation") or {}
    if evaluation:
        serialised["evaluation"] = {
            "available": bool(evaluation.get("available")),
            "realized_pl": serialise_number(evaluation.get("realized_pl")),
            "unrealized_pl": serialise_number(evaluation.get("unrealized_pl")),
            "total_value": serialise_number(evaluation.get("total_value")),
            "final_cash": serialise_number(evaluation.get("final_cash")),
            "final_position": serialise_number(evaluation.get("final_position")),
            "final_price": serialise_number(evaluation.get("final_price")),
        }
    else:
        serialised["evaluation"] = {"available": False}

    for collection_key in ("orders", "decisions"):
        entries = []
        for entry in report.get(collection_key, []) or []:
            entries.append(
                {
                    "step": int(entry.get("step", 0)),
                    "action": entry.get("action"),
                    "price": serialise_number(entry.get("price")),
                    "probability": serialise_number(entry.get("probability")),
                    "expected_return": serialise_number(entry.get("expected_return")),
                    "timestamp": int(
                        serialise_number(entry.get("timestamp") or entry.get("ts"))
                    ),
                    "datetime": entry.get("datetime"),
                    "cash_after": serialise_number(entry.get("cash_after")),
                    "position_after": serialise_number(entry.get("position_after")),
                    "position_value": serialise_number(entry.get("position_value")),
                    "reason": entry.get("reason"),
                }
            )
        serialised[collection_key] = entries

    return serialised
