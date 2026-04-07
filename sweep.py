#!/usr/bin/env python3
"""Automated parameter sweep for strategy experiments.

Runs N experiments by modifying strategy.py parameters, running backtest,
and recording results. Restores original strategy.py after each experiment.
"""

import subprocess
import re
import os
import sys
import time
import json
import itertools
import copy

REPO = os.path.dirname(os.path.abspath(__file__))
STRATEGY_PATH = os.path.join(REPO, "strategy.py")
RESULTS_PATH = os.path.join(REPO, "results.tsv")
RUN_LOG = os.path.join(REPO, "run.log")

# Read original strategy.py
with open(STRATEGY_PATH, "r") as f:
    ORIGINAL_STRATEGY = f.read()

# Current best
best_score = -999.0
best_desc = "baseline"
exp_count = 0


def set_param(content: str, name: str, value) -> str:
    """Replace a parameter value in strategy.py content."""
    if isinstance(value, float):
        val_str = f"{value}"
    elif isinstance(value, int):
        val_str = f"{value}"
    else:
        val_str = str(value)
    # Match patterns like: NAME = 123 or NAME = 0.088
    pattern = rf'^({name}\s*=\s*)([^\n#]+)'
    replacement = rf'\g<1>{val_str}'
    return re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)


def set_code_line(content: str, old_line: str, new_line: str) -> str:
    """Replace a specific code line."""
    return content.replace(old_line, new_line)


def run_backtest() -> dict:
    """Run backtest and return parsed results."""
    env = os.environ.copy()
    env["DAILY_MODE"] = "1"
    try:
        result = subprocess.run(
            ["uv", "run", "backtest.py"],
            capture_output=True, text=True, timeout=300,
            cwd=REPO, env=env
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"score": -999, "error": "timeout"}

    with open(RUN_LOG, "w") as f:
        f.write(output)

    metrics = {}
    for line in output.split("\n"):
        for key in ["score", "sharpe", "total_return_pct", "max_drawdown_pct",
                     "num_trades", "win_rate_pct", "profit_factor"]:
            if line.strip().startswith(f"{key}:"):
                try:
                    metrics[key] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    if "score" not in metrics:
        metrics["score"] = -999
    return metrics


def run_experiment(description: str, content: str) -> dict:
    """Write strategy, run backtest, record results, restore original."""
    global exp_count, best_score, best_desc
    exp_count += 1

    # Write modified strategy
    with open(STRATEGY_PATH, "w") as f:
        f.write(content)

    # Run backtest
    t0 = time.time()
    metrics = run_backtest()
    elapsed = time.time() - t0

    score = metrics.get("score", -999)
    sharpe = metrics.get("sharpe", 0)
    max_dd = metrics.get("max_drawdown_pct", 0)
    num_trades = metrics.get("num_trades", 0)
    total_ret = metrics.get("total_return_pct", 0)
    win_rate = metrics.get("win_rate_pct", 0)
    pf = metrics.get("profit_factor", 0)

    status = "PASS" if score > best_score else "FAIL"
    if score > best_score:
        best_score = score
        best_desc = description

    # Record
    with open(RESULTS_PATH, "a") as f:
        f.write(f"exp{exp_count}\t{score:.3f}\t{sharpe:.3f}\t{max_dd:.3f}\t{status}\t{description}\n")

    marker = "***" if status == "PASS" else "   "
    print(f"{marker} [{exp_count:3d}] score={score:8.3f} sharpe={sharpe:8.3f} dd={max_dd:5.2f}% trades={num_trades:5.0f} ret={total_ret:6.1f}% wr={win_rate:4.1f}% pf={pf:5.2f} [{elapsed:4.0f}s] {description}", flush=True)

    # Restore original
    with open(STRATEGY_PATH, "w") as f:
        f.write(ORIGINAL_STRATEGY)

    return metrics


def main():
    global best_score
    experiments = []

    # ── PHASE 1: Single-parameter sweeps ──────────────────────────

    # ATR_STOP_MULT sweep
    for v in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 6.5, 7.0, 8.0, 10.0]:
        c = set_param(ORIGINAL_STRATEGY, "ATR_STOP_MULT", v)
        experiments.append((f"ATR_STOP_MULT={v}", c))

    # BASE_POSITION_PCT sweep
    for v in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        c = set_param(ORIGINAL_STRATEGY, "BASE_POSITION_PCT", v)
        experiments.append((f"BASE_POSITION_PCT={v}", c))

    # BASE_THRESHOLD sweep
    for v in [0.003, 0.005, 0.008, 0.010, 0.015, 0.018, 0.020, 0.025, 0.030]:
        c = set_param(ORIGINAL_STRATEGY, "BASE_THRESHOLD", v)
        experiments.append((f"BASE_THRESHOLD={v}", c))

    # MIN_VOTES sweep
    for v in [2, 3, 5, 6]:
        c = set_param(ORIGINAL_STRATEGY, "MIN_VOTES", v)
        experiments.append((f"MIN_VOTES={v}", c))

    # COOLDOWN_BARS sweep
    for v in [0, 1, 2, 4, 5, 6, 8, 10]:
        c = set_param(ORIGINAL_STRATEGY, "COOLDOWN_BARS", v)
        experiments.append((f"COOLDOWN_BARS={v}", c))

    # RSI_PERIOD sweep
    for v in [4, 5, 6, 7, 10, 12, 14, 20]:
        c = set_param(ORIGINAL_STRATEGY, "RSI_PERIOD", v)
        experiments.append((f"RSI_PERIOD={v}", c))

    # EMA_FAST sweep
    for v in [3, 5, 9, 12, 15]:
        c = set_param(ORIGINAL_STRATEGY, "EMA_FAST", v)
        experiments.append((f"EMA_FAST={v}", c))

    # EMA_SLOW sweep
    for v in [15, 20, 30, 35, 40, 50]:
        c = set_param(ORIGINAL_STRATEGY, "EMA_SLOW", v)
        experiments.append((f"EMA_SLOW={v}", c))

    # SHORT_WINDOW sweep
    for v in [3, 4, 8, 10, 12]:
        c = set_param(ORIGINAL_STRATEGY, "SHORT_WINDOW", v)
        experiments.append((f"SHORT_WINDOW={v}", c))

    # MED_WINDOW sweep
    for v in [6, 8, 10, 15, 18, 20, 24]:
        c = set_param(ORIGINAL_STRATEGY, "MED_WINDOW", v)
        experiments.append((f"MED_WINDOW={v}", c))

    # MACD_FAST sweep
    for v in [8, 10, 12, 16, 18]:
        c = set_param(ORIGINAL_STRATEGY, "MACD_FAST", v)
        experiments.append((f"MACD_FAST={v}", c))

    # MACD_SLOW sweep
    for v in [18, 20, 26, 30, 35]:
        c = set_param(ORIGINAL_STRATEGY, "MACD_SLOW", v)
        experiments.append((f"MACD_SLOW={v}", c))

    # MACD_SIGNAL sweep
    for v in [5, 7, 11, 13]:
        c = set_param(ORIGINAL_STRATEGY, "MACD_SIGNAL", v)
        experiments.append((f"MACD_SIGNAL={v}", c))

    # BB_PERIOD sweep
    for v in [3, 7, 10, 14, 20]:
        c = set_param(ORIGINAL_STRATEGY, "BB_PERIOD", v)
        experiments.append((f"BB_PERIOD={v}", c))

    # VOL_LOOKBACK sweep
    for v in [12, 18, 24, 48, 72, 100]:
        c = set_param(ORIGINAL_STRATEGY, "VOL_LOOKBACK", v)
        experiments.append((f"VOL_LOOKBACK={v}", c))

    # RSI_BULL / RSI_BEAR sweep
    for bull, bear in [(40, 60), (42, 58), (45, 55), (50, 50), (55, 45)]:
        c = set_param(ORIGINAL_STRATEGY, "RSI_BULL", bull)
        c = set_param(c, "RSI_BEAR", bear)
        experiments.append((f"RSI_BULL={bull}_BEAR={bear}", c))

    # RSI_OVERBOUGHT / RSI_OVERSOLD sweep
    for ob, os_ in [(65, 35), (70, 30), (80, 20), (85, 15), (90, 10)]:
        c = set_param(ORIGINAL_STRATEGY, "RSI_OVERBOUGHT", ob)
        c = set_param(c, "RSI_OVERSOLD", os_)
        experiments.append((f"RSI_OB={ob}_OS={os_}", c))

    # TARGET_VOL sweep
    for v in [0.008, 0.010, 0.012, 0.018, 0.020, 0.025]:
        c = set_param(ORIGINAL_STRATEGY, "TARGET_VOL", v)
        experiments.append((f"TARGET_VOL={v}", c))

    # ATR_LOOKBACK sweep
    for v in [8, 12, 16, 20, 30, 36, 48]:
        c = set_param(ORIGINAL_STRATEGY, "ATR_LOOKBACK", v)
        experiments.append((f"ATR_LOOKBACK={v}", c))

    # TAKE_PROFIT_PCT sweep (effective vs disabled)
    for v in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0]:
        c = set_param(ORIGINAL_STRATEGY, "TAKE_PROFIT_PCT", v)
        experiments.append((f"TAKE_PROFIT_PCT={v}", c))

    # BB compression threshold sweep
    for v in [30, 50, 70, 80, 95, 99]:
        c = set_code_line(ORIGINAL_STRATEGY,
                          "bb_compressed = bb_pctile < 90",
                          f"bb_compressed = bb_pctile < {v}")
        experiments.append((f"bb_threshold={v}", c))

    # MED2_WINDOW sweep
    for v in [16, 20, 28, 32, 48]:
        c = set_param(ORIGINAL_STRATEGY, "MED2_WINDOW", v)
        experiments.append((f"MED2_WINDOW={v}", c))

    # LONG_WINDOW sweep
    for v in [24, 30, 42, 48, 60, 72]:
        c = set_param(ORIGINAL_STRATEGY, "LONG_WINDOW", v)
        experiments.append((f"LONG_WINDOW={v}", c))

    # vshort threshold multiplier variations
    for mult in [0.3, 0.5, 0.9, 1.0, 1.2]:
        c = set_code_line(ORIGINAL_STRATEGY,
                          "vshort_bull = ret_vshort > dyn_threshold * 0.7",
                          f"vshort_bull = ret_vshort > dyn_threshold * {mult}")
        c = set_code_line(c,
                          "vshort_bear = ret_vshort < -dyn_threshold * 0.7",
                          f"vshort_bear = ret_vshort < -dyn_threshold * {mult}")
        experiments.append((f"vshort_mult={mult}", c))

    # Dynamic threshold vol_ratio scaling
    for lo, hi in [(0.1, 0.9), (0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.0, 1.0)]:
        c = set_code_line(ORIGINAL_STRATEGY,
                          "dyn_threshold = BASE_THRESHOLD * (0.3 + vol_ratio * 0.7)",
                          f"dyn_threshold = BASE_THRESHOLD * ({lo} + vol_ratio * {hi})")
        experiments.append((f"dyn_thresh_scale={lo}_{hi}", c))

    # dyn_threshold clamp range
    for lo, hi in [(0.002, 0.015), (0.003, 0.025), (0.008, 0.030), (0.010, 0.040)]:
        c = set_code_line(ORIGINAL_STRATEGY,
                          "dyn_threshold = max(0.005, min(0.020, dyn_threshold))",
                          f"dyn_threshold = max({lo}, min({hi}, dyn_threshold))")
        experiments.append((f"dyn_clamp={lo}_{hi}", c))

    print(f"Phase 1: {len(experiments)} single-param experiments queued", flush=True)

    # ── PHASE 2: Multi-parameter combos ──────────────────────────
    # Will be populated after Phase 1 based on top performers
    phase2_base = len(experiments)

    # Pre-generate some promising combos
    for atr in [4.0, 5.0, 5.5, 6.0]:
        for pos in [0.06, 0.088, 0.10, 0.12]:
            for thresh in [0.008, 0.012, 0.015]:
                c = set_param(ORIGINAL_STRATEGY, "ATR_STOP_MULT", atr)
                c = set_param(c, "BASE_POSITION_PCT", pos)
                c = set_param(c, "BASE_THRESHOLD", thresh)
                experiments.append((f"combo_ATR={atr}_POS={pos}_THR={thresh}", c))

    # MIN_VOTES x COOLDOWN combos
    for mv in [3, 4, 5]:
        for cd in [0, 2, 3, 5]:
            c = set_param(ORIGINAL_STRATEGY, "MIN_VOTES", mv)
            c = set_param(c, "COOLDOWN_BARS", cd)
            experiments.append((f"combo_VOTES={mv}_CD={cd}", c))

    # EMA combos
    for fast in [5, 7, 9]:
        for slow in [20, 26, 35]:
            if fast >= slow:
                continue
            c = set_param(ORIGINAL_STRATEGY, "EMA_FAST", fast)
            c = set_param(c, "EMA_SLOW", slow)
            experiments.append((f"combo_EMA={fast}/{slow}", c))

    # RSI combo sweeps
    for period in [6, 8, 12]:
        for bull, bear in [(45, 55), (48, 52), (50, 50)]:
            for ob, os_ in [(70, 30), (75, 25), (80, 20)]:
                c = set_param(ORIGINAL_STRATEGY, "RSI_PERIOD", period)
                c = set_param(c, "RSI_BULL", bull)
                c = set_param(c, "RSI_BEAR", bear)
                c = set_param(c, "RSI_OVERBOUGHT", ob)
                c = set_param(c, "RSI_OVERSOLD", os_)
                experiments.append((f"combo_RSI_p={period}_b={bull}/{bear}_ob={ob}/{os_}", c))

    # MACD combos
    for fast in [10, 12, 14]:
        for slow in [20, 23, 26]:
            for sig in [7, 9, 11]:
                if fast >= slow:
                    continue
                c = set_param(ORIGINAL_STRATEGY, "MACD_FAST", fast)
                c = set_param(c, "MACD_SLOW", slow)
                c = set_param(c, "MACD_SIGNAL", sig)
                experiments.append((f"combo_MACD={fast}/{slow}/{sig}", c))

    # Window combos
    for short in [4, 6, 8]:
        for med in [10, 12, 16]:
            if short >= med:
                continue
            c = set_param(ORIGINAL_STRATEGY, "SHORT_WINDOW", short)
            c = set_param(c, "MED_WINDOW", med)
            experiments.append((f"combo_WIN={short}/{med}", c))

    print(f"Phase 2: {len(experiments) - phase2_base} combo experiments queued", flush=True)
    print(f"Total: {len(experiments)} experiments", flush=True)

    # Trim to 200 if we have more
    if len(experiments) > 200:
        experiments = experiments[:200]
        print(f"Trimmed to 200 experiments", flush=True)

    # Run the baseline first
    print(f"\n{'='*100}")
    print(f"BASELINE: score=13.299 (current strategy.py)")
    print(f"{'='*100}\n")
    best_score = 13.299  # Set baseline

    # Run all experiments
    t_total = time.time()
    for desc, content in experiments:
        run_experiment(desc, content)

    elapsed_total = time.time() - t_total
    print(f"\n{'='*100}")
    print(f"SWEEP COMPLETE: {exp_count} experiments in {elapsed_total/60:.1f} minutes")
    print(f"Best score: {best_score:.3f} ({best_desc})")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
