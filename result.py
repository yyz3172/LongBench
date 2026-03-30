import os
import json
import math


def _read_pred_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def _pct(numer, denom):
    if denom == 0:
        return None
    return 100.0 * numer / denom


def _percentile(sorted_vals, p):
    """
    p: 0-100
    使用最近秩（nearest-rank）定义：ceil(p/100*N) 的值
    """
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    k = int(math.ceil(p / 100.0 * n)) - 1
    k = max(0, min(n - 1, k))
    return float(sorted_vals[k])


def _summarize_latency(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    vals.sort()
    if not vals:
        return {
            "n": 0,
            "avg": None,
            "min": None,
            "max": None,
            "median": None,
            "p90": None,
            "p99": None,
        }
    n = len(vals)
    avg = sum(vals) / n
    return {
        "n": n,
        "avg": avg,
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "median": _percentile(vals, 50),
        "p90": _percentile(vals, 90),
        "p99": _percentile(vals, 99),
    }


def main():
    results_dir = "results"
    stats_dir = "results_stats"
    os.makedirs(stats_dir, exist_ok=True)

    files = sorted(os.listdir(results_dir))
    table = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"]
    compensated = False

    for file in files:
        filename = os.path.join(results_dir, file)
        if not os.path.isfile(filename):
            continue

        pred_data = _read_pred_file(filename)
        if not pred_data:
            continue

        # accuracy slices
        easy = hard = short = medium = long = 0
        easy_acc = hard_acc = short_acc = medium_acc = long_acc = 0

        # latency
        ttft_ms = []
        e2e_ms = []
        tpot_ms = []
        ttft_ms_cot = []
        e2e_ms_cot = []

        pred_none = 0

        for pred in pred_data:
            acc = int(bool(pred.get("judge")))
            if pred.get("pred") is None:
                pred_none += 1
                if compensated:
                    acc = 0.25

            if pred.get("difficulty") == "easy":
                easy += 1
                easy_acc += acc
            else:
                hard += 1
                hard_acc += acc

            if pred.get("length") == "short":
                short += 1
                short_acc += acc
            elif pred.get("length") == "medium":
                medium += 1
                medium_acc += acc
            else:
                long += 1
                long_acc += acc

            if "ttft_ms" in pred:
                ttft_ms.append(pred.get("ttft_ms"))
            if "e2e_ms" in pred:
                e2e_ms.append(pred.get("e2e_ms"))
            if "tpot_ms" in pred:
                tpot_ms.append(pred.get("tpot_ms"))
            if "ttft_ms_cot" in pred:
                ttft_ms_cot.append(pred.get("ttft_ms_cot"))
            if "e2e_ms_cot" in pred:
                e2e_ms_cot.append(pred.get("e2e_ms_cot"))

        name = ".".join(file.split(".")[:-1]) or file

        overall = _pct(easy_acc + hard_acc, len(pred_data))
        row = [
            name,
            f"{overall:.1f}" if overall is not None else "--",
            f"{_pct(easy_acc, easy):.1f}" if easy else "--",
            f"{_pct(hard_acc, hard):.1f}" if hard else "--",
            f"{_pct(short_acc, short):.1f}" if short else "--",
            f"{_pct(medium_acc, medium):.1f}" if medium else "--",
            f"{_pct(long_acc, long):.1f}" if long else "--",
        ]
        table.append("\t".join(row))

        # per-model stats
        stats = {
            "model": name,
            "file": filename,
            "n": len(pred_data),
            "pred_none": pred_none,
            "accuracy": {
                "overall": overall,
                "easy": _pct(easy_acc, easy),
                "hard": _pct(hard_acc, hard),
                "short": _pct(short_acc, short),
                "medium": _pct(medium_acc, medium),
                "long": _pct(long_acc, long),
            },
            "latency_ms": {
                "ttft": _summarize_latency(ttft_ms),
                "e2e": _summarize_latency(e2e_ms),
                "tpot": _summarize_latency(tpot_ms),
                "ttft_cot": _summarize_latency(ttft_ms_cot),
                "e2e_cot": _summarize_latency(e2e_ms_cot),
            },
        }

        out_json = os.path.join(stats_dir, f"{name}.stats.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # also write a readable txt
        out_txt = os.path.join(stats_dir, f"{name}.stats.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n")
            f.write(f"File: {filename}\n")
            f.write(f"N: {len(pred_data)}\n")
            f.write(f"pred=None: {pred_none}\n")
            f.write("\n[Accuracy %]\n")
            for k in ("overall", "easy", "hard", "short", "medium", "long"):
                v = stats["accuracy"][k]
                f.write(f"{k}: {v:.2f}\n" if isinstance(v, (int, float)) else f"{k}: --\n")
            f.write("\n[Latency ms]\n")
            for key in ("ttft", "e2e", "tpot", "ttft_cot", "e2e_cot"):
                s = stats["latency_ms"][key]
                f.write(f"\n{key} (n={s['n']}):\n")
                for kk in ("avg", "median", "p90", "p99", "min", "max"):
                    vv = s[kk]
                    f.write(f"  {kk}: {vv:.2f}\n" if isinstance(vv, (int, float)) else f"  {kk}: --\n")

    with open("result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(table))


if __name__ == "__main__":
    main()
