import json
import os
import random
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config", "router_config.json")
CANDIDATE_DIR = os.path.join(ROOT, "evolve", "candidates")


def load_base() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def mutate(base: dict) -> dict:
    candidate = dict(base)
    delta = random.choice([-40, -20, 0, 20, 40])
    candidate["complex_length"] = max(60, int(base.get("complex_length", 180)) + delta)
    return candidate


def score(candidate: dict) -> dict:
    return {
        "success_at_1": 0.0,
        "tool_call_precision": 0.0,
        "tool_call_failure_rate": 0.0,
        "avg_steps": 0.0,
        "latency_p95": 0.0,
        "safety_violations": 0,
    }


def main() -> None:
    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    base = load_base()
    candidate = mutate(base)
    metrics = score(candidate)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "candidate_id": f"candidate_{stamp}",
        "source": "router_config.json",
        "candidate": candidate,
        "metrics": metrics,
        "status": "STAGED",
    }

    path = os.path.join(CANDIDATE_DIR, f"candidate_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
