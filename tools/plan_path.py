def _difficulty_fit(difficulty:int, level:int|None)->float:
    if level is None: return 0.7
    return max(0.0, 1.0 - abs((difficulty or 2) - level)/4)

def _freshness(version:str|None, user_ver:str|None)->float:
    if not version or not user_ver: return 0.7
    return 1.0 if version==user_ver else 0.6

def score_candidate(c, user_goal:str, user:dict):
    R = c.get("score", 0.6)
    C = 0.8 if any(w in c["snippet"].lower() for w in user_goal.lower().split()[:5]) else 0.5
    F = _freshness(c.get("version"), user.get("product_version"))
    D = _difficulty_fit(c.get("difficulty",2), user.get("level"))
    P = 0.7
    return 0.35*R + 0.20*C + 0.20*F + 0.15*D + 0.10*P

def plan_path(candidates:list[dict], user_goal:str, time_budget:int=30, user:dict|None=None):
    user = user or {}
    ranked = sorted(candidates, key=lambda c: score_candidate(c, user_goal, user), reverse=True)
    if not ranked:
        return {"primary":None,"alternatives":[],"prerequisites":[],"post_tasks":[]}
    for c in ranked: c["estimated_time_min"] = c.get("duration_min", 15)
    primary = next((c for c in ranked if c["estimated_time_min"]<=time_budget), ranked[0])
    alts = [c for c in ranked if c is not primary][:2]
    prereqs = []
    return {
      "primary": {"id": primary["material_id"], "why": "Mejor cobertura vs. objetivo y tiempo", "estimated_time_min": primary["estimated_time_min"]},
      "alternatives": [{"id": a["material_id"], "why": "Alternativa útil"} for a in alts],
      "prerequisites": [{"id": p.get("material_id",""), "why": "Concepto base"} for p in prereqs],
      "post_tasks": []
    }
