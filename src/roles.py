"""Deterministic, label-free ESS feature and role utilities. No learner code."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["essround", "cntry", "idno", "raw_row_index"]
SOURCE_ROLES = ["fit", "validation", "source_calibration"]
TARGET_ROLES = ["local_reserve", "test"]


def canonical_features(raw, cohort, config):
    """Return the approved semantic inputs; never changes the reporting cohort."""
    result = cohort[KEYS].copy()
    for spec in config["features"]:
        name = spec["name"]
        if spec["source"] == "existing approved cohort":
            result[name] = cohort[name].astype(str)
        else:
            n = pd.to_numeric(raw[spec["source"]], errors="coerce").astype(float)
            if spec["type"] == "continuous":
                assert np.isfinite(n).all() and n.between(25, 89).all()
                result[name] = n.to_numpy()
            elif name == "hhmmb":
                valid = np.isfinite(n) & n.ge(1) & n.mod(1).eq(0)
                result[name] = ["unknown" if not ok else ("5+" if v >= 5 else str(int(v)))
                                for v, ok in zip(n, valid)]
            else:
                valid_codes = set(spec["valid_codes"])
                result[name] = [str(int(v)) if np.isfinite(v) and v in valid_codes else "unknown" for v in n]
    return result


def canonical_ids(frame):
    n = pd.to_numeric(frame.idno, errors="raise").to_numpy(dtype=np.float64)
    assert np.isfinite(n).all() and (n % 1 == 0).all() and (np.abs(n) < 2**53).all()
    return [str(int(v)) for v in n]


def hashes(frame, purpose, seed):
    ids = canonical_ids(frame)
    values = [f"{purpose}|{seed}|{int(r)}|{c}|{i}" for r, c, i in zip(frame.essround, frame.cntry, ids)]
    return [hashlib.sha256(v.encode("ascii")).hexdigest() for v in values]


def roles_for_wave(cohort, wave, config):
    result = cohort[KEYS].copy()
    ids = canonical_ids(cohort)
    for purpose in ["local_reserve", "score_ties"] + (["source_roles"] if wave == "R7" else []):
        result[purpose + "_hash"] = hashes(cohort, purpose, config["seeds"][purpose])
    result["target_rank"] = 0
    result["target_role"] = ""
    result["source_role"] = "not_source_wave"
    if wave == "R7":
        result["source_rank"] = 0
    for country, ix in cohort.groupby("cntry", sort=True).groups.items():
        order = sorted(ix, key=lambda i: (result.at[i, "local_reserve_hash"], ids[i]))
        for rank, i in enumerate(order, 1):
            result.at[i, "target_rank"] = rank
            result.at[i, "target_role"] = "local_reserve" if rank <= 100 else "test"
        if wave == "R7":
            order = sorted(ix, key=lambda i: (result.at[i, "source_roles_hash"], ids[i]))
            nf, nv = 70 * len(ix) // 100, 15 * len(ix) // 100
            for rank, i in enumerate(order, 1):
                result.at[i, "source_rank"] = rank
                result.at[i, "source_role"] = "fit" if rank <= nf else ("validation" if rank <= nf + nv else "source_calibration")
    a = pd.Series(cohort.anweight.to_numpy(dtype=np.float64), index=cohort.index)
    result["target_within_weight"] = a / a.groupby([cohort.cntry, result.target_role]).transform("sum")
    if wave == "R7":
        result["source_within_weight"] = a / a.groupby([cohort.cntry, result.source_role]).transform("sum")
    return result


def role_mask(roles, fold, role):
    if role in SOURCE_ROLES:
        return roles.cntry.ne(fold) & roles.source_role.eq(role)
    if role in TARGET_ROLES:
        return roles.cntry.eq(fold) & roles.target_role.eq(role)
    raise ValueError("Unknown role")


def transform_inputs(features, metadata, fold, view="tree"):
    """Apply source-only vocabularies and optional approved age scaling to a copy."""
    if view not in ["tree", "neural"]:
        raise ValueError("view must be tree or neural")
    x = features[metadata["feature_names"]].copy()
    record = metadata["folds"][fold]
    for name, vocab in record["categorical_vocabularies"].items():
        x[name] = x[name].where(x[name].isin(vocab["observed_in_source_fit"]), "unknown")
    if view == "neural":
        z = record["age_source_fit_moments"]
        x["agea"] = (x.agea - z["mean"]) / z["scale"]
    return x

