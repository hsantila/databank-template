from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from fairmd.lipids.api import get_FF, get_OP, get_quality
from fairmd.lipids.analib.formfactor import calc_ff_scaling_distance
from fairmd.lipids.auxiliary.opconvertor import build_nice_OPdict
from fairmd.lipids.experiment import ExperimentCollection
from fairmd.lipids.molecules import Lipid
from fairmd.lipids.ipylib import plotFormFactor, plotOrderParameters


def _as_id_list(value) -> list[str]:
    """Normalize experiment metadata into a flat list of experiment IDs."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_as_id_list(item))
        return out
    if isinstance(value, Iterable):
        out = []
        for item in value:
            out.extend(_as_id_list(item))
        return out
    return [str(value)]


def _op_to_nested(op_dict: dict | None) -> dict:
    """Convert OP dictionary values to the nested shape expected by ipylib."""
    if not op_dict:
        return {}

    nested: dict[str, list] = {}
    for key, value in op_dict.items():
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
            nested[key] = value
        else:
            nested[key] = [value]
    return nested


def _registry_ready_op(op_dict: dict | None, lipid: str) -> dict:
    """Convert OP data into the registry-sorted structure used by FAIRMD plots."""
    if not op_dict:
        return {}

    try:
        lipid_obj = Lipid(lipid)
        return build_nice_OPdict(op_dict, lipid_obj)
    except Exception:
        return _op_to_nested(op_dict)


def _ff_to_curve(ff_data) -> list[list[float]] | None:
    """Convert FF data to rows [q, I, err?] usable by ipylib plotFormFactor."""
    if ff_data is None:
        return None

    if isinstance(ff_data, np.ndarray):
        if ff_data.ndim != 2 or ff_data.shape[1] < 2:
            return None
        return ff_data.tolist()

    if isinstance(ff_data, list):
        if len(ff_data) == 0:
            return None
        return ff_data

    if isinstance(ff_data, dict):
        q_vals = ff_data.get("q")
        i_vals = ff_data.get("I")
        if q_vals is None or i_vals is None:
            return None
        n = min(len(q_vals), len(i_vals))
        return [[float(q_vals[i]), float(i_vals[i])] for i in range(n)]

    return None


def _load_op_experiment_dict(
    lipid: str,
    exp_ids: list[str],
    op_collection: ExperimentCollection,
) -> dict:
    """Merge OP experiment dictionaries for a lipid using ExperimentCollection."""
    op_exp_flat = {}
    for exp_id in exp_ids:
        exp = op_collection.get(exp_id)
        if exp is None:
            continue
        lipid_data = exp.data.get(lipid)
        if not lipid_data:
            continue
        op_exp_flat.update(lipid_data)
    return _op_to_nested(op_exp_flat)


def _load_first_ff_experiment_curve(
    exp_ids: list[str],
    ff_collection: ExperimentCollection,
) -> tuple[str | None, list[list[float]] | None]:
    """Load the first available FF experiment curve from ExperimentCollection."""
    for exp_id in exp_ids:
        exp = ff_collection.get(exp_id)
        if exp is None:
            continue
        curve = _ff_to_curve(exp.data)
        if curve is not None:
            return exp_id, curve
    return None, None


def plotSimulation_fixed(system, lipid: str):  # noqa: N802
    """
    Robust drop-in replacement for fairmd.lipids.ipylib.plotSimulation.

    The intent matches the upstream helper:
    - print DOI and quality summary
    - plot simulated FF and experimental FF when available
    - plot simulated OP and experimental OP for the selected lipid when available

    This version tolerates modern EXPERIMENT schemas where ORDERPARAMETER entries
    are lists of experiment IDs instead of dictionaries.
    """
    print("DOI:", system.get("DOI", "N/A"))

    experiments = system.get("EXPERIMENT", {})
    op_ids = _as_id_list(experiments.get("ORDERPARAMETER", {}).get(lipid, []))
    ff_ids = _as_id_list(experiments.get("FORMFACTOR", []))

    ff_quality = np.nan
    try:
        ff_quality = get_quality(system, experiment="FF")
    except Exception:
        pass
    if not np.isnan(ff_quality):
        print("Form factor quality:", ff_quality)

    ff_scale = 1.0
    try:
        ff_quality_raw = get_FF(system)
        ff_sim = _ff_to_curve(ff_quality_raw)
        if ff_sim is None:
            raise ValueError("Invalid simulation FF data format")
    except Exception as exc:
        ff_sim = None
        print(f"Simulation FF not available: {exc}")

    try:
        op_sim_all = get_OP(system)
        op_sim = _op_to_nested(op_sim_all.get(lipid))
    except Exception as exc:
        op_sim = {}
        print(f"Simulation OP not available for {lipid}: {exc}")

    # Load experiments through FAIRMD experiment API to avoid direct filesystem assumptions.
    ExperimentCollection.clear_instance()
    op_collection = ExperimentCollection.load_from_data("OPExperiment")
    ExperimentCollection.clear_instance()
    ff_collection = ExperimentCollection.load_from_data("FFExperiment")

    op_exp = _load_op_experiment_dict(lipid, op_ids, op_collection)
    ff_exp_id, ff_exp = _load_first_ff_experiment_curve(ff_ids, ff_collection)

    try:
        if ff_sim is not None:
            plotFormFactor(ff_sim, 1, "Simulation", "red")
        if ff_exp is not None:
            try:
                ff_scale = float(
                    calc_ff_scaling_distance(np.array(ff_exp), np.array(ff_sim))[0]
                )
                if ff_exp_id is not None:
                    print(f"Using FF scale {ff_scale:.4g} from experiment {ff_exp_id}.")
            except Exception:
                ff_scale = 1.0
                print("Could not compute FF scale; using 1.0.")
            plotFormFactor(ff_exp, ff_scale, "Experiment", "black")
        else:
            print("No matched FF experiment curve found.")
        plt.show()
    except Exception as exc:
        plt.show()
        print(f"Form factor plotting failed: {exc}")

    op_sim_lipid = op_sim.get(lipid) if isinstance(op_sim, dict) and lipid in op_sim else op_sim
    op_sim = _registry_ready_op(op_sim_lipid, lipid)
    op_exp = _registry_ready_op(op_exp, lipid)

    if op_sim and op_exp:
        plotOrderParameters(op_sim, op_exp)
    elif not op_sim:
        print(f"Simulation OP data missing for {lipid}.")
    else:
        print(f"No matched OP experiment data found for {lipid}.")
