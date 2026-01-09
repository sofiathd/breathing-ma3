import os
import argparse
from src.config_loader import load_config
from src.pipeline.batch import BreathingPipeline
from src.pipeline.specs import ExperimentSpec

def main():
    """Entry point to run the end-to-end breathing pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    frames_root = cfg["paths"]["frames_root_base"]
    results_dir = cfg["paths"]["results_dir"]
    cosmed = cfg["paths"]["cosmed"]

    cameras = cfg["experiment"]["cameras"]
    participants = cfg["experiment"]["participants"]
    takes_order = cfg["experiment"]["takes_order"]
    rois = cfg["experiment"]["rois_to_test"]
    pcas = cfg["experiment"]["pca_settings"]

    lo_hz = cfg["signal"]["lo_hz"]
    hi_hz = cfg["signal"]["hi_hz"]

    my_experiments = []
    for i in takes_order:
        for p in participants:
            for cam in cameras:
                my_experiments.append(
                    ExperimentSpec(
                        marker=f"take{i+1}",
                        camera=cam,
                        fps=cfg["experiment"]["fps"][p],
                        folder=f"{p}_{i+1}",
                        cosmed_path=cosmed[p],
                        subject=p,
                    )
                )

    pipeline = BreathingPipeline(base_results_dir=results_dir)

    pipeline.run_batch(
        frame_root_base=frames_root,
        experiment_specs=my_experiments,
        rois=rois,
        pcas=pcas,
    )

if __name__ == "__main__":
    main()