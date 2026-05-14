# preprocess_intent_labels.py
#
# Adapted from Nadeem Mohamed's IntentNetViT
# Original repo: https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction
#
# Modifications:
#   1. Added transform_annotations_to_city_frame() function
#   2. FIXED: heuristic is computed on city-frame COPY of annotations
#      but ORIGINAL ego-frame positions are preserved in the saved file.
#
#   Why this fix is needed:
#      AV2 annotations.feather stores positions in ego frame per timestamp.
#      Each timestamp has a different ego frame because the ego vehicle moves.
#      When the heuristic computes heading change across timestamps, it compares
#      positions in different ego frames — contaminating the intention labels.
#      EVIDENCE: one scene showed 74/78 vehicles labelled TURN_LEFT — physically
#      impossible, caused by the ego vehicle turning left.
#
#   What the fix does:
#      Transforms a COPY of annotations to city frame before running the heuristic.
#      The original ego-frame positions are preserved in the saved file so that
#      detection GT boxes remain correct (anchors are in ego frame).
#      Only the heuristic_intent column changes.
#
#   SOURCED: transformation formula from sensor_to_mf.py — validated to 3 decimal
#   places in cross-model validation (thesis notes Section 6.3)

import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import traceback
from collections import namedtuple

from constants import (AV2_MAP_AVAILABLE, SHAPELY_AVAILABLE, VEHICLE_CATEGORIES,
                       INTENTIONS_MAP)
from dataset import ScenarioValidator
from heuristic_labeling import get_vehicle_intention_heuristic_enhanced

# --- Configuration ---
OUTPUT_ANNOTATION_FILENAME = "annotations_with_intent.feather"


# =============================================================================
# transform_annotations_to_city_frame
# =============================================================================

def transform_annotations_to_city_frame(
    annotations_df: pd.DataFrame,
    ego_poses_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforms annotation positions and headings from ego frame to city frame.

    Used ONLY for heuristic computation — the returned dataframe is a temporary
    copy. Original ego-frame positions are preserved in the saved output file.

    Transformation formula:
        pos_city_x = ego_tx + cos(ego_yaw) * agent_ego_x - sin(ego_yaw) * agent_ego_y
        pos_city_y = ego_ty + sin(ego_yaw) * agent_ego_x + cos(ego_yaw) * agent_ego_y
        heading_city = agent_heading_ego + ego_yaw

    SOURCED: sensor_to_mf.py — validated to 3 decimal places (thesis Section 6.3)

    Args:
        annotations_df: copy of annotations (ego frame)
        ego_poses_df:   ego poses from city_SE3_egovehicle.feather

    Returns:
        annotations_df with tx_m, ty_m, qx, qy, qz, qw updated to city frame.
    """
    df = annotations_df.copy()
    ego_pose_lookup = ego_poses_df.set_index('timestamp_ns')
    missing_pose_count = 0

    for ts_ns, group_idx in df.groupby('timestamp_ns').groups.items():
        if ts_ns not in ego_pose_lookup.index:
            missing_pose_count += len(group_idx)
            continue

        ego_row = ego_pose_lookup.loc[ts_ns]
        ego_tx = float(ego_row['tx_m'])
        ego_ty = float(ego_row['ty_m'])

        try:
            ego_quat = np.array([
                float(ego_row['qx']),
                float(ego_row['qy']),
                float(ego_row['qz']),
                float(ego_row['qw'])
            ])
            ego_yaw = R.from_quat(ego_quat).as_euler('xyz')[2]
        except (ValueError, KeyError):
            missing_pose_count += len(group_idx)
            continue

        cos_yaw = np.cos(ego_yaw)
        sin_yaw = np.sin(ego_yaw)

        agent_ego_x = df.loc[group_idx, 'tx_m'].values.astype(float)
        agent_ego_y = df.loc[group_idx, 'ty_m'].values.astype(float)

        city_x = ego_tx + cos_yaw * agent_ego_x - sin_yaw * agent_ego_y
        city_y = ego_ty + sin_yaw * agent_ego_x + cos_yaw * agent_ego_y

        df.loc[group_idx, 'tx_m'] = city_x
        df.loc[group_idx, 'ty_m'] = city_y

        for idx in group_idx:
            try:
                agent_quat = np.array([
                    float(df.loc[idx, 'qx']),
                    float(df.loc[idx, 'qy']),
                    float(df.loc[idx, 'qz']),
                    float(df.loc[idx, 'qw'])
                ])
                agent_heading_ego = R.from_quat(agent_quat).as_euler('xyz')[2]
                heading_city = agent_heading_ego + ego_yaw
                city_quat = R.from_euler('z', heading_city).as_quat()
                df.loc[idx, 'qx'] = city_quat[0]
                df.loc[idx, 'qy'] = city_quat[1]
                df.loc[idx, 'qz'] = city_quat[2]
                df.loc[idx, 'qw'] = city_quat[3]
            except (ValueError, KeyError):
                continue

    if missing_pose_count > 0:
        print(f"  Warning: {missing_pose_count} annotations had no matching ego pose.")

    return df


# =============================================================================
# preprocess_scenario
# =============================================================================

def preprocess_scenario(scenario_info: namedtuple, force_recompute: bool = False):
    """
    Processes a single scenario: loads annotations, calculates intentions, and saves.

    KEY DESIGN:
    - Heuristic is computed on a city-frame COPY of annotations (correct headings)
    - Original ego-frame positions are preserved in the saved output file
    - Only heuristic_intent column is added to the original dataframe
    - Detection GT boxes remain in ego frame → anchors match correctly

    Returns "processed", "skipped", or "failed".
    """
    log_dir = Path(scenario_info.log_dir)
    log_id = log_dir.name
    annotations_path = Path(scenario_info.annotations_path)
    map_json_path = Path(scenario_info.map_path)
    output_path = log_dir / OUTPUT_ANNOTATION_FILENAME
    ego_pose_path = log_dir / "city_SE3_egovehicle.feather"

    if not force_recompute and output_path.exists():
        return "skipped"

    try:
        # Load raw annotations — ego frame positions
        annotations_df = pd.read_feather(annotations_path)

        # Load ego poses for city-frame transformation
        if not ego_pose_path.is_file():
            print(f"  ERROR: ego pose file missing for {log_id}")
            return "failed"
        ego_poses_df = pd.read_feather(ego_pose_path)

        # Transform a COPY to city frame for heuristic computation
        # Original annotations_df stays in ego frame
        annotations_city = transform_annotations_to_city_frame(
            annotations_df.copy(), ego_poses_df
        )

        # Load map — unchanged from Nadeem's original
        static_map = None
        if AV2_MAP_AVAILABLE:
            map_dir = map_json_path.parent
            if map_dir.is_dir() and any(map_dir.glob("log_map_archive_*.json")):
                from av2.map.map_api import ArgoverseStaticMap
                static_map = ArgoverseStaticMap.from_map_dir(
                    map_dir, build_raster=False
                )
            else:
                print(f"  Warning: Map data missing for {log_id}.")

        # Run heuristic on city-frame copy
        # Heuristic logic unchanged from Nadeem's original
        def calculate_intent_for_row(row, full_log_df, current_static_map):
            if row['category'] in VEHICLE_CATEGORIES:
                return get_vehicle_intention_heuristic_enhanced(
                    track_id=row['track_uuid'],
                    current_ts_ns=row['timestamp_ns'],
                    all_log_gt_boxes_df=full_log_df,
                    static_map=current_static_map
                )
            return -1

        if not hasattr(pd.Series, 'progress_apply'):
            tqdm.pandas(desc=f"Intent Calc {log_id[:8]}", leave=False)

        annotations_city['heuristic_intent'] = annotations_city.progress_apply(
            lambda row: calculate_intent_for_row(row, annotations_city, static_map),
            axis=1
        )

        # Copy ONLY intent labels back to original ego-frame dataframe
        # Positions stay in ego frame — correct for detection GT
        annotations_df['heuristic_intent'] = annotations_city['heuristic_intent']

        # Save: ego-frame positions + correct city-frame-computed intent labels
        annotations_df.to_feather(output_path)
        return "processed"

    except Exception as e:
        print(f"  ERROR processing scenario {log_id}: {e}")
        traceback.print_exc()
        return "failed"


# =============================================================================
# main — unchanged from Nadeem's original
# =============================================================================

def main(data_root_dir: str, splits: list[str] = None, force_recompute: bool = False):
    if splits is None:
        splits = ["train", "val"]

    print(f"Starting intention label pre-computation.")
    print(f"NOTE: Heuristic computed in city frame. Ego-frame positions preserved.")
    print(f"Output file: {OUTPUT_ANNOTATION_FILENAME}")
    print(f"Force recompute: {force_recompute}")

    overall_start_time = time.time()
    total_processed = 0
    total_skipped = 0
    total_failed = 0

    for split_name in splits:
        print(f"\nProcessing split: {split_name}")
        split_dir = Path(data_root_dir) / split_name
        if not split_dir.is_dir():
            print(f"  Directory not found: {split_dir}. Skipping.")
            continue

        validator = ScenarioValidator(str(split_dir), skip_known_corrupted=False)
        valid_scenarios = validator.find_valid_scenarios()

        if not valid_scenarios:
            print(f"  No valid scenarios found in {split_dir}.")
            continue

        print(f"  Found {len(valid_scenarios)} scenarios in {split_name} split.")

        split_processed = 0
        split_skipped = 0
        split_failed = 0

        for scenario_info in tqdm(
            valid_scenarios,
            desc=f"Processing {split_name}",
            unit="scenario"
        ):
            result = preprocess_scenario(scenario_info, force_recompute)
            if result == "processed":
                split_processed += 1
            elif result == "skipped":
                split_skipped += 1
            else:
                split_failed += 1

        print(f"  Finished {split_name}: "
              f"Processed={split_processed}, "
              f"Skipped={split_skipped}, "
              f"Failed={split_failed}")
        total_processed += split_processed
        total_skipped += split_skipped
        total_failed += split_failed

    overall_end_time = time.time()
    print(f"\nPre-computation finished.")
    print(f"  Total time: {(overall_end_time - overall_start_time) / 60:.2f} minutes")
    print(f"  Processed:  {total_processed}")
    print(f"  Skipped:    {total_skipped}")
    print(f"  Failed:     {total_failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute vehicle intention labels for AV2 dataset."
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory of the AV2 sensor dataset."
    )
    parser.add_argument(
        "--splits", nargs='+', default=["train", "val"],
        help="Dataset splits to process. Default: train val."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-computation even if output files already exist."
    )
    args = parser.parse_args()

    main(
        data_root_dir=args.data_root,
        splits=args.splits,
        force_recompute=args.force
    )
