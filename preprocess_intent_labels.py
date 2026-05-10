# preprocess_intent_labels.py
#
# Adapted from Nadeem Mohamed's IntentNetViT
# Original repo: https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction
#
# Modifications:
#   1. Added transform_annotations_to_city_frame() function
#      Transforms annotation positions and headings from ego frame to city frame
#      before running the intention heuristic.
#
#   Why this fix is needed:
#      AV2 annotations.feather stores positions in ego frame per timestamp.
#      Each timestamp has a different ego frame because the ego vehicle moves.
#      When the heuristic computes heading change across timestamps, it compares
#      positions in different ego frames. If the ego turned left, all surrounding
#      vehicles appear to turn left even if they went straight in reality.
#      EVIDENCE: one scene showed 74/78 vehicles labelled TURN_LEFT — physically
#      impossible, caused by the ego vehicle turning left.
#      FIX: transform all positions to city frame (absolute coordinates) first,
#      so heading changes reflect real vehicle motion regardless of ego rotation.
#
#   What is unchanged:
#      The heuristic logic, thresholds, and all other processing are identical
#      to Nadeem's original. The sole change is the coordinate frame of the
#      input positions passed to the heuristic.
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
# NEW FUNCTION: transform_annotations_to_city_frame
# =============================================================================

def transform_annotations_to_city_frame(
    annotations_df: pd.DataFrame,
    ego_poses_df: pd.DataFrame
) -> pd.DataFrame:
    """
    NEW — Transforms annotation positions and headings from ego frame to city frame.

    AV2 annotations.feather stores positions in ego frame per timestamp.
    This means tx_m and ty_m are relative to the ego vehicle at each specific
    timestamp — a different coordinate frame for every row.

    When the heuristic compares positions across timestamps to compute heading
    change, it is comparing values from different ego frames. If the ego vehicle
    rotated between timestamps, this rotation contaminates the heading change
    computation — vehicles going straight appear to turn.

    This function transforms every annotation position to city frame (absolute
    coordinates) so all timestamps share the same reference frame. The heuristic
    then correctly measures how vehicles actually moved in the real world.

    Transformation formula:
        pos_city_x = ego_tx + cos(ego_yaw) * agent_ego_x - sin(ego_yaw) * agent_ego_y
        pos_city_y = ego_ty + sin(ego_yaw) * agent_ego_x + cos(ego_yaw) * agent_ego_y
        heading_city = agent_heading_ego + ego_yaw

    SOURCED: sensor_to_mf.py — same formula validated to 3 decimal places
    in cross-model validation (thesis notes Section 6.3).

    Args:
        annotations_df: raw annotations from annotations.feather (ego frame)
        ego_poses_df:   ego poses from city_SE3_egovehicle.feather (city frame)

    Returns:
        annotations_df with tx_m, ty_m, qx, qy, qz, qw updated to city frame.
        All other columns unchanged.
    """
    # Work on a copy — do not modify the original dataframe
    df = annotations_df.copy()

    # Build timestamp → ego pose lookup for efficiency
    # Avoids repeated dataframe filtering inside the loop
    ego_pose_lookup = ego_poses_df.set_index('timestamp_ns')

    missing_pose_count = 0

    # Process each unique timestamp group
    # All annotations at the same timestamp share the same ego pose
    for ts_ns, group_idx in df.groupby('timestamp_ns').groups.items():

        # Get ego pose at this timestamp
        if ts_ns not in ego_pose_lookup.index:
            missing_pose_count += len(group_idx)
            continue

        ego_row = ego_pose_lookup.loc[ts_ns]

        # Ego position in city frame (absolute coordinates)
        ego_tx = float(ego_row['tx_m'])
        ego_ty = float(ego_row['ty_m'])

        # Ego yaw (heading) in city frame
        # Extract rotation around z-axis from ego quaternion
        # SOURCED: same quaternion → yaw extraction as sensor_to_mf.py
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

        # Get all agent positions in ego frame for this timestamp group
        agent_ego_x = df.loc[group_idx, 'tx_m'].values.astype(float)
        agent_ego_y = df.loc[group_idx, 'ty_m'].values.astype(float)

        # Transform positions from ego frame to city frame
        # SOURCED: sensor_to_mf.py transformation formula
        city_x = ego_tx + cos_yaw * agent_ego_x - sin_yaw * agent_ego_y
        city_y = ego_ty + sin_yaw * agent_ego_x + cos_yaw * agent_ego_y

        # Update positions in dataframe
        df.loc[group_idx, 'tx_m'] = city_x
        df.loc[group_idx, 'ty_m'] = city_y

        # Transform heading (quaternion) from ego frame to city frame
        # For each agent: heading_city = heading_ego + ego_yaw
        # Then convert back to quaternion for compatibility with heuristic
        # SOURCED: sensor_to_mf.py heading transformation
        for idx in group_idx:
            try:
                agent_quat = np.array([
                    float(df.loc[idx, 'qx']),
                    float(df.loc[idx, 'qy']),
                    float(df.loc[idx, 'qz']),
                    float(df.loc[idx, 'qw'])
                ])
                agent_heading_ego = R.from_quat(agent_quat).as_euler('xyz')[2]

                # Add ego yaw to get city-frame heading
                heading_city = agent_heading_ego + ego_yaw

                # Convert back to quaternion
                # R.from_euler('z', angle) rotates around z-axis
                # .as_quat() returns [qx, qy, qz, qw]
                city_quat = R.from_euler('z', heading_city).as_quat()

                df.loc[idx, 'qx'] = city_quat[0]
                df.loc[idx, 'qy'] = city_quat[1]
                df.loc[idx, 'qz'] = city_quat[2]
                df.loc[idx, 'qw'] = city_quat[3]

            except (ValueError, KeyError):
                # Leave this row's quaternion unchanged if conversion fails
                continue

    if missing_pose_count > 0:
        print(f"  Warning: {missing_pose_count} annotations had no matching ego pose "
              f"— left in ego frame.")

    return df


# =============================================================================
# preprocess_scenario — modified to add city-frame transformation
# =============================================================================

def preprocess_scenario(scenario_info: namedtuple, force_recompute: bool = False):
    """
    Processes a single scenario: loads annotations, calculates intentions, and saves.

    MODIFICATION vs Nadeem's original:
    Before running the heuristic, annotations are transformed from ego frame
    to city frame using transform_annotations_to_city_frame().
    This eliminates the ego rotation contamination in heading change computation.
    The heuristic logic and thresholds are completely unchanged from Nadeem's original.

    Returns "processed", "skipped", or "failed".
    """
    log_dir = Path(scenario_info.log_dir)
    log_id = log_dir.name
    annotations_path = Path(scenario_info.annotations_path)
    map_json_path = Path(scenario_info.map_path)
    output_path = log_dir / OUTPUT_ANNOTATION_FILENAME

    # NEW: path to ego pose file — always present in valid scenarios
    ego_pose_path = log_dir / "city_SE3_egovehicle.feather"

    if not force_recompute and output_path.exists():
        return "skipped"

    try:
        # Load raw annotations — unchanged from Nadeem's original
        annotations_df = pd.read_feather(annotations_path)

        # NEW: load ego poses for coordinate transformation
        if not ego_pose_path.is_file():
            print(f"  ERROR: ego pose file missing for {log_id}: {ego_pose_path}")
            return "failed"

        ego_poses_df = pd.read_feather(ego_pose_path)

        # NEW: transform annotation positions from ego frame to city frame
        # This is the fix for the ego rotation contamination problem.
        # After this transformation, tx_m and ty_m are absolute city coordinates
        # consistent across all timestamps — the heuristic can correctly compute
        # heading changes that reflect real vehicle motion in the world.
        annotations_df = transform_annotations_to_city_frame(
            annotations_df, ego_poses_df
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

        # Calculate intention labels
        # Heuristic is completely unchanged from Nadeem's original
        # It now receives city-frame positions instead of ego-frame positions
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

        annotations_df['heuristic_intent'] = annotations_df.progress_apply(
            lambda row: calculate_intent_for_row(row, annotations_df, static_map),
            axis=1
        )

        annotations_df.to_feather(output_path)
        return "processed"

    except Exception as e:
        print(f"  ERROR processing scenario {log_id}: {e}")
        traceback.print_exc()
        return "failed"


# =============================================================================
# main — unchanged from Nadeem's original except the note about the fix
# =============================================================================

def main(data_root_dir: str, splits: list[str] = None, force_recompute: bool = False):
    """
    Main function to iterate over dataset splits and preprocess intention labels.
    Unchanged from Nadeem's original except preprocess_scenario now applies
    the city-frame coordinate transformation before running the heuristic.
    """
    if splits is None:
        splits = ["train", "val"]

    print(f"Starting intention label pre-computation.")
    print(f"NOTE: Annotations transformed to city frame before heuristic.")
    print(f"      Fixes ego rotation contamination in heading change computation.")
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
