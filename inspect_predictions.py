import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from constants import (GRID_HEIGHT_PX, GRID_WIDTH_PX, ANCHOR_CONFIGS_PAPER,
                       LIDAR_TOTAL_CHANNELS, MAP_CHANNELS, INTENTIONS_MAP_REV)
from dataset import ArgoverseIntentNetDataset, collate_fn
from model_vit import IntentNetViT
from utils import generate_anchors, decode_box_predictions, apply_nms, compute_axis_aligned_iou

# ── Config ────────────────────────────────────────────────────────────────────
VAL_DATA_DIR         = "/content/drive/MyDrive/Amir_Dataset/ViT-project/av2/sensor/val"
CHECKPOINT_PATH      = "/content/drive/MyDrive/Amir_Dataset/ViT-project_checkpoints/vit_model.pth"
HIVT_CSV_PATH        = "/content/drive/MyDrive/Amir_Dataset/HiVT-project_Confidence/hivt_focal_inspection.csv"
OUTPUT_CSV           = "/content/drive/MyDrive/Amir_Dataset/ViT-project_Confidence/nadeem_targeted_inspection.csv"

CONFIDENCE_THRESHOLD = 0.1
NMS_IOU_THRESHOLD    = 0.2
IOU_MATCH_THRESHOLD  = 0.5
BATCH_SIZE           = 1       # one sequence at a time so we can target exact frames
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Step 1: Load HiVT CSV to get target focal agents ─────────────────────────
# Each row: scenario_id, log_id, track_id (focal agent UUID), predicted_intention
print("Loading HiVT CSV...")
hivt_df = pd.read_csv(HIVT_CSV_PATH)

# Build lookup: log_id → list of (scenario_id, focal_track_id)
# A single log can produce multiple windows so there may be multiple entries per log
hivt_lookup = {}
for _, row in hivt_df.iterrows():
    log_id = row['log_id']
    if log_id not in hivt_lookup:
        hivt_lookup[log_id] = []
    hivt_lookup[log_id].append({
        'scenario_id':   row['scenario_id'],
        'focal_track_id': str(row['track_id']),
        'hivt_predicted': row['predicted_intention'],
        'hivt_actual':    row['actual_intention'],
        'hivt_correct':   row['correct'],
        'hivt_conf':      row['traj_confidence'],
    })

print(f"  {len(hivt_df)} HiVT scenarios across {len(hivt_lookup)} logs")

# ── Step 2: Load model ────────────────────────────────────────────────────────
print("\nLoading Nadeem model...")
ckpt  = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model = IntentNetViT(backbone_cfg=ckpt['backbone_cfg']).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("✅ Model loaded")

# ── Step 3: Load dataset ──────────────────────────────────────────────────────
# Nadeem's dataset returns one BEV sequence per entry.
# Each sequence corresponds to one timestamp window in the sensor log.
# dataset.sequences[i]['log_id'] tells us which log this BEV frame belongs to.
print("\nLoading val dataset...")
dataset = ArgoverseIntentNetDataset(data_dir=VAL_DATA_DIR, is_train=False)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=0, collate_fn=collate_fn)
print(f"✅ {len(dataset)} BEV sequences across all val logs")

# ── Step 4: Generate anchors ──────────────────────────────────────────────────
anchors = generate_anchors(
    bev_height=GRID_HEIGHT_PX,
    bev_width=GRID_WIDTH_PX,
    feature_map_stride=8,
    anchor_configs=ANCHOR_CONFIGS_PAPER
).to(DEVICE)

# ── Step 5: Build a map from log_id → sorted list of sequence indices
# This lets us find which BEV frame index is closest to timestep 49
# in a given sensor log.
# Nadeem's dataset is ordered chronologically within each log, so
# the BEV frame closest to the window boundary (t=49) is the one
# with sequence index = HIST_STEPS - 1 = 49 within that log's frames.
print("\nBuilding sequence index map...")
log_to_seq_indices = {}
for i, seq in enumerate(dataset.sequences):
    log_id = seq['log_id']
    if log_id not in log_to_seq_indices:
        log_to_seq_indices[log_id] = []
    log_to_seq_indices[log_id].append(i)

# ── Step 6: Run inference — focal agents only ─────────────────────────────────
# For each log that appears in the HiVT CSV, find the BEV frame closest
# to timestep 49 of each scenario window, run Nadeem on it, and look for
# the focal agent by matching track_id in the GT annotations.
print("\nRunning targeted Nadeem inference...")
rows = []

# We iterate through all BEV sequences but only record a result when
# the sequence belongs to a log that HiVT processed AND the BEV frame
# index within that log matches the target timestep for a scenario window.

# Pre-compute: for each scenario in HiVT CSV, what is the target BEV
# frame index within its log?
# scenario_id format: {log_id}_w{window_idx:03d}
# Each window starts at start_idx = window_idx * STEP_SIZE in the sensor log.
# The last observed timestep in the window is start_idx + HIST_STEPS - 1.
# In Nadeem's BEV dataset, sequences within a log are indexed 0..N-1
# chronologically, so the target BEV index = start_idx + HIST_STEPS - 1.
HIST_STEPS = 50
STEP_SIZE  = 25   # must match your conversion script

# Build target_bev_index per (log_id, scenario_id)
scenario_targets = {}   # (log_id, scenario_id) → target_bev_local_index
for log_id, entries in hivt_lookup.items():
    for entry in entries:
        sid = entry['scenario_id']
        # parse window index from scenario_id suffix _wNNN
        try:
            w_idx = int(sid.rsplit('_w', 1)[1])
        except (IndexError, ValueError):
            w_idx = 0
        start_idx = w_idx * STEP_SIZE
        target_local = start_idx + HIST_STEPS - 1   # 0-indexed within log
        scenario_targets[(log_id, sid)] = target_local

# Now run inference
with torch.inference_mode():
    for global_seq_idx, batch in enumerate(tqdm(loader, unit="seq")):
        if batch is None:
            continue

        seq_info = dataset.sequences[global_seq_idx]
        log_id   = seq_info['log_id']

        # Only process logs that appear in HiVT CSV
        if log_id not in hivt_lookup:
            continue

        # Local index of this BEV frame within its log
        local_indices = log_to_seq_indices[log_id]
        local_idx     = local_indices.index(global_seq_idx)

        # Check if this local_idx matches any scenario target for this log
        matched_entries = []
        for entry in hivt_lookup[log_id]:
            sid    = entry['scenario_id']
            target = scenario_targets.get((log_id, sid), -1)
            if local_idx == target:
                matched_entries.append(entry)

        if not matched_entries:
            continue   # this BEV frame is not needed

        # ── Run Nadeem on this BEV frame ──────────────────────────────────────
        lidar_bev = batch["lidar_bev"].to(DEVICE, non_blocking=True)
        map_bev   = batch["map_bev"].to(DEVICE, non_blocking=True)
        gt        = batch["gt_list"][0]

        gt_boxes   = gt.get('boxes_xywha', torch.empty((0, 5)))
        gt_intents = gt.get('intentions',  torch.empty(0, dtype=torch.long))
        gt_track_ids = gt.get('track_ids', [])

        det_cls_logits, det_box_preds_rel, intent_logits = model(lidar_bev, map_bev)

        scores = torch.sigmoid(det_cls_logits[0])
        if scores.ndim > 1:
            scores = scores.squeeze(-1)

        # For each matched scenario entry, look up the focal agent in GT
        for entry in matched_entries:
            focal_track_id  = entry['focal_track_id']
            scenario_id     = entry['scenario_id']

            # Find this focal agent in GT by track_id
            focal_gt_idx = None
            for gi, tid in enumerate(gt_track_ids):
                if str(tid) == focal_track_id:
                    focal_gt_idx = gi
                    break

            if focal_gt_idx is None:
                # Focal agent not visible in this BEV frame — record as not detected
                rows.append({
                    'scenario_id'       : scenario_id,
                    'log_id'            : log_id,
                    'track_id'          : focal_track_id,
                    'predicted_intent'  : 'NOT DETECTED',
                    'actual_intent'     : INTENTIONS_MAP_REV.get(
                                            gt_intents[focal_gt_idx].item(), 'UNKNOWN'
                                          ) if focal_gt_idx is not None else 'UNKNOWN',
                    'correct'           : False,
                    'det_confidence'    : 0.0,
                    'intent_confidence' : 0.0,
                    'hivt_predicted'    : entry['hivt_predicted'],
                    'hivt_actual'       : entry['hivt_actual'],
                    'hivt_correct'      : entry['hivt_correct'],
                    'hivt_conf'         : entry['hivt_conf'],
                    'models_agree'      : False,
                })
                continue

            focal_gt_box    = gt_boxes[focal_gt_idx]          # [5]
            focal_gt_intent = gt_intents[focal_gt_idx].item()
            focal_gt_intent_name = INTENTIONS_MAP_REV.get(focal_gt_intent, 'UNKNOWN')

            # ── Post-process predictions ──────────────────────────────────────
            keep = torch.where(scores >= CONFIDENCE_THRESHOLD)[0]

            if keep.numel() == 0:
                rows.append({
                    'scenario_id'       : scenario_id,
                    'log_id'            : log_id,
                    'track_id'          : focal_track_id,
                    'predicted_intent'  : 'NOT DETECTED',
                    'actual_intent'     : focal_gt_intent_name,
                    'correct'           : False,
                    'det_confidence'    : 0.0,
                    'intent_confidence' : 0.0,
                    'hivt_predicted'    : entry['hivt_predicted'],
                    'hivt_actual'       : entry['hivt_actual'],
                    'hivt_correct'      : entry['hivt_correct'],
                    'hivt_conf'         : entry['hivt_conf'],
                    'models_agree'      : False,
                })
                continue

            scores_f  = scores[keep]
            boxes_dec = decode_box_predictions(det_box_preds_rel[0][keep], anchors[keep])
            nms_keep  = apply_nms(boxes_dec, scores_f, NMS_IOU_THRESHOLD)

            if nms_keep.numel() == 0:
                rows.append({
                    'scenario_id'       : scenario_id,
                    'log_id'            : log_id,
                    'track_id'          : focal_track_id,
                    'predicted_intent'  : 'NOT DETECTED',
                    'actual_intent'     : focal_gt_intent_name,
                    'correct'           : False,
                    'det_confidence'    : 0.0,
                    'intent_confidence' : 0.0,
                    'hivt_predicted'    : entry['hivt_predicted'],
                    'hivt_actual'       : entry['hivt_actual'],
                    'hivt_correct'      : entry['hivt_correct'],
                    'hivt_conf'         : entry['hivt_conf'],
                    'models_agree'      : False,
                })
                continue

            pred_scores       = scores_f[nms_keep].cpu()
            pred_boxes        = boxes_dec[nms_keep].cpu()
            intent_probs_all  = torch.softmax(intent_logits[0][keep][nms_keep], dim=-1)
            pred_intents      = torch.argmax(intent_probs_all, dim=-1).cpu()
            intent_confidence = intent_probs_all.max(dim=-1).values.cpu()

            # ── Match focal GT box to best prediction by IoU ──────────────────
            focal_box_4 = focal_gt_box[:4].unsqueeze(0).float()   # [1, 4]
            iou_vec     = compute_axis_aligned_iou(pred_boxes[:, :4].float(), focal_box_4).squeeze(1)

            best_iou, best_pred_idx = iou_vec.max(dim=0)

            if best_iou >= IOU_MATCH_THRESHOLD:
                pi = best_pred_idx.item()
                pred_intent_name = INTENTIONS_MAP_REV.get(pred_intents[pi].item(), 'UNKNOWN')
                det_conf         = pred_scores[pi].item()
                int_conf         = intent_confidence[pi].item()
                correct          = pred_intent_name == focal_gt_intent_name
            else:
                pred_intent_name = 'NOT DETECTED'
                det_conf         = 0.0
                int_conf         = 0.0
                correct          = False

            # ── Agreement between Nadeem and HiVT ────────────────────────────
            # Both models predicted the same intention class?
            models_agree = (pred_intent_name == entry['hivt_predicted'] and
                            pred_intent_name != 'NOT DETECTED')

            rows.append({
                'scenario_id'       : scenario_id,
                'log_id'            : log_id,
                'track_id'          : focal_track_id,
                'predicted_intent'  : pred_intent_name,
                'actual_intent'     : focal_gt_intent_name,
                'correct'           : correct,
                'det_confidence'    : det_conf,
                'intent_confidence' : int_conf,
                'hivt_predicted'    : entry['hivt_predicted'],
                'hivt_actual'       : entry['hivt_actual'],
                'hivt_correct'      : entry['hivt_correct'],
                'hivt_conf'         : entry['hivt_conf'],
                'models_agree'      : models_agree,
            })

# ── Step 7: Build DataFrame ───────────────────────────────────────────────────
df = pd.DataFrame(rows)

# ── Step 8: Print results ─────────────────────────────────────────────────────
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

print("\n" + "="*90)
print("PER-SCENARIO RESULTS (one row per focal agent per window)")
print("="*90)
print(df[[
    'scenario_id', 'track_id',
    'predicted_intent', 'hivt_predicted', 'actual_intent',
    'models_agree', 'correct', 'hivt_correct',
    'det_confidence', 'intent_confidence', 'hivt_conf'
]].to_string(index=False))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("SUMMARY")
print("="*90)

detected = df[df['predicted_intent'] != 'NOT DETECTED']
n_total  = len(df)
n_det    = len(detected)
n_nadeem_correct = detected['correct'].sum()
n_hivt_correct   = df['hivt_correct'].sum()
n_agree          = df['models_agree'].sum()

print(f"  Total scenarios (focal agents)   : {n_total}")
print(f"  Nadeem detected focal agent      : {n_det} / {n_total} ({100*n_det/max(n_total,1):.1f}%)")
print(f"  Nadeem correct intention         : {n_nadeem_correct} / {n_det} ({100*n_nadeem_correct/max(n_det,1):.1f}%)")
print(f"  HiVT correct intention           : {n_hivt_correct} / {n_total} ({100*n_hivt_correct/max(n_total,1):.1f}%)")
print(f"  Models AGREE on intention        : {n_agree} / {n_total} ({100*n_agree/max(n_total,1):.1f}%)")
print(f"  Avg Nadeem det confidence        : {detected['det_confidence'].mean():.4f}")
print(f"  Avg Nadeem intent confidence     : {detected['intent_confidence'].mean():.4f}")
print(f"  Avg HiVT traj confidence         : {df['hivt_conf'].mean():.4f}")

print(f"\n  Per-intention agreement:")
for intent in sorted(df['actual_intent'].unique()):
    sub      = df[df['actual_intent'] == intent]
    agree    = sub['models_agree'].sum()
    n_sub    = len(sub)
    print(f"    {intent:<22}: agree {agree}/{n_sub} ({100*agree/max(n_sub,1):.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ CSV saved to: {OUTPUT_CSV}")
