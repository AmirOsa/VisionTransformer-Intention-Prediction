import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, "/content/VisionTransformer-Intention-Prediction")

from constants import (GRID_HEIGHT_PX, GRID_WIDTH_PX, ANCHOR_CONFIGS_PAPER,
                       LIDAR_TOTAL_CHANNELS, MAP_CHANNELS, INTENTIONS_MAP_REV)
from dataset import ArgoverseIntentNetDataset, collate_fn
from model_vit import IntentNetViT
from utils import generate_anchors, decode_box_predictions, apply_nms, compute_axis_aligned_iou

# ── Config ────────────────────────────────────────────────────────────────────
VAL_DATA_DIR  = "/content/drive/MyDrive/Amir_Dataset/ViT-project/av2/sensor/val"
CHECKPOINT    = "/content/drive/MyDrive/Amir_Dataset/ViT-project_checkpoints/vit_model.pth"
HIVT_CSV      = "/content/drive/MyDrive/Amir_Dataset/HiVT-project_Confidence/hivt_focal_inspection.csv"
OUTPUT_CSV    = "/content/drive/MyDrive/Amir_Dataset/ViT-project_Confidence/nadeem_targeted_inspection.csv"

HIST_STEPS = 50
STEP_SIZE  = 25   # must match your conversion script
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Step 1: Load HiVT CSV ─────────────────────────────────────────────────────
print("Loading HiVT CSV...")
hivt_df = pd.read_csv(HIVT_CSV)
print(f"  {len(hivt_df)} HiVT scenarios across {hivt_df['log_id'].nunique()} logs")

# ── Step 2: Load model ────────────────────────────────────────────────────────
print("\nLoading Nadeem model...")
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model = IntentNetViT(backbone_cfg=ckpt['backbone_cfg']).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("✅ Model loaded")

# ── Step 3: Load dataset metadata ────────────────────────────────────────────
# We only need dataset.sequences for indexing — no DataLoader needed
print("\nLoading val dataset...")
dataset = ArgoverseIntentNetDataset(data_dir=VAL_DATA_DIR, is_train=False)
print(f"✅ {len(dataset)} BEV sequences across all val logs")

# ── Step 4: Generate anchors ──────────────────────────────────────────────────
anchors = generate_anchors(
    bev_height=GRID_HEIGHT_PX,
    bev_width=GRID_WIDTH_PX,
    feature_map_stride=8,
    anchor_configs=ANCHOR_CONFIGS_PAPER
).to(DEVICE)

# Precompute anchor centers once for direct intention lookup
anchors_cpu = anchors.cpu()
anchor_cx = anchors_cpu[:, 0] + anchors_cpu[:, 2] / 2   # [num_anchors]
anchor_cy = anchors_cpu[:, 1] + anchors_cpu[:, 3] / 2   # [num_anchors]

# ── Step 5: Build log_id → sorted sequence indices map ───────────────────────
# Fast — just reads metadata, no data loading
print("\nBuilding sequence index map...")
log_to_seq_indices = {}
for i, seq in enumerate(dataset.sequences):
    lid = seq['log_id']
    if lid not in log_to_seq_indices:
        log_to_seq_indices[lid] = []
    log_to_seq_indices[lid].append(i)

# ── Step 6: Compute exact global sequence index per HiVT scenario ─────────────
# scenario_id = {log_id}_w{w_idx:03d}
# target local index within log = w_idx * STEP_SIZE + HIST_STEPS - 1
# This is the BEV frame at timestep 49 — the last observed moment
print("Computing target sequence indices...")
targets = []

for _, row in hivt_df.iterrows():
    log_id = row['log_id']
    sid    = row['scenario_id']

    if log_id not in log_to_seq_indices:
        print(f"  ⚠️  Log {log_id} not in Nadeem val set — skipping {sid}")
        continue

    try:
        w_idx = int(sid.rsplit('_w', 1)[1])
    except (IndexError, ValueError):
        w_idx = 0

    target_local = w_idx * STEP_SIZE + HIST_STEPS - 1
    seq_indices  = log_to_seq_indices[log_id]

    if target_local >= len(seq_indices):
        print(f"  ⚠️  Target local index {target_local} out of range "
              f"({len(seq_indices)} seqs) for {sid} — skipping")
        continue

    global_idx = seq_indices[target_local]

    targets.append({
        'global_idx'     : global_idx,
        'scenario_id'    : sid,
        'log_id'         : log_id,
        'focal_track_id' : str(row['track_id']),
        'hivt_predicted' : row['predicted_intention'],
        'hivt_actual'    : row['actual_intention'],
        'hivt_correct'   : row['correct'],
        'hivt_conf'      : row['traj_confidence'],
    })

print(f"  {len(targets)} scenarios to process")

# ── Step 7: Run inference — exactly len(targets) forward passes ───────────────
# Directly index into dataset[global_idx] — no iteration over all 2551 sequences
print(f"\nRunning targeted Nadeem inference ({len(targets)} forward passes)...")
rows = []

with torch.inference_mode():
    for t in targets:
        global_idx     = t['global_idx']
        scenario_id    = t['scenario_id']
        log_id         = t['log_id']
        focal_track_id = t['focal_track_id']

        # ── Load only the one sequence we need ───────────────────────────────
        sample = dataset[global_idx]
        if sample is None:
            print(f"  ⚠️  dataset[{global_idx}] returned None for {scenario_id}")
            continue

        batch = collate_fn([sample])
        if batch is None:
            continue

        lidar_bev = batch["lidar_bev"].to(DEVICE)
        map_bev   = batch["map_bev"].to(DEVICE)
        gt        = batch["gt_list"][0]

        gt_boxes     = gt.get('boxes_xywha', torch.empty((0, 5)))
        gt_intents   = gt.get('intentions',  torch.empty(0, dtype=torch.long))
        gt_track_ids = gt.get('track_ids', [])

        # ── Forward pass ──────────────────────────────────────────────────────
        det_cls_logits, det_box_preds_rel, intent_logits = model(lidar_bev, map_bev)
        # intent_logits shape: [1, num_anchors, 8]

        # ── Find focal agent in GT by track_id ───────────────────────────────
        focal_gt_idx = None
        for gi, tid in enumerate(gt_track_ids):
            if str(tid) == focal_track_id:
                focal_gt_idx = gi
                break

        if focal_gt_idx is None:
            print(f"  ⚠️  Focal agent {focal_track_id} not in GT for {scenario_id}")
            rows.append({
                'scenario_id'       : scenario_id,
                'log_id'            : log_id,
                'track_id'          : focal_track_id,
                'predicted_intent'  : 'NOT IN GT',
                'actual_intent'     : 'UNKNOWN',
                'correct'           : False,
                'intent_confidence' : 0.0,
                'hivt_predicted'    : t['hivt_predicted'],
                'hivt_actual'       : t['hivt_actual'],
                'hivt_correct'      : t['hivt_correct'],
                'hivt_conf'         : t['hivt_conf'],
                'models_agree'      : False,
            })
            continue

        focal_gt_box         = gt_boxes[focal_gt_idx]
        focal_gt_intent      = gt_intents[focal_gt_idx].item()
        focal_gt_intent_name = INTENTIONS_MAP_REV.get(focal_gt_intent, 'UNKNOWN')

        # ── Direct anchor intention lookup ────────────────────────────────────
        # Find the anchor closest to the focal agent GT box center,
        # then read intention logits directly — no detection threshold needed.
        # This gives us a pure intention prediction regardless of objectness score.
        focal_cx_val = (focal_gt_box[0] + focal_gt_box[2] / 2).item()
        focal_cy_val = (focal_gt_box[1] + focal_gt_box[3] / 2).item()

        dists       = (anchor_cx - focal_cx_val)**2 + (anchor_cy - focal_cy_val)**2
        best_anchor = dists.argmin().item()

        intent_logits_focal = intent_logits[0][best_anchor]        # [8]
        intent_probs        = torch.softmax(intent_logits_focal, dim=-1)
        pred_intent_idx     = intent_probs.argmax().item()
        pred_intent_name    = INTENTIONS_MAP_REV.get(pred_intent_idx, 'UNKNOWN')
        intent_conf         = intent_probs.max().item()

        correct      = pred_intent_name == focal_gt_intent_name
        models_agree = pred_intent_name == t['hivt_predicted']

        print(f"  {scenario_id}: "
              f"Nadeem={pred_intent_name} "
              f"HiVT={t['hivt_predicted']} "
              f"GT={focal_gt_intent_name} "
              f"agree={models_agree} "
              f"conf={intent_conf:.3f}")

        rows.append({
            'scenario_id'       : scenario_id,
            'log_id'            : log_id,
            'track_id'          : focal_track_id,
            'predicted_intent'  : pred_intent_name,
            'actual_intent'     : focal_gt_intent_name,
            'correct'           : correct,
            'intent_confidence' : intent_conf,
            'hivt_predicted'    : t['hivt_predicted'],
            'hivt_actual'       : t['hivt_actual'],
            'hivt_correct'      : t['hivt_correct'],
            'hivt_conf'         : t['hivt_conf'],
            'models_agree'      : models_agree,
        })

# ── Step 8: Results ───────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

if df.empty:
    print("\n⚠️  No results — check log_id overlap between HiVT CSV and Nadeem val set")
else:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    print("\n" + "="*100)
    print("PER-SCENARIO RESULTS (one row per focal agent per window)")
    print("="*100)
    print(df[[
        'scenario_id', 'track_id',
        'predicted_intent', 'hivt_predicted', 'actual_intent',
        'models_agree', 'correct', 'hivt_correct',
        'intent_confidence', 'hivt_conf'
    ]].to_string(index=False))

    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    n_total          = len(df)
    n_nadeem_correct = df['correct'].sum()
    n_hivt_correct   = df['hivt_correct'].sum()
    n_agree          = df['models_agree'].sum()

    print(f"  Total scenarios processed        : {n_total}")
    print(f"  Nadeem correct intention         : {n_nadeem_correct} / {n_total} "
          f"({100*n_nadeem_correct/max(n_total,1):.1f}%)")
    print(f"  HiVT correct intention           : {n_hivt_correct} / {n_total} "
          f"({100*n_hivt_correct/max(n_total,1):.1f}%)")
    print(f"  Models AGREE on intention        : {n_agree} / {n_total} "
          f"({100*n_agree/max(n_total,1):.1f}%)")
    print(f"  Avg Nadeem intent confidence     : {df['intent_confidence'].mean():.4f}")
    print(f"  Avg HiVT traj confidence         : {df['hivt_conf'].mean():.4f}")

    print(f"\n  Per-intention breakdown (ground truth):")
    for intent in sorted(df['actual_intent'].unique()):
        sub   = df[df['actual_intent'] == intent]
        agree = sub['models_agree'].sum()
        n_cor = sub['correct'].sum()
        n_sub = len(sub)
        print(f"    {intent:<22}: "
              f"Nadeem correct {n_cor}/{n_sub}  "
              f"models agree {agree}/{n_sub} ({100*agree/max(n_sub,1):.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ CSV saved to: {OUTPUT_CSV}")
