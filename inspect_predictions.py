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
OUTPUT_CSV           = "/content/drive/MyDrive/Amir_Dataset/predictions_inspection.csv"
CONFIDENCE_THRESHOLD = 0.1
NMS_IOU_THRESHOLD    = 0.2
IOU_MATCH_THRESHOLD  = 0.5
BATCH_SIZE           = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
ckpt  = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model = IntentNetViT(backbone_cfg=ckpt['backbone_cfg']).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("✅ Model loaded")

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading val dataset...")
dataset = ArgoverseIntentNetDataset(data_dir=VAL_DATA_DIR, is_train=False)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=0, collate_fn=collate_fn)
print(f"✅ {len(dataset)} sequences across all val logs")

# ── Generate anchors ──────────────────────────────────────────────────────────
anchors = generate_anchors(
    bev_height=GRID_HEIGHT_PX,
    bev_width=GRID_WIDTH_PX,
    feature_map_stride=8,
    anchor_configs=ANCHOR_CONFIGS_PAPER
).to(DEVICE)

# ── Run inference ─────────────────────────────────────────────────────────────
print("\nRunning inference...")
rows = []
sample_idx = 0

with torch.inference_mode():
    for batch in tqdm(loader, unit="batch"):
        if batch is None:
            continue

        lidar_bev = batch["lidar_bev"].to(DEVICE, non_blocking=True)
        map_bev   = batch["map_bev"].to(DEVICE, non_blocking=True)
        gt_list   = batch["gt_list"]

        det_cls_logits, det_box_preds_rel, intent_logits = model(lidar_bev, map_bev)

        for b_idx in range(lidar_bev.shape[0]):
            gt         = gt_list[b_idx]
            gt_boxes   = gt.get('boxes_xywha', torch.empty((0, 5)))
            gt_intents = gt.get('intentions', torch.empty(0, dtype=torch.long))
            log_id     = dataset.sequences[sample_idx][0] if hasattr(dataset, 'sequences') else f"sample_{sample_idx}"
            sample_idx += 1

            # ── Post-process predictions ──────────────────────────────────────
            scores = torch.sigmoid(det_cls_logits[b_idx])
            if scores.ndim > 1:
                scores = scores.squeeze(-1)

            keep = torch.where(scores >= CONFIDENCE_THRESHOLD)[0]
            if keep.numel() == 0:
                for gt_i in range(gt_boxes.shape[0]):
                    gt_intent_name = INTENTIONS_MAP_REV.get(gt_intents[gt_i].item(), "UNKNOWN")
                    rows.append({
                        'log_id'            : log_id,
                        'agent_idx'         : gt_i,
                        'predicted_intent'  : 'NOT DETECTED',
                        'actual_intent'     : gt_intent_name,
                        'correct'           : False,
                        'det_confidence'    : 0.0,
                        'intent_confidence' : 0.0,
                    })
                continue

            scores_f  = scores[keep]
            boxes_dec = decode_box_predictions(det_box_preds_rel[b_idx][keep], anchors[keep])
            nms_keep  = apply_nms(boxes_dec, scores_f, NMS_IOU_THRESHOLD)

            if nms_keep.numel() == 0:
                continue

            pred_scores   = scores_f[nms_keep].cpu()
            pred_boxes    = boxes_dec[nms_keep].cpu()

            intent_probs_all  = torch.softmax(intent_logits[b_idx][keep][nms_keep], dim=-1)
            pred_intents      = torch.argmax(intent_probs_all, dim=-1).cpu()
            intent_confidence = intent_probs_all.max(dim=-1).values.cpu()

            num_gt   = gt_boxes.shape[0]
            num_pred = pred_boxes.shape[0]

            if num_gt == 0:
                continue

            # ── Match predictions to GT by IoU ────────────────────────────────
            iou_matrix = compute_axis_aligned_iou(pred_boxes[:, :4].float(), gt_boxes[:, :4].float())
            gt_matched = torch.zeros(num_gt, dtype=torch.bool)
            sort_idx   = torch.argsort(pred_scores, descending=True)

            for i in range(num_pred):
                pi = sort_idx[i]
                if iou_matrix[pi].numel() == 0:
                    continue
                best_iou, best_gt = torch.max(iou_matrix[pi], dim=0)
                if best_iou >= IOU_MATCH_THRESHOLD and not gt_matched[best_gt]:
                    gt_matched[best_gt] = True
                    pred_intent_name    = INTENTIONS_MAP_REV.get(pred_intents[pi].item(), "UNKNOWN")
                    gt_intent_name      = INTENTIONS_MAP_REV.get(gt_intents[best_gt].item(), "UNKNOWN")
                    rows.append({
                        'log_id'            : log_id,
                        'agent_idx'         : best_gt.item(),
                        'predicted_intent'  : pred_intent_name,
                        'actual_intent'     : gt_intent_name,
                        'correct'           : pred_intent_name == gt_intent_name,
                        'det_confidence'    : pred_scores[pi].item(),
                        'intent_confidence' : intent_confidence[pi].item(),
                    })

            # Log unmatched GT as missed
            for gt_i in range(num_gt):
                if not gt_matched[gt_i]:
                    gt_intent_name = INTENTIONS_MAP_REV.get(gt_intents[gt_i].item(), "UNKNOWN")
                    rows.append({
                        'log_id'            : log_id,
                        'agent_idx'         : gt_i,
                        'predicted_intent'  : 'NOT DETECTED',
                        'actual_intent'     : gt_intent_name,
                        'correct'           : False,
                        'det_confidence'    : 0.0,
                        'intent_confidence' : 0.0,
                    })

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

# ── Helper: print summary for a subset ───────────────────────────────────────
def print_log_summary(subset, label):
    detected  = subset[subset['predicted_intent'] != 'NOT DETECTED']
    n_total   = len(subset)
    n_det     = len(detected)
    n_correct = detected['correct'].sum()

    avg_det_conf    = detected['det_confidence'].mean()    if n_det > 0 else 0.0
    avg_int_conf    = detected['intent_confidence'].mean() if n_det > 0 else 0.0
    intent_acc      = 100 * n_correct / max(n_det, 1)

    print(f"  Total agents (all sequences) : {n_total}")
    print(f"  Detected                     : {n_det} / {n_total} ({100*n_det/max(n_total,1):.1f}%)")
    print(f"  Correct intention            : {n_correct} / {n_det} ({intent_acc:.1f}%)")
    print(f"  Avg detection confidence     : {avg_det_conf:.4f}")
    print(f"  Avg intention confidence     : {avg_int_conf:.4f}")
    print(f"\n  Per-intention accuracy:")
    for intent in sorted(subset['actual_intent'].unique()):
        sub = detected[detected['actual_intent'] == intent]
        if len(sub) == 0:
            continue
        acc = sub['correct'].mean() * 100
        avg_ic = sub['intent_confidence'].mean()
        print(f"    {intent:<20}: {acc:.1f}%  ({sub['correct'].sum()}/{len(sub)})  "
              f"avg intent conf: {avg_ic:.4f}")

# ── Print per log ─────────────────────────────────────────────────────────────
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)

for log_id in df['log_id'].unique():
    print("\n" + "="*80)
    print(f"LOG: {log_id}")
    print("="*80)
    log_df = df[df['log_id'] == log_id].copy()
    print(log_df[['agent_idx', 'predicted_intent', 'actual_intent',
                  'correct', 'det_confidence', 'intent_confidence']].to_string(index=False))
    print()
    print_log_summary(log_df, log_id)

# ── Grand total summary ───────────────────────────────────────────────────────
print("\n" + "="*80)
print("GRAND TOTAL — ALL 5 VAL LOGS")
print("="*80)
print_log_summary(df, "ALL LOGS")

# ── Save CSV ──────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ CSV saved to: {OUTPUT_CSV}")
