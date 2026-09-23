#!/usr/bin/env python3
"""The deployed Tooth-AI inference pipeline, lifted verbatim out of app.py.

app.py is a Streamlit application: importing it executes UI code at module
scope. This module holds the pure functions instead, copied from app.py with
NO logic changes, in the order main() calls them on the "Complete Scan" path.
Each block is preceded by its exact source range in app.py.

THE DEPLOYED CONFIGURATION (verified round 1 + round 2)
-------------------------------------------------------
  teeth      output/resnet50_9class_20k/model_final.pth   NUM_CLASSES 9
             INPUT.MIN_SIZE_TEST 800 / MAX_SIZE_TEST 1333
  anomalies  output/rtx4060_48k/model_final.pth           NUM_CLASSES 41
             INPUT.MIN_SIZE_TEST 512 / MAX_SIZE_TEST 1024

  The two models resize differently. There is NO 1024x512 resize anywhere in
  the deployed path; app.py never resizes, it relies on DefaultPredictor's
  ResizeShortestEdge, which is driven by the values above.

  MODEL.ROI_HEADS.SCORE_THRESH_TEST is overridden to 0.05 at app.py:375,
  then to `threshold` (default 0.05) at app.py:405.

  Two passes only (original + CLAHE). apply_gamma and flip_instances_horizontal
  exist in app.py at :454 and :475 and are called nowhere.

THE ONE DELIBERATE DEVIATION
----------------------------
process_predictions carries the class-routing bug at app.py:953-958 verbatim,
but routes it through _is_tooth() so it can be switched. Nothing else differs.
See _is_tooth for the original six lines quoted exactly.
"""
import cv2
import numpy as np

# cv2 and numpy are module-level here because app.py imports them at module
# scope too (app.py:14-15) and the copied bodies reference them as globals.
# torch and detectron2.structures stay lazy -- app.py imports those inside the
# functions that use them, so the copies do the same.



# =========================================================================
# Constants -- app.py
# =========================================================================

# --- app.py:29-39 (verbatim) ----------------------------------------
CLASSES = [
    "Tooth",                      # ID 0
    "Caries",                     # ID 1
    "Crown",                      # ID 2
    "Filling",                    # ID 3
    "Implant",                    # ID 4
    "Prefabricated metal post",   # ID 5
    "Retained root",              # ID 6
    "Root canal filling",         # ID 7
    "Root canal obturation"       # ID 8
]

# --- app.py:41-41 (verbatim) ----------------------------------------
ANOMALY_CLASSES = set(CLASSES[1:])  # All except "Tooth"

# --- app.py:58-58 (verbatim) ----------------------------------------
FDI_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]

# --- app.py:59-59 (verbatim) ----------------------------------------
FDI_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

# --- app.py:305-315 (verbatim) --------------------------------------
PATHOLOGY_CLASS_MAP = {
    **{i: "Tooth" for i in range(33)},  # IDs 0-32 map to "Tooth"
    33: "Caries",
    34: "Crown",
    35: "Filling",
    36: "Implant",
    37: "Prefabricated metal post",
    38: "Retained root",
    39: "Root canal filling",
    40: "Root canal obturation",
}


# =========================================================================
# Step 0  helpers
# =========================================================================

# --- app.py:652-667 (verbatim) --------------------------------------
def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


# =========================================================================
# Step 1  two-pass inference + fusion
# =========================================================================

# --- app.py:384-423 (verbatim) --------------------------------------
def run_inference(predictor, image_bgr, threshold=0.05):
    """
    FOCUSED 2-PASS INFERENCE PIPELINE
    
    Uses two complementary passes for robust detection:
    1. Original image - catches clear teeth and anomalies
    2. CLAHE enhanced - recovers teeth hidden in spinal shadow zone
    
    Results are fused using Weighted Box Fusion (WBF) with tight merging.
    
    Note: Previous 4-pass approach (+ gamma + H-flip) caused severe
    over-detection (59 teeth instead of ~28-32) due to noise amplification.
    
    Args:
        predictor: Detectron2 predictor
        image_bgr: BGR image (numpy array)
        threshold: Base detection threshold
        
    Returns:
        outputs: Detectron2 Instances with fused detections
    """
    predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    
    # === PASS 1: Original Image ===
    outputs_original = predictor(image_bgr)
    
    # === PASS 2: CLAHE Enhanced (recovers spinal shadow zone) ===
    clahe_image = apply_clahe(image_bgr, clip_limit=2.5, tile_size=(8, 8))
    outputs_clahe = predictor(clahe_image)
    
    # === FUSION via Weighted Box Fusion ===
    # Higher IoU threshold (0.55) ensures overlapping boxes from both passes
    # are properly merged rather than kept as separate detections
    fused_outputs = weighted_box_fusion(
        [outputs_original, outputs_clahe],
        iou_threshold=0.55,
        skip_box_threshold=threshold
    )
    
    return fused_outputs

# --- app.py:426-451 (verbatim) --------------------------------------
def apply_clahe(image_bgr, clip_limit=2.5, tile_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    local contrast in X-ray images, particularly in the spinal shadow zone.
    
    Args:
        image_bgr: BGR image
        clip_limit: Threshold for contrast limiting (2.0-3.0 optimal for X-ray)
        tile_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced BGR image
    """
    # Convert to LAB color space (L = lightness)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l_channel)
    
    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return result

# --- app.py:499-649 (verbatim) --------------------------------------
def weighted_box_fusion(outputs_list, iou_threshold=0.55, skip_box_threshold=0.05):
    """
    Consensus-Weighted Box Fusion (WBF) for 2-pass ensemble.
    
    Tracks which pass each detection came from. Consensus logic:
      - Detection in BOTH passes (2/2): Full confidence → max(scores)
      - Detection in ONLY ONE pass (1/2): Penalized → score × 0.6
    
    Single-pass detections are likely noise from one augmentation and
    should not have the same weight as consensus detections.
    
    Args:
        outputs_list: List of Detectron2 outputs from each pass
        iou_threshold: IoU threshold for matching boxes across passes
        skip_box_threshold: Minimum score to consider a box
        
    Returns:
        Fused Detectron2 outputs with consensus-weighted scores
    """
    import torch
    from detectron2.structures import Instances, Boxes
    
    SINGLE_PASS_PENALTY = 0.6  # 40% penalty for single-pass detections
    num_passes = len(outputs_list)
    
    # Collect all detections WITH pass origin tracking
    all_boxes = []
    all_scores = []
    all_classes = []
    all_masks = []
    all_pass_ids = []  # Track which pass each detection came from
    
    for pass_id, outputs in enumerate(outputs_list):
        instances = outputs["instances"].to("cpu")
        if len(instances) == 0:
            continue
            
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if score >= skip_box_threshold:
                all_boxes.append(box)
                all_scores.append(score)
                all_classes.append(cls)
                all_pass_ids.append(pass_id)
                if masks is not None:
                    all_masks.append(masks[i])
    
    if not all_boxes:
        # Return empty instances
        empty = Instances(outputs_list[0]["instances"].image_size)
        empty.pred_boxes = Boxes(torch.empty(0, 4))
        empty.scores = torch.empty(0)
        empty.pred_classes = torch.empty(0, dtype=torch.int64)
        return {"instances": empty}
    
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_classes = np.array(all_classes)
    all_pass_ids = np.array(all_pass_ids)
    
    # Group by class and apply consensus-weighted WBF within each class
    fused_boxes = []
    fused_scores = []
    fused_classes = []
    fused_masks = []
    
    unique_classes = np.unique(all_classes)
    
    for cls in unique_classes:
        cls_mask = all_classes == cls
        cls_boxes = all_boxes[cls_mask]
        cls_scores = all_scores[cls_mask]
        cls_pass_ids = all_pass_ids[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        
        # Sort by score
        sorted_idx = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[sorted_idx]
        cls_scores = cls_scores[sorted_idx]
        cls_pass_ids = cls_pass_ids[sorted_idx]
        cls_indices = cls_indices[sorted_idx]
        
        used = [False] * len(cls_boxes)
        
        for i in range(len(cls_boxes)):
            if used[i]:
                continue
            
            # Find all overlapping boxes from any pass
            cluster_boxes = [cls_boxes[i]]
            cluster_scores = [cls_scores[i]]
            cluster_pass_ids = {cls_pass_ids[i]}
            cluster_mask_indices = [cls_indices[i]]
            used[i] = True
            
            for j in range(i + 1, len(cls_boxes)):
                if used[j]:
                    continue
                    
                iou = compute_iou(cls_boxes[i], cls_boxes[j])
                if iou > iou_threshold:
                    cluster_boxes.append(cls_boxes[j])
                    cluster_scores.append(cls_scores[j])
                    cluster_pass_ids.add(cls_pass_ids[j])
                    cluster_mask_indices.append(cls_indices[j])
                    used[j] = True
            
            # Weighted average of cluster boxes
            cluster_boxes = np.array(cluster_boxes)
            cluster_scores = np.array(cluster_scores)
            weights = cluster_scores / cluster_scores.sum()
            
            fused_box = np.average(cluster_boxes, axis=0, weights=weights)
            
            # CONSENSUS SCORING:
            # - Both passes agree → full max(score)
            # - Single pass only → penalize by 40% (likely noise)
            num_contributing_passes = len(cluster_pass_ids)
            base_score = float(np.max(cluster_scores))
            
            if num_contributing_passes >= min(2, num_passes):
                # Consensus: detection confirmed by both passes
                fused_score = base_score
            else:
                # Single-pass only: apply penalty
                fused_score = base_score * SINGLE_PASS_PENALTY
            
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_classes.append(cls)
            
            # Use mask from highest scoring box
            if all_masks:
                best_idx = cluster_mask_indices[0]
                fused_masks.append(all_masks[best_idx])
    
    # Create fused Instances
    image_size = outputs_list[0]["instances"].image_size
    fused_instances = Instances(image_size)
    fused_instances.pred_boxes = Boxes(torch.tensor(np.array(fused_boxes), dtype=torch.float32))
    fused_instances.scores = torch.tensor(np.array(fused_scores), dtype=torch.float32)
    fused_instances.pred_classes = torch.tensor(np.array(fused_classes), dtype=torch.int64)
    
    if fused_masks:
        fused_instances.pred_masks = torch.tensor(np.array(fused_masks))
    
    return {"instances": fused_instances}


# =========================================================================
# Step 2  thresholding
# =========================================================================

# --- app.py:896-983 (verbatim) --------------------------------------
def process_predictions(outputs, image_width, teeth_threshold=0.15, anomaly_threshold=0.25, class_map=None, fix_cls_routing=False):
    """
    Process raw model outputs into teeth and anomaly lists.
    Uses DYNAMIC ZONE-BASED THRESHOLDING to compensate for center zone weakness.
    
    Args:
        outputs: Raw Detectron2 outputs
        image_width: Width of the input image for zone calculation
        teeth_threshold: Base threshold (used for anomalies)
        anomaly_threshold: Minimum confidence for anomalies
        class_map: Optional dict mapping class_id -> class_name
        
    Returns:
        (teeth, anomalies) - lists of detection dicts
    """
    instances = outputs["instances"].to("cpu")
    
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []
    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
    
    teeth = []
    anomalies = []
    
    # Define "Center Zone" (where spine shadow causes weak detection)
    center_start = image_width * 0.35
    center_end = image_width * 0.65
    
    for i, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
        # Determine class name using specific map or default global CLASSES
        if class_map and cls_id in class_map:
            class_name = class_map[cls_id]
        elif class_map is None and cls_id < len(CLASSES):
            class_name = CLASSES[cls_id]
        else:
            class_name = f"Unknown_{cls_id}"
        
        # Calculate box center X
        x_center = (box[0] + box[2]) / 2
        is_center_zone = (x_center > center_start) and (x_center < center_end)
        
        detection = {
            'box': box.tolist(),
            'class_id': int(cls_id),
            'class_name': class_name,
            'score': float(score),
            'mask': masks[i] if len(masks) > i else None,
            'is_center_zone': is_center_zone  # Track for debugging
        }
        
        # DYNAMIC THRESHOLDING FOR TEETH
        # Center zone (35-65%): Slightly relaxed threshold (model scores lower here)
        # Outer zones: Use slider value as-is (back teeth are clear)
        # The user's slider value (teeth_threshold) is always respected as minimum floor
        
        # Handle both 9-class (class_name == "Tooth") and 41-class (cls_id 0-32) models
        # DEVIATION: original six lines are quoted verbatim in _is_tooth().
        is_tooth = _is_tooth(class_name, cls_id, fix_cls_routing)
        
        if is_tooth:
            # Center zone gets a small relaxation (70% of slider), but never below 0.08
            # This prevents ghost detections while still recovering spinal-shadow teeth
            if is_center_zone:
                threshold = max(teeth_threshold * 0.7, 0.08)
            else:
                threshold = teeth_threshold
            if score >= threshold:
                teeth.append(detection)
        elif class_name in ANOMALY_CLASSES or (class_map and class_name != "Tooth"):
            # Structural anomalies (Crown, Implant, Metal Post) are visually distinctive -
            # use a lower threshold so they aren't filtered out
            STRUCTURAL_ANOMALIES = {"Crown", "Implant", "Prefabricated metal post"}
            if class_name in STRUCTURAL_ANOMALIES:
                effective_threshold = max(anomaly_threshold * 0.6, 0.10)
            else:
                effective_threshold = anomaly_threshold
            
            if score >= effective_threshold:
                anomalies.append(detection)
    
    # Deduplicate overlapping anomalies (keep highest scoring)
    anomalies = deduplicate_anomalies(anomalies, iou_threshold=0.6)
    
    return teeth, anomalies

# --- app.py:986-1025 (verbatim) -------------------------------------
def deduplicate_anomalies(anomalies, iou_threshold=0.6):
    """
    Remove duplicate anomaly detections by keeping only the highest scoring
    box when multiple anomalies overlap significantly.
    
    Solves: "Filling (86%)" overlapping with "Root Canal (97%)" on same tooth
    
    Args:
        anomalies: List of anomaly detection dicts
        iou_threshold: IoU above which boxes are considered duplicates
        
    Returns:
        Deduplicated list of anomalies
    """
    if len(anomalies) <= 1:
        return anomalies
    
    # Sort by score (highest first)
    anomalies = sorted(anomalies, key=lambda x: x['score'], reverse=True)
    
    keep = []
    used = [False] * len(anomalies)
    
    for i in range(len(anomalies)):
        if used[i]:
            continue
        
        keep.append(anomalies[i])
        used[i] = True
        
        # Mark overlapping boxes as used (they will be discarded)
        for j in range(i + 1, len(anomalies)):
            if used[j]:
                continue
            
            iou = compute_iou(anomalies[i]['box'], anomalies[j]['box'])
            if iou > iou_threshold:
                used[j] = True  # Discard lower-scoring overlapping box
    
    return keep


# =========================================================================
# Step 3-5  geometric engine (NMS, jaw split, FDI, anomaly mapping)
# =========================================================================

# --- app.py:66-296 (verbatim) ---------------------------------------
class DentalGeometricEngine:
    """Brain of the system: Converts raw detections to dental chart."""
    
    @staticmethod
    def get_box_center(box):
        """Get center point of a bounding box."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @staticmethod
    def get_box_width(box):
        """Get width of bounding box."""
        return box[2] - box[0]
    
    @staticmethod
    def split_jaws(teeth_boxes):
        """
        Split teeth into upper and lower jaw using simple Y-median splitting.
        Fast and reliable for panoramic X-rays where jaws are clearly separated.
        
        Args:
            teeth_boxes: List of dicts with 'box' key
            
        Returns:
            (upper_jaw, lower_jaw) - lists of tooth detections
        """
        if len(teeth_boxes) < 2:
            # Not enough teeth to split
            return teeth_boxes, []
        
        # Get Y centers
        y_centers = [DentalGeometricEngine.get_box_center(t['box'])[1] for t in teeth_boxes]
        
        # Simple median split (much faster than KMeans)
        median_y = np.median(y_centers)
        
        upper_jaw = [t for t, y in zip(teeth_boxes, y_centers) if y < median_y]
        lower_jaw = [t for t, y in zip(teeth_boxes, y_centers) if y >= median_y]
        
        return upper_jaw, lower_jaw
    
    @staticmethod
    def sort_and_assign_fdi(jaw_teeth, jaw_type='upper'):
        """
        Sort teeth left-to-right and assign FDI numbers with gap detection.
        
        Args:
            jaw_teeth: List of tooth detections for one jaw
            jaw_type: 'upper' or 'lower'
            
        Returns:
            List of dicts with 'box', 'fdi', 'score', 'quadrant'
        """
        if not jaw_teeth:
            return []
        
        # Sort by X center (left to right)
        sorted_teeth = sorted(jaw_teeth, 
                             key=lambda t: DentalGeometricEngine.get_box_center(t['box'])[0])
        
        # FDI sequence for this jaw
        fdi_sequence = FDI_UPPER if jaw_type == 'upper' else FDI_LOWER
        
        # Assign FDI with gap detection
        result = []
        fdi_index = 0
        
        for i, tooth in enumerate(sorted_teeth):
            # Gap detection: check distance to previous tooth
            if i > 0 and fdi_index < len(fdi_sequence):
                prev_tooth = sorted_teeth[i - 1]
                curr_center_x = DentalGeometricEngine.get_box_center(tooth['box'])[0]
                prev_center_x = DentalGeometricEngine.get_box_center(prev_tooth['box'])[0]
                
                gap = curr_center_x - prev_center_x
                local_width = DentalGeometricEngine.get_box_width(tooth['box'])
                
                # If gap > 1.6x local width, assume missing tooth(s)
                if gap > local_width * 1.6:
                    missing_count = int(gap / local_width) - 1
                    missing_count = min(missing_count, 3)  # Cap at 3 missing
                    fdi_index += missing_count
            
            # Assign FDI number
            if fdi_index < len(fdi_sequence):
                fdi_number = fdi_sequence[fdi_index]
            else:
                fdi_number = fdi_sequence[-1]  # Fallback to last
            
            # Determine quadrant from FDI
            quadrant = fdi_number // 10
            
            result.append({
                'box': tooth['box'],
                'fdi': fdi_number,
                'score': tooth['score'],
                'quadrant': quadrant,
                'mask': tooth.get('mask'),
                'findings': []  # Will be populated by anomaly mapping
            })
            
            fdi_index += 1
        
        return result
    
    @staticmethod
    def compute_iou(box1, box2):
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    @staticmethod
    def map_anomalies(teeth, anomalies, iou_threshold=0.15):
        """
        Map anomalies to their associated teeth based on IoU overlap.
        
        Args:
            teeth: List of tooth dicts with 'box', 'fdi', etc.
            anomalies: List of anomaly dicts with 'box', 'class_name', 'score'
            iou_threshold: Minimum IoU to associate anomaly with tooth
            
        Returns:
            Updated teeth list with 'findings' populated
        """
        for anomaly in anomalies:
            anom_box = anomaly['box']
            best_tooth = None
            best_iou = iou_threshold
            
            for tooth in teeth:
                iou = DentalGeometricEngine.compute_iou(anom_box, tooth['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_tooth = tooth
            
            if best_tooth:
                finding = {
                    'type': anomaly['class_name'],
                    'score': anomaly['score'],
                    'box': anom_box,
                    'mask': anomaly.get('mask')  # Include mask for precise visualization
                }
                best_tooth['findings'].append(finding)
                anomaly['associated_fdi'] = best_tooth['fdi']
            else:
                # Anomaly without associated tooth - add as standalone (unmarked)
                anomaly['associated_fdi'] = None
        
        return teeth
    
    @staticmethod
    def compute_diou(box1, box2):
        """
        Compute Distance-IoU between two boxes.
        
        DIoU = IoU - (ρ²(b, b_gt) / c²)
        
        Where:
          ρ² = squared Euclidean distance between box centers
          c² = squared diagonal of the smallest enclosing box
        
        This penalizes detections whose centers are close together,
        even if their standard IoU is low (critical for packed molars).
        """
        # Standard IoU components
        iou = DentalGeometricEngine.compute_iou(box1, box2)
        
        # Center distance (ρ²)
        cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
        rho_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
        
        # Enclosing box diagonal (c²)
        enc_x1 = min(box1[0], box2[0])
        enc_y1 = min(box1[1], box2[1])
        enc_x2 = max(box1[2], box2[2])
        enc_y2 = max(box1[3], box2[3])
        c_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
        
        if c_sq == 0:
            return iou
        
        diou = iou - (rho_sq / c_sq)
        return diou
    
    @staticmethod
    def apply_nms(detections, iou_threshold=0.4):
        """
        Apply Distance-IoU (DIoU) based Non-Maximum Suppression.
        
        Uses DIoU instead of standard IoU to better handle tightly packed
        molars where center proximity matters more than overlap area.
        
        DIoU = IoU - (center_distance² / enclosing_diagonal²)
        
        Args:
            detections: List of detection dicts with 'box' and 'score'
            iou_threshold: DIoU threshold for suppression (0.4 recommended)
        """
        if not detections:
            return []
        
        # Sort by score (highest first)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove detections with high DIoU (overlapping OR close centers)
            remaining = []
            for det in detections:
                diou = DentalGeometricEngine.compute_diou(best['box'], det['box'])
                if diou < iou_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep


# =========================================================================
# Step 5  finding hierarchy
# =========================================================================

# --- app.py:1028-1078 (verbatim) ------------------------------------
def clean_clinical_findings(teeth):
    """
    Apply hierarchy rules to clean up findings per tooth.
    
    Hierarchy Rules:
    1. Root Canal > Filling (remove Filling) - Root canal implies filling material
    2. Crown > Filling (remove Filling) - Crown covers filling
    3. Keep highest scoring of same type (e.g. duplicate Caries)
    
    Args:
        teeth: List of tooth dicts with 'findings'
        
    Returns:
        Teeth list with cleaned findings
    """
    for tooth in teeth:
        findings = tooth.get('findings', [])
        if not findings:
            continue
            
        # 1. Deduplicate same type (keep max score)
        # Sort by score descending
        findings.sort(key=lambda x: x['score'], reverse=True)
        unique_findings = []
        seen_types = set()
        
        for f in findings:
            if f['type'] not in seen_types:
                unique_findings.append(f)
                seen_types.add(f['type'])
        
        # 2. Apply Hierarchy Rules
        types_present = seen_types
        clean_findings = []
        
        for f in unique_findings:
            t = f['type']
            
            # Rule: Root Canal hides Filling
            if t == 'Filling' and ('Root canal filling' in types_present or 'Root canal obturation' in types_present):
                continue
                
            # Rule: Crown hides Filling
            if t == 'Filling' and 'Crown' in types_present:
                continue
            
            clean_findings.append(f)
            
        tooth['findings'] = clean_findings
        
    return teeth


# =========================================================================
# Step 6  anatomical validation subsystems
# =========================================================================

# --- app.py:818-854 (verbatim) --------------------------------------
def recursive_prune_teeth(teeth, max_teeth=32, step=0.02, max_threshold=0.90):
    """
    Subsystem 5: Anatomical Count Check — Recursive Pruning
    
    If the teeth count exceeds the anatomical maximum (32 for adults),
    incrementally raise the confidence floor and remove the lowest-scoring
    teeth until the count is plausible.
    
    Mathematical logic:
      threshold_n = threshold_0 + n × step
      teeth_n = {t ∈ teeth | t.score ≥ threshold_n}
      Repeat until |teeth_n| ≤ max_teeth or threshold_n > max_threshold
    
    Args:
        teeth: List of tooth dicts with 'score'
        max_teeth: Maximum anatomically valid count (32)
        step: Confidence threshold increment per iteration
        max_threshold: Safety cap to prevent over-pruning
    
    Returns:
        Pruned teeth list with count ≤ max_teeth
    """
    if len(teeth) <= max_teeth:
        return teeth
    
    # Start from the minimum score in the current set
    current_threshold = min(t['score'] for t in teeth)
    
    while len(teeth) > max_teeth and current_threshold < max_threshold:
        current_threshold += step
        teeth = [t for t in teeth if t['score'] >= current_threshold]
    
    # If still over limit (unlikely), hard-prune by keeping top-N by score
    if len(teeth) > max_teeth:
        teeth = sorted(teeth, key=lambda t: t['score'], reverse=True)[:max_teeth]
    
    return teeth

# --- app.py:674-739 (verbatim) --------------------------------------
def resolve_fdi_conflicts(teeth):
    """
    Subsystem 3: Global Anatomical Constraint (GAC)
    
    Enforces Winner-Takes-All: no FDI number can appear more than once.
    
    Algorithm:
      1. Group teeth by FDI number
      2. For each duplicate group, the highest-confidence detection keeps the FDI
      3. Losers are reassigned to the nearest unoccupied FDI slot in the
         same quadrant, or discarded if no valid slot exists
    
    Args:
        teeth: List of tooth dicts with 'fdi', 'score', 'box', 'quadrant'
    
    Returns:
        Deduplicated teeth list with unique FDI assignments
    """
    if not teeth:
        return teeth
    
    # All valid FDI numbers for adult dentition
    VALID_FDIS = set(FDI_UPPER + FDI_LOWER)
    
    # Step 1: Find occupied FDI numbers and detect conflicts
    fdi_groups = {}
    for tooth in teeth:
        fdi = tooth['fdi']
        if fdi not in fdi_groups:
            fdi_groups[fdi] = []
        fdi_groups[fdi].append(tooth)
    
    resolved = []
    occupied_fdis = set()
    
    # Step 2: Process each FDI group
    for fdi, group in fdi_groups.items():
        if len(group) == 1:
            # No conflict — keep as-is
            resolved.append(group[0])
            occupied_fdis.add(fdi)
        else:
            # CONFLICT: Multiple teeth claim the same FDI
            # Winner-Takes-All: highest confidence keeps the label
            group_sorted = sorted(group, key=lambda t: t['score'], reverse=True)
            winner = group_sorted[0]
            resolved.append(winner)
            occupied_fdis.add(fdi)
            
            # Losers: try to reassign to nearest unoccupied FDI in same quadrant
            for loser in group_sorted[1:]:
                quadrant = loser['quadrant']
                # Get valid FDIs in this quadrant, sorted by distance to claimed FDI
                quadrant_fdis = [f for f in VALID_FDIS 
                                if f // 10 == quadrant and f not in occupied_fdis]
                
                if quadrant_fdis:
                    # Assign to nearest unoccupied slot
                    nearest = min(quadrant_fdis, key=lambda f: abs(f - fdi))
                    loser['fdi'] = nearest
                    loser['quadrant'] = nearest // 10
                    resolved.append(loser)
                    occupied_fdis.add(nearest)
                # else: discard — no valid slot available (anatomically impossible)
    
    return resolved

# --- app.py:742-815 (verbatim) --------------------------------------
def suppress_metal_artifacts(teeth, anomalies, proximity_px=5, penalty=0.5):
    """
    Subsystem 4: Metal Artifact Suppression (MAS)
    
    Radiographic flare from metal dental work (crowns, fillings, root canals)
    causes the model to hallucinate Caries near these structures.
    
    Rule: If a Caries detection overlaps with (IoU > 0.1) or is within
    `proximity_px` pixels of a radiopaque material, its confidence score
    is penalized by `penalty` (50%).
    
    Args:
        teeth: List of tooth dicts (with 'findings')
        anomalies: List of raw anomaly dicts
        proximity_px: Pixel proximity threshold for flare detection
        penalty: Score multiplier for penalized caries (0.5 = 50% reduction)
    
    Returns:
        Updated (teeth, anomalies) with penalized caries scores
    """
    METAL_CLASSES = {"Crown", "Filling", "Root canal filling", "Root canal obturation",
                     "Implant", "Prefabricated metal post"}
    
    def boxes_within_proximity(box1, box2, px):
        """Check if two boxes are within `px` pixels of each other."""
        # Expand box1 by px in all directions and check overlap
        expanded = [box1[0] - px, box1[1] - px, box1[2] + px, box1[3] + px]
        return compute_iou(expanded, box2) > 0
    
    # Collect all metal anomaly boxes (from both teeth findings and raw anomalies)
    metal_boxes = []
    
    # From teeth findings
    for tooth in teeth:
        for finding in tooth.get('findings', []):
            if finding['type'] in METAL_CLASSES:
                metal_boxes.append(finding['box'])
    
    # From raw anomalies
    for anom in anomalies:
        if anom.get('class_name', '') in METAL_CLASSES:
            metal_boxes.append(anom['box'])
    
    if not metal_boxes:
        return teeth, anomalies
    
    # Penalize caries in teeth findings
    for tooth in teeth:
        penalized_findings = []
        for finding in tooth.get('findings', []):
            if finding['type'] == 'Caries':
                near_metal = False
                for metal_box in metal_boxes:
                    iou = compute_iou(finding['box'], metal_box)
                    if iou > 0.1 or boxes_within_proximity(finding['box'], metal_box, proximity_px):
                        near_metal = True
                        break
                if near_metal:
                    finding['score'] *= penalty
                    finding['type'] = f"Caries (near metal, penalized)"
            penalized_findings.append(finding)
        tooth['findings'] = penalized_findings
    
    # Penalize caries in raw anomalies
    for anom in anomalies:
        if anom.get('class_name', '') == 'Caries':
            for metal_box in metal_boxes:
                iou = compute_iou(anom['box'], metal_box)
                if iou > 0.1 or boxes_within_proximity(anom['box'], metal_box, proximity_px):
                    anom['score'] *= penalty
                    anom['class_name'] = f"Caries (near metal, penalized)"
                    break
    
    return teeth, anomalies

# --- app.py:857-893 (verbatim) --------------------------------------
def validate_anatomy(teeth, anomalies, anomaly_threshold=0.15):
    """
    Subsystem 6: Master Anatomical Validation Orchestrator
    
    Called immediately before report generation. Applies all anatomical
    plausibility checks in the correct order:
    
    Pipeline:
      1. Recursive Prune     → teeth count ≤ 32
      2. Resolve FDI Conflicts → no duplicate FDI numbers
      3. Metal Artifact Suppression → penalize caries near metal
      4. Re-deduplicate anomalies  → remove penalized below threshold
    
    Args:
        teeth: List of tooth dicts (post-FDI assignment)
        anomalies: List of raw anomaly dicts
        anomaly_threshold: Minimum score for anomalies after penalties
    
    Returns:
        (validated_teeth, validated_anomalies) — anatomically plausible results
    """
    # Step 1: Enforce maximum tooth count (32 for adult dentition)
    teeth = recursive_prune_teeth(teeth, max_teeth=32)
    
    # Step 2: Resolve duplicate FDI assignments (Winner-Takes-All)
    teeth = resolve_fdi_conflicts(teeth)
    
    # Step 3: Suppress radiographic flare artifacts near metal work
    teeth, anomalies = suppress_metal_artifacts(teeth, anomalies)
    
    # Step 4: Remove anomalies whose scores fell below threshold after penalties
    for tooth in teeth:
        tooth['findings'] = [f for f in tooth.get('findings', []) 
                            if f['score'] >= anomaly_threshold]
    anomalies = [a for a in anomalies if a['score'] >= anomaly_threshold]
    
    return teeth, anomalies



# ---------------------------------------------------------------------------
# THE ONLY DEVIATION FROM app.py
# ---------------------------------------------------------------------------
def _is_tooth(class_name, cls_id, fix_cls_routing=False):
    """Decide whether a detection is a tooth.

    app.py:953-958 reads, verbatim:

        # DEVIATION: original six lines are quoted verbatim in _is_tooth().
        is_tooth = _is_tooth(class_name, cls_id, fix_cls_routing)

    With the 9-class model this is wrong. That model's class ids are
    0=Tooth, 1=Caries, 2=Crown, 3=Filling, 4=Implant, 5=Prefabricated metal
    post, 6=Retained root, 7=Root canal filling, 8=Root canal obturation --
    every one of them <= 32. So the `elif` fires for all eight anomaly
    classes, and every anomaly the primary model finds is routed into the
    TEETH list: it enters DIoU-NMS, the y-median jaw split, FDI assignment,
    recursive pruning and FDI conflict resolution. The
    `elif class_name in ANOMALY_CLASSES` branch below it is unreachable for
    the 9-class model.

    fix_cls_routing=False reproduces the deployed behaviour exactly (default).
    fix_cls_routing=True  applies the correction: the cls_id fallback is used
                          only when the class name is not a known anomaly.
    """
    if class_name == "Tooth":
        return True
    if fix_cls_routing and class_name in ANOMALY_CLASSES:
        return False
    return cls_id <= 32
