#!/usr/bin/env python3
"""
Video Tracking Example for Selective Tracking
Demonstrates natural language-driven multi-object tracking using Grounding DINO + CLIP-Enhanced ByteTrack
"""
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

from groundingdino.util.inference import load_model, load_image, predict
from selectivetrack.clip_tracker import CLIPTracker


def draw_tracks(frame, tracks, colors=None):
    """
    Draw bounding boxes and track IDs on frame.
    
    Args:
        frame: Input frame (numpy array)
        tracks: List of track objects with tlwh and track_id attributes
        colors: Optional dict mapping track_id to BGR color tuple
    
    Returns:
        Annotated frame
    """
    if colors is None:
        colors = {}
    
    for track in tracks:
        if not hasattr(track, 'tlwh') or not hasattr(track, 'track_id'):
            continue
            
        tlwh = track.tlwh
        track_id = track.track_id
        
        x1, y1, w, h = tlwh
        x2, y2 = x1 + w, y1 + h
        
        # Get or generate color for this track
        if track_id not in colors:
            np.random.seed(track_id)
            colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        
        color = colors[track_id]
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw track ID label
        label = f"ID: {track_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 4), 
                     (int(x1) + label_size[0], int(y1)), color, -1)
        cv2.putText(frame, label, (int(x1), int(y1) - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description='Video tracking with Selective Tracking')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--prompt', type=str, required=True, 
                       help='Text prompt for detection (e.g., "red car", "person walking")')
    parser.add_argument('--output', type=str, default='output.mp4', 
                       help='Path to output video file')
    parser.add_argument('--config', type=str, 
                       default='groundingdino/config/GroundingDINO_SwinT_OGC.py',
                       help='Path to Grounding DINO config file')
    parser.add_argument('--weights', type=str,
                       default='weights/groundingdino_swint_ogc.pth',
                       help='Path to Grounding DINO weights')
    parser.add_argument('--box-threshold', type=float, default=0.35,
                       help='Box confidence threshold')
    parser.add_argument('--text-threshold', type=float, default=0.25,
                       help='Text confidence threshold')
    parser.add_argument('--track-thresh', type=float, default=0.5,
                       help='Tracking confidence threshold')
    parser.add_argument('--track-buffer', type=int, default=30,
                       help='Tracking buffer size (frames)')
    parser.add_argument('--match-thresh', type=float, default=0.8,
                       help='Matching threshold for tracking')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--display', action='store_true',
                       help='Display video while processing')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("Loading Grounding DINO model...")
    model = load_model(args.config, args.weights, device=args.device)
    
    print("Initializing CLIP tracker...")
    # Initialize tracker with CLIP model for appearance matching
    tracker = CLIPTracker(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        device=args.device
    )
    
    # Open video
    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"Output will be saved to: {args.output}")
    print(f"Detection prompt: '{args.prompt}'")
    print("\nProcessing video...")
    
    frame_idx = 0
    colors = {}  # Track ID to color mapping
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Convert BGR to RGB for Grounding DINO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            boxes, logits, phrases = predict(
                model=model,
                image=frame_rgb,
                caption=args.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.device
            )
            
            # Convert boxes to xyxy format and scale to image size
            if len(boxes) > 0:
                boxes_xyxy = boxes * torch.tensor([width, height, width, height])
                boxes_np = boxes_xyxy.cpu().numpy()
                scores_np = logits.cpu().numpy()
                
                # Prepare detections with scores
                dets = np.concatenate([boxes_np, scores_np[:, None]], axis=1)
            else:
                dets = np.empty((0, 5))
            
            # Update tracker
            if hasattr(tracker, 'update_with_clip'):
                # For CLIP-enhanced tracker, pass frame for appearance features
                tracks = tracker.update_with_clip(
                    dets, 
                    frame_rgb,
                    args.prompt
                )
            else:
                # For basic tracker
                tracks = tracker.update(dets, [height, width], [height, width])
            
            # Draw tracks on frame
            frame_annotated = draw_tracks(frame.copy(), tracks, colors)
            
            # Add frame info
            info_text = f"Frame: {frame_idx}/{total_frames} | Tracks: {len(tracks)}"
            cv2.putText(frame_annotated, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write to output
            out.write(frame_annotated)
            
            # Display if requested
            if args.display:
                cv2.imshow('Tracking', frame_annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopping (user interrupt)...")
                    break
            
            # Print progress
            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames "
                      f"({100*frame_idx/total_frames:.1f}%) - {len(tracks)} active tracks")
    
    finally:
        # Clean up
        cap.release()
        out.release()
        if args.display:
            cv2.destroyAllWindows()
    
    print(f"\nDone! Output saved to: {args.output}")
    print(f"Processed {frame_idx} frames")


if __name__ == '__main__':
    main()
