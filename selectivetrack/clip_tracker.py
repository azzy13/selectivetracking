import numpy as np
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from selectivetrack import matching
from .basetrack import BaseTrack, TrackState

_EPS = 1e-6

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = float(score)
        self.tracklet_len = 0
        self.embedding = None  # CPU torch.FloatTensor [D], unit norm

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = float(new_track.score)
        if new_track.embedding is not None:
            self.embedding = new_track.embedding.detach().float().cpu()
            self.embedding = self.embedding / (self.embedding.norm() + _EPS)

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = float(new_track.score)

        if new_track.embedding is not None:
            new_e = new_track.embedding.detach().float().cpu()
            if self.embedding is None:
                self.embedding = new_e
            else:
                self.embedding = 0.9 * self.embedding + 0.1 * new_e
            self.embedding = self.embedding / (self.embedding.norm() + _EPS)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-12)
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class CLIPTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = float(args.track_thresh) + 0.10
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # optional: only use CLIP in low/unconfirmed passes
        self.use_clip_in_high = bool(getattr(args, "use_clip_in_high", False))
        self.use_clip_in_low = bool(getattr(args, "use_clip_in_low", True))
        self.use_clip_in_unconf = bool(getattr(args, "use_clip_in_unconf", True))

        # text-grounding at matching time (strict expression enforcement)
        self.use_text_gate_matching = bool(getattr(args, "use_text_gate_matching", True))
        self.text_gate_mode = str(getattr(args, "text_gate_mode", "penalty"))  # "penalty" or "hard"
        self.text_gate_weight = float(getattr(args, "text_gate_weight", 0.5))  # weight for penalty mode

    # ---------- helpers ----------
    @staticmethod
    def _embedding_cost(tracks, detections):
        """
        Return (cost, valid_mask).
        cost[i,j] = 0.5 * (1 - cosine) in [0,1] WHEN BOTH embeddings exist.
        If either emb missing -> valid_mask[i,j]=False (callers fallback to IoU).
        """
        nT, nD = len(tracks), len(detections)
        if nT == 0 or nD == 0:
            C = np.zeros((nT, nD), dtype=np.float32)
            M = np.zeros((nT, nD), dtype=bool)
            return C, M

        C = np.zeros((nT, nD), dtype=np.float32)
        M = np.zeros((nT, nD), dtype=bool)
        for i, trk in enumerate(tracks):
            te = getattr(trk, "embedding", None)
            if te is None:
                continue
            te = te.float().view(-1)
            for j, det in enumerate(detections):
                de = getattr(det, "embedding", None)
                if de is None:
                    continue
                de = de.float().view(-1)
                sim = F.cosine_similarity(te.unsqueeze(0), de.unsqueeze(0), dim=-1).item()
                sim = max(min(sim, 1.0), -1.0)
                C[i, j] = 0.5 * (1.0 - sim)  # 0 best, 1 worst
                M[i, j] = True
        return C.astype(np.float32), M

    @staticmethod
    def _text_grounding_cost(detections, text_embedding, text_sim_thresh=0.25):
        """
        Compute text-to-image similarity cost for each detection.

        Args:
            detections: List of STrack objects with .embedding attribute
            text_embedding: torch.Tensor [C, D] (C classes, D=512 for CLIP)
            text_sim_thresh: Minimum similarity to consider a match

        Returns:
            cost: np.ndarray [N] where N=len(detections)
                  0.0 if text_sim >= thresh (good match to expression)
                  1.0 if text_sim < thresh (violates expression)
            valid_mask: np.ndarray [N] bool, True if detection has embedding
        """
        nD = len(detections)
        if nD == 0:
            return np.zeros(nD, dtype=np.float32), np.zeros(nD, dtype=bool)

        cost = np.zeros(nD, dtype=np.float32)
        valid = np.zeros(nD, dtype=bool)

        # Normalize text embedding (should already be normalized, but ensure)
        tnorm = text_embedding / (text_embedding.norm(dim=-1, keepdim=True) + 1e-6)

        for j, det in enumerate(detections):
            de = getattr(det, "embedding", None)
            if de is None:
                # No embedding → can't compute text similarity → assume valid
                cost[j] = 0.0
                valid[j] = False
                continue

            # Normalize detection embedding
            v = de.detach().float()
            v = v / (v.norm() + 1e-6)

            # Compute max text-to-image similarity across all class text embeddings
            # text_embedding: [C, D], v: [D] → sim: [C]
            sim = torch.max(torch.matmul(tnorm, v.to(tnorm.device)))
            sim_val = sim.item()

            # Convert to cost: high sim → low cost
            if sim_val >= text_sim_thresh:
                cost[j] = 0.0  # Matches expression well
            else:
                cost[j] = 1.0  # Violates expression

            valid[j] = True

        return cost.astype(np.float32), valid

    @staticmethod
    def _fuse_iou_and_clip(iou_for_blend, emb_cost, emb_valid_mask, *,
                           lambda_weight=0.23, adaptive=True, iou_for_weight=None):
        """
        Blend IoU distance (in [0,1]) with appearance cost (in [0,1]).
        - Use iou_for_weight = RAW IoU distance (without score fusion!) for the adaptive term.
        - Do NO HARM where emb is unavailable (fall back to IoU).
        """
        if iou_for_blend.size == 0:
            return iou_for_blend

        fused = iou_for_blend.copy().astype(np.float32)
        if iou_for_weight is None:
            iou_for_weight = iou_for_blend

        if adaptive:
            w = lambda_weight * iou_for_weight  # more appearance when IoU is weak
        else:
            w = np.full_like(fused, fill_value=lambda_weight, dtype=np.float32)

        idx = emb_valid_mask
        if idx.any():
            fused[idx] = (1.0 - w[idx]) * iou_for_blend[idx] + w[idx] * emb_cost[idx]
        return fused

    @staticmethod
    def _fuse_iou_clip_and_text(
        iou_dist, emb_cost, emb_valid, text_cost_per_det, text_valid_per_det,
        *, lambda_visual=0.23, lambda_text=0.5, text_mode="penalty", adaptive=True, iou_for_weight=None
    ):
        """
        Three-way fusion: IoU + Visual CLIP + Text-grounding.

        Args:
            iou_dist: [M, N] IoU-based distance
            emb_cost: [M, N] Visual CLIP cost (track_emb ↔ det_emb)
            emb_valid: [M, N] bool mask for valid visual embeddings
            text_cost_per_det: [N] Text-grounding cost (0.0=good, 1.0=bad)
            text_valid_per_det: [N] bool mask for valid text costs
            lambda_visual: Weight for visual CLIP cost
            lambda_text: Weight for text-grounding cost
            text_mode: "penalty" (soft gating) or "hard" (block bad matches)
            adaptive: Use adaptive visual weighting based on IoU
            iou_for_weight: [M, N] raw IoU for adaptive weighting

        Returns:
            fused_cost: [M, N] Combined distance matrix
        """
        nT, nD = iou_dist.shape
        if nT == 0 or nD == 0:
            return iou_dist.copy()

        # Start with IoU distance
        fused = iou_dist.copy().astype(np.float32)
        if iou_for_weight is None:
            iou_for_weight = iou_dist

        # Broadcast text cost to [M, N]
        text_cost_matrix = np.tile(text_cost_per_det.reshape(1, -1), (nT, 1))  # [M, N]

        # Apply text-grounding penalty/gate
        if text_mode == "hard":
            # Hard gating: Block matches to detections that violate expression
            for j in range(nD):
                if text_valid_per_det[j] and text_cost_per_det[j] > 0.5:  # Bad match
                    fused[:, j] = 999.0  # Block all tracks from matching this detection
        else:
            # Soft penalty: Add text cost weighted by lambda_text
            text_mask = text_valid_per_det.astype(bool)  # [N]
            if text_mask.any():
                # Add text penalty to all valid text detections
                fused[:, text_mask] = fused[:, text_mask] + lambda_text * text_cost_matrix[:, text_mask]

        # Fuse visual CLIP cost (adaptive weighting)
        if adaptive:
            w_visual = lambda_visual * iou_for_weight  # [M, N]
        else:
            w_visual = np.full_like(fused, fill_value=lambda_visual, dtype=np.float32)

        idx = emb_valid  # [M, N] bool mask
        if idx.any():
            # fused = fused + w_visual * emb_cost (additive since text already added)
            # Actually, let's do proper normalization:
            # fused = (1 - w_visual) * iou + w_visual * emb + lambda_text * text
            # But text already added, so just add visual component
            fused[idx] = (1.0 - w_visual[idx]) * iou_dist[idx] + w_visual[idx] * emb_cost[idx]
            # Re-add text penalty on top
            if text_mode == "penalty":
                for j in range(nD):
                    if text_valid_per_det[j]:
                        fused[:, j] = fused[:, j] + lambda_text * text_cost_per_det[j]

        return fused

    # ---------- main update ----------
    def update(self, detections, detection_embeddings, img_info, text_embedding, class_names=None):
        """
        detections: np.float32 [N,5] -> [x1,y1,x2,y2,score]
        detection_embeddings: list of None or CPU FloatTensor [D] (unit norm)
        text_embedding: torch.FloatTensor [C,D] (unit norm, on device, FP32)
        """
        self.frame_id += 1
        activated_starcks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # ===== no detections =====
        if detections is None or len(detections) == 0:
            for track in self.tracked_stracks:
                if track.state == TrackState.Tracked:
                    track.mark_lost()
                    lost_stracks.append(track)
            self._finalize_lists(activated_starcks, refind_stracks, lost_stracks, removed_stracks)
            return [t for t in self.tracked_stracks if t.is_activated]

        # ----- split boxes/scores -----
        dets_np = detections.astype(np.float32, copy=False)
        scores = dets_np[:, 4]
        bboxes = dets_np[:, :4]

        high_thr = float(self.args.track_thresh)
        low_thr = float(getattr(self.args, "low_thresh", 0.1))

        remain_inds = scores > high_thr
        low_mask = (scores > low_thr) & (scores <= high_thr)

        dets_hi = bboxes[remain_inds]
        scores_hi = scores[remain_inds]
        emb_hi = [detection_embeddings[i] for i in range(len(detection_embeddings)) if remain_inds[i]]

        dets_lo = bboxes[low_mask]
        scores_lo = scores[low_mask]
        emb_lo = [detection_embeddings[i] for i in range(len(detection_embeddings)) if low_mask[i]]

        # ----- text-sim gate (NEVER drop when emb is None) -----
        tnorm = text_embedding / (text_embedding.norm(dim=-1, keepdim=True) + 1e-6)
        sim_thr = float(getattr(self.args, "text_sim_thresh", 0.0))

        def _gate_and_build(dets_xyxy, scores_vec, embs_list):
            keep_dets, keep_scores, keep_embs = [], [], []
            for i, e in enumerate(embs_list):
                if e is None:
                    keep_dets.append(dets_xyxy[i]); keep_scores.append(scores_vec[i]); keep_embs.append(None)
                    continue
                v = e.detach().float()
                v = v / (v.norm() + 1e-6)
                sim = torch.max(torch.matmul(tnorm, v.to(tnorm.device)))
                if sim.item() >= sim_thr:
                    keep_dets.append(dets_xyxy[i]); keep_scores.append(scores_vec[i]); keep_embs.append(e)
            return keep_dets, keep_scores, keep_embs

        if len(dets_hi):
            dets_hi, scores_hi, emb_hi = _gate_and_build(dets_hi, scores_hi, emb_hi)
        else:
            dets_hi, scores_hi, emb_hi = [], [], []
        if len(dets_lo):
            dets_lo, scores_lo, emb_lo = _gate_and_build(dets_lo, scores_lo, emb_lo)
        else:
            dets_lo, scores_lo, emb_lo = [], [], []

        # ----- wrap as STrack + attach normalized embeddings -----
        def _mk_tracks(dets_list, scores_list, embs_list):
            tracks = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_list, scores_list)]
            for i, trk in enumerate(tracks):
                e = embs_list[i]
                if e is not None:
                    fe = e.detach().float().cpu()
                    trk.embedding = fe / (fe.norm() + 1e-6)
            return tracks

        detections_hi = _mk_tracks(dets_hi, scores_hi, emb_hi)
        detections_lo = _mk_tracks(dets_lo, scores_lo, emb_lo)

        # ----- split confirmed/unconfirmed -----
        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # ===== Step 1: HIGH-score association =====
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # ByteTrack: IoU (+ score fusion). Keep CLIP out of this to protect MOTA.
        iou1 = matching.iou_distance(strack_pool, detections_hi).astype(np.float32)
        if not self.args.mot20 and iou1.size and len(detections_hi) > 0:
            iou1 = matching.fuse_score(iou1, detections_hi)

        # Compute text-grounding cost for detections (expression enforcement)
        text_cost_hi, text_valid_hi = np.zeros(0), np.zeros(0, dtype=bool)
        if self.use_text_gate_matching and len(detections_hi) > 0:
            text_sim_thresh = float(getattr(self.args, "text_sim_thresh", 0.25))
            text_cost_hi, text_valid_hi = self._text_grounding_cost(
                detections_hi, text_embedding, text_sim_thresh
            )

        if self.use_clip_in_high:
            # optional re-blend (rarely needed on KITTI)
            iou1_raw = matching.iou_distance(strack_pool, detections_hi).astype(np.float32)
            emb1, mask1 = self._embedding_cost(strack_pool, detections_hi)

            # Use three-way fusion if text-grounding is enabled
            if self.use_text_gate_matching and len(text_cost_hi) > 0:
                dists1 = self._fuse_iou_clip_and_text(
                    iou1, emb1, mask1, text_cost_hi, text_valid_hi,
                    lambda_visual=self.args.lambda_weight,
                    lambda_text=self.text_gate_weight,
                    text_mode=self.text_gate_mode,
                    adaptive=True,
                    iou_for_weight=iou1_raw
                )
            else:
                dists1 = self._fuse_iou_and_clip(
                    iou1, emb1, mask1,
                    lambda_weight=self.args.lambda_weight, adaptive=True, iou_for_weight=iou1_raw
                )
        else:
            # No visual CLIP, but still apply text-grounding if enabled
            if self.use_text_gate_matching and len(text_cost_hi) > 0:
                dists1 = iou1.copy()
                if self.text_gate_mode == "hard":
                    for j in range(len(detections_hi)):
                        if text_valid_hi[j] and text_cost_hi[j] > 0.5:
                            dists1[:, j] = 999.0
                else:
                    for j in range(len(detections_hi)):
                        if text_valid_hi[j]:
                            dists1[:, j] += self.text_gate_weight * text_cost_hi[j]
            else:
                dists1 = iou1

        matches1, u_track, u_det_hi = matching.linear_assignment(dists1, thresh=self.args.match_thresh)

        for itracked, idet in matches1:
            track = strack_pool[itracked]
            det = detections_hi[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # ===== Step 2: LOW-score association =====
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        iou2 = matching.iou_distance(r_tracked_stracks, detections_lo).astype(np.float32)

        # Compute text-grounding cost for low-confidence detections
        text_cost_lo, text_valid_lo = np.zeros(0), np.zeros(0, dtype=bool)
        if self.use_text_gate_matching and len(detections_lo) > 0:
            text_sim_thresh = float(getattr(self.args, "text_sim_thresh", 0.25))
            text_cost_lo, text_valid_lo = self._text_grounding_cost(
                detections_lo, text_embedding, text_sim_thresh
            )

        if self.use_clip_in_low:
            emb2, mask2 = self._embedding_cost(r_tracked_stracks, detections_lo)

            # Use three-way fusion if text-grounding is enabled
            if self.use_text_gate_matching and len(text_cost_lo) > 0:
                dists2 = self._fuse_iou_clip_and_text(
                    iou2, emb2, mask2, text_cost_lo, text_valid_lo,
                    lambda_visual=self.args.lambda_weight,
                    lambda_text=self.text_gate_weight,
                    text_mode=self.text_gate_mode,
                    adaptive=True,
                    iou_for_weight=iou2
                )
            else:
                dists2 = self._fuse_iou_and_clip(
                    iou2, emb2, mask2,
                    lambda_weight=self.args.lambda_weight, adaptive=True, iou_for_weight=iou2
                )
        else:
            # No visual CLIP, but still apply text-grounding if enabled
            if self.use_text_gate_matching and len(text_cost_lo) > 0:
                dists2 = iou2.copy()
                if self.text_gate_mode == "hard":
                    for j in range(len(detections_lo)):
                        if text_valid_lo[j] and text_cost_lo[j] > 0.5:
                            dists2[:, j] = 999.0
                else:
                    for j in range(len(detections_lo)):
                        if text_valid_lo[j]:
                            dists2[:, j] += self.text_gate_weight * text_cost_lo[j]
            else:
                dists2 = iou2

        matches2, u_track2, _ = matching.linear_assignment(dists2, thresh=0.5)

        for itracked, idet in matches2:
            track = r_tracked_stracks[itracked]
            det = detections_lo[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track2:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ===== Step 3: Unconfirmed association (use remaining HIGHs) =====
        detections_left = [detections_hi[i] for i in u_det_hi]

        iou3 = matching.iou_distance(unconfirmed, detections_left).astype(np.float32)
        if not self.args.mot20 and iou3.size and len(detections_left) > 0:
            iou3 = matching.fuse_score(iou3, detections_left)

        # Compute text-grounding cost for remaining high-confidence detections
        text_cost_left, text_valid_left = np.zeros(0), np.zeros(0, dtype=bool)
        if self.use_text_gate_matching and len(detections_left) > 0:
            text_sim_thresh = float(getattr(self.args, "text_sim_thresh", 0.25))
            text_cost_left, text_valid_left = self._text_grounding_cost(
                detections_left, text_embedding, text_sim_thresh
            )

        if self.use_clip_in_unconf:
            iou3_raw = matching.iou_distance(unconfirmed, detections_left).astype(np.float32)
            emb3, mask3 = self._embedding_cost(unconfirmed, detections_left)

            # Use three-way fusion if text-grounding is enabled
            if self.use_text_gate_matching and len(text_cost_left) > 0:
                dists3 = self._fuse_iou_clip_and_text(
                    iou3, emb3, mask3, text_cost_left, text_valid_left,
                    lambda_visual=self.args.lambda_weight,
                    lambda_text=self.text_gate_weight,
                    text_mode=self.text_gate_mode,
                    adaptive=True,
                    iou_for_weight=iou3_raw
                )
            else:
                dists3 = self._fuse_iou_and_clip(
                    iou3, emb3, mask3,
                    lambda_weight=self.args.lambda_weight, adaptive=True, iou_for_weight=iou3_raw
                )
        else:
            # No visual CLIP, but still apply text-grounding if enabled
            if self.use_text_gate_matching and len(text_cost_left) > 0:
                dists3 = iou3.copy()
                if self.text_gate_mode == "hard":
                    for j in range(len(detections_left)):
                        if text_valid_left[j] and text_cost_left[j] > 0.5:
                            dists3[:, j] = 999.0
                else:
                    for j in range(len(detections_left)):
                        if text_valid_left[j]:
                            dists3[:, j] += self.text_gate_weight * text_cost_left[j]
            else:
                dists3 = iou3

        matches3, u_unconfirmed, u_det_left = matching.linear_assignment(dists3, thresh=0.7)

        for itracked, idet in matches3:
            unconfirmed[itracked].update(detections_left[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # ===== Step 4: Initialize new tracks from remaining HIGH detections =====
        for inew in u_det_left:
            track = detections_left[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # ===== Step 5: Housekeeping =====
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ===== Finalize lists =====
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        return [track for track in self.tracked_stracks if track.is_activated]

    def _finalize_lists(self, activated, refind, lost, removed):
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)


def joint_stracks(tlista, tlistb):
    exists, res = {}, []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb).astype(np.float32)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
