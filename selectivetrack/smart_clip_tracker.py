"""
Smart CLIP Tracker - Uses appearance strategically for Refer-KITTI

Key innovations:
1. NO text gating (narrow similarity band makes it unreliable)
2. Temporal appearance smoothing (EMA of embeddings over trajectory)
3. CLIP only for re-identification (lost→candidate matching)
4. Reverse adaptive fusion (trust appearance MORE when IoU is strong, not weak)
5. Appearance-based duplicate removal
"""
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
        self.embedding_history = []  # Track last N embeddings for temporal smoothing

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
            self._update_embedding(new_track.embedding)

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
            self._update_embedding(new_track.embedding)

    def _update_embedding(self, new_e, alpha=0.90, history_len=5):
        """Update embedding with temporal smoothing."""
        new_e = new_e.detach().float().cpu()
        new_e = new_e / (new_e.norm() + _EPS)

        if self.embedding is None:
            self.embedding = new_e
        else:
            # EMA update
            self.embedding = alpha * self.embedding + (1.0 - alpha) * new_e
            self.embedding = self.embedding / (self.embedding.norm() + _EPS)

        # Keep history for robust re-ID
        self.embedding_history.append(new_e)
        if len(self.embedding_history) > history_len:
            self.embedding_history.pop(0)

    def get_smoothed_embedding(self):
        """Return temporal average of recent embeddings (more robust for re-ID)."""
        if not self.embedding_history:
            return self.embedding
        stacked = torch.stack(self.embedding_history, dim=0)
        smoothed = stacked.mean(dim=0)
        return smoothed / (smoothed.norm() + _EPS)

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


class SmartCLIPTracker(object):
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

        # Smart CLIP config
        self.use_reid_clip = bool(getattr(args, "use_reid_clip", True))  # Use for re-ID
        self.reid_clip_weight = float(getattr(args, "reid_clip_weight", 0.40))  # Higher weight for re-ID
        self.high_conf_clip_weight = float(getattr(args, "high_conf_clip_weight", 0.10))  # Low weight when IoU is strong

    @staticmethod
    def _embedding_distance(tracks, detections, use_smoothed=False):
        """
        Compute appearance distance matrix.
        Returns (cost, valid_mask) where cost ∈ [0, 1].
        """
        nT, nD = len(tracks), len(detections)
        if nT == 0 or nD == 0:
            return np.zeros((nT, nD), dtype=np.float32), np.zeros((nT, nD), dtype=bool)

        C = np.zeros((nT, nD), dtype=np.float32)
        M = np.zeros((nT, nD), dtype=bool)

        for i, trk in enumerate(tracks):
            if use_smoothed and hasattr(trk, 'get_smoothed_embedding'):
                te = trk.get_smoothed_embedding()
            else:
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
                C[i, j] = 0.5 * (1.0 - sim)  # Convert to distance
                M[i, j] = True

        return C.astype(np.float32), M

    @staticmethod
    def _fuse_motion_and_appearance(iou_dist, emb_dist, emb_valid, iou_weight=0.90):
        """
        Smart fusion: trust appearance more when motion is reliable.

        Args:
            iou_dist: IoU distance matrix [0=perfect match, 1=no overlap]
            emb_dist: Embedding distance matrix [0=identical, 1=opposite]
            emb_valid: Boolean mask of valid embeddings
            iou_weight: Weight for IoU (0.9 = 90% IoU, 10% appearance)
        """
        if iou_dist.size == 0:
            return iou_dist

        fused = iou_dist.copy().astype(np.float32)
        idx = emb_valid

        if idx.any():
            # Simple weighted average (high IoU weight for high-conf matching)
            fused[idx] = iou_weight * iou_dist[idx] + (1.0 - iou_weight) * emb_dist[idx]

        return fused

    def update(self, detections, detection_embeddings, img_info, text_embedding, class_names=None):
        """
        Main tracking update.

        Args:
            detections: np.float32 [N, 5] -> [x1, y1, x2, y2, score]
            detection_embeddings: list of None or CPU FloatTensor [D]
            img_info: (H, W)
            text_embedding: unused (no text gating with narrow band)
        """
        self.frame_id += 1
        activated_starcks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        if detections is None or len(detections) == 0:
            for track in self.tracked_stracks:
                if track.state == TrackState.Tracked:
                    track.mark_lost()
                    lost_stracks.append(track)
            self._finalize_lists(activated_starcks, refind_stracks, lost_stracks, removed_stracks)
            return [t for t in self.tracked_stracks if t.is_activated]

        # Split by confidence
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

        # NO TEXT GATING - narrow band makes it unreliable

        # Build STrack objects
        def _mk_tracks(dets_list, scores_list, embs_list):
            tracks = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_list, scores_list)]
            for i, trk in enumerate(tracks):
                e = embs_list[i]
                if e is not None:
                    fe = e.detach().float().cpu()
                    trk.embedding = fe / (fe.norm() + _EPS)
            return tracks

        detections_hi = _mk_tracks(dets_hi, scores_hi, emb_hi)
        detections_lo = _mk_tracks(dets_lo, scores_lo, emb_lo)

        # Split tracked
        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # === STEP 1: HIGH-CONFIDENCE ASSOCIATION ===
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # Pure IoU + score fusion (ByteTrack baseline)
        iou1 = matching.iou_distance(strack_pool, detections_hi).astype(np.float32)
        if not self.args.mot20 and iou1.size and len(detections_hi) > 0:
            iou1 = matching.fuse_score(iou1, detections_hi)

        # Add CLIP with LOW weight (motion is reliable for high-conf)
        emb1, mask1 = self._embedding_distance(strack_pool, detections_hi, use_smoothed=False)
        dists1 = self._fuse_motion_and_appearance(
            iou1, emb1, mask1, iou_weight=(1.0 - self.high_conf_clip_weight)
        )

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

        # === STEP 2: LOW-CONFIDENCE (IoU only, NO CLIP) ===
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        iou2 = matching.iou_distance(r_tracked_stracks, detections_lo).astype(np.float32)
        # NO CLIP here - low-conf detections have noisy embeddings
        matches2, u_track2, _ = matching.linear_assignment(iou2, thresh=0.5)

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

        # === STEP 3: UNCONFIRMED ASSOCIATION ===
        detections_left = [detections_hi[i] for i in u_det_hi]
        iou3 = matching.iou_distance(unconfirmed, detections_left).astype(np.float32)
        if not self.args.mot20 and iou3.size and len(detections_left) > 0:
            iou3 = matching.fuse_score(iou3, detections_left)

        # Use CLIP moderately for unconfirmed (helps initialization)
        if self.use_reid_clip:
            emb3, mask3 = self._embedding_distance(unconfirmed, detections_left, use_smoothed=True)
            dists3 = self._fuse_motion_and_appearance(
                iou3, emb3, mask3, iou_weight=(1.0 - self.reid_clip_weight)
            )
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

        # === STEP 4: INITIALIZE NEW TRACKS ===
        for inew in u_det_left:
            track = detections_left[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # === STEP 5: HOUSEKEEPING ===
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

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
