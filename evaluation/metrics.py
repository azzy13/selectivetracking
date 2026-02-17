# compute_metrics.py
import os, glob
import pandas as pd
import motmetrics as mm

class MotMetricsEvaluator:
    def __init__(self, distth=0.5, fmt='mot15-2D'):
        self.distth = distth
        self.fmt = fmt
        self.mh = mm.metrics.create()
        self.metrics = [
            'num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr',
            'precision', 'recall', 'num_switches',
            'mostly_tracked', 'mostly_lost','partially_tracked', 'num_fragmentations',
            'num_false_positives', 'num_misses', 'num_objects', 'num_matches',
        ]

    def _pair(self, gt_folder, res_folder):
        # Try nested structure first (MOT17 style)
        gt_files = glob.glob(os.path.join(gt_folder, "*", "gt", "gt.txt"))
        
        # Fallback to flat format (VisDrone style)
        if not gt_files:
            gt_files = glob.glob(os.path.join(gt_folder, "*.txt"))
        
        res_files = glob.glob(os.path.join(res_folder, "*.txt"))
        print(f"📂 Found {len(gt_files)} GT files and {len(res_files)} result files")
        
        # Map by sequence name
        if gt_files and "/gt/gt.txt" in gt_files[0]:
            # Nested: extract from path/to/sequence_name/gt/gt.txt
            gt_map = {os.path.basename(os.path.dirname(os.path.dirname(p))): p 
                    for p in gt_files}
        else:
            # Flat: extract from sequence_name.txt
            gt_map = {os.path.splitext(os.path.basename(p))[0]: p 
                    for p in gt_files}
        
        res_map = {os.path.splitext(os.path.basename(p))[0]: p 
                for p in res_files}

        common = sorted(set(gt_map) & set(res_map))
        if not common:
            print(f"⚠ No matches — GT: {list(gt_map.keys())[:5]}, RES: {list(res_map.keys())[:5]}")
        return [(name, gt_map[name], res_map[name]) for name in common]


    def evaluate(self, gt_folder, res_folder, verbose=True):
        triples = self._pair(gt_folder, res_folder)
        if not triples:
            if verbose: print("No matching GT/RES files.")
            return None

        summaries = []
        for name, gt_p, res_p in triples:
            gt = mm.io.loadtxt(gt_p, fmt=self.fmt, min_confidence=1)
            res = mm.io.loadtxt(res_p, fmt=self.fmt)
            acc = mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=self.distth)
            summary = self.mh.compute(acc, metrics=self.metrics, name=name)
            summaries.append(summary)
            if verbose:
                print(f"\n===== {name} =====")
                print(mm.io.render_summary(summary,
                      namemap=mm.io.motchallenge_metric_names,
                      formatters=self.mh.formatters))

        df = pd.concat(summaries)
        avg = df.mean(numeric_only=True); avg.name = 'AVG'
        df = pd.concat([df, pd.DataFrame([avg])])
        if verbose:
            print("\n====== AVERAGE ======")
            print(df)
        return df
