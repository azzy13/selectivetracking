# Spool worker — selective tracking without ROS

`Dockerfile.spool` runs the same pipeline as the ROS image
(`GroundingDINO → ByteTrack → scene-graph mission filter → colour re-ID`) but
takes its work from a directory on a mounted volume instead of a camera topic.
Anything that can write a file can drive it.

## Build and run

```bash
docker build -f docker/Dockerfile.spool -t selectivetracking-spool:latest .

docker run --rm --gpus all \
  -v /path/to/spool:/spool \
  -v /path/to/weights:/weights:ro \
  -e GDINO_PROMPT="red car." \
  selectivetracking-spool:latest
```

No staging script is needed. The ROS image requires `docker/build_ros2.sh` to
vendor a pinned `trinity_msgs` into the build context first; there are no
message definitions here, so a plain `docker build` is the whole story.

Or with compose: `docker compose -f docker/docker-compose.spool.yml up --build`.

## The spool contract

```
/spool/in       drop videos here (+ optional <stem>.json sidecar)
/spool/work     claimed, in flight
/spool/out      published results, one directory per job
/spool/failed   inputs that errored, beside a <name>.error.txt
```

All four are created at startup if absent. Recognised inputs are `.mp4`,
`.avi`, `.mov`, `.mkv`, `.m4v`, `.webm`; dotfiles are ignored.

A published job directory contains:

| File | What |
|---|---|
| `annotated.mp4` | Rendered video with track boxes and IDs |
| `tracks.txt` | MOT-format results — `frame,id,x,y,w,h,score,-1,-1,-1` |
| `result.json` | Prompt, arguments, timing, artifact list |
| `run.log` | Full stdout/stderr of the pipeline |
| `<original>.mp4` | The input, moved in so the directory is self-describing |

### Writing into the spool safely

**Write to a temp name and rename into `in/`.** A rename is atomic within a
filesystem, so the worker can never observe a half-copied file:

```bash
cp big.mp4 /path/to/spool/in/.staging-big.mp4
mv /path/to/spool/in/.staging-big.mp4 /path/to/spool/in/big.mp4
```

For writers that cannot do this, `SPOOL_SETTLE_SECONDS` (default 5) is the
fallback: a file is not claimed until its mtime has stopped moving for that long.
Raise it if a slow copy still gets picked up early.

### Reading results safely

Poll `/spool/out` for directories. Results are assembled under `out/.tmp-*` and
renamed into place, so a directory not starting with `.` is always complete — a
consumer never sees a partial job.

### Per-job overrides

A `<stem>.json` beside the video overrides the defaults for that job only:

```json
{
  "prompt": "red car behind the bus",
  "args": ["--box-threshold", "0.4", "--fp16"]
}
```

`args` is passed through verbatim to `demo/inference_w_worker.py` — see
`--help` there for the full set. Note it *replaces* `GDINO_EXTRA_ARGS` rather
than adding to it, so repeat `--fp16` if you want it. A malformed sidecar fails
the job rather than being ignored: a prompt that silently reverted to the
default would produce a plausible-looking answer to the wrong question.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SPOOL_DIR` | `/spool` | Root of the spool |
| `GDINO_WEIGHTS` | `/weights/groundingdino_swinb_cogcoor.pth` | Detector checkpoint |
| `GDINO_CONFIG` | `/app/groundingdino/config/GroundingDINO_SwinB_cfg.py` | Model config |
| `GDINO_PROMPT` | `red car.` | Default prompt |
| `GDINO_EXTRA_ARGS` | `--fp16` | Default extra pipeline flags |
| `SPOOL_POLL_SECONDS` | `2` | Idle poll interval |
| `SPOOL_SETTLE_SECONDS` | `5` | Quiet period before a file is claimed |
| `SPOOL_SCRATCH_DIR` | `/var/tmp/spool-work` | Frame extraction scratch |
| `SPOOL_ONCE` | unset | Drain the spool and exit, instead of watching |
| `SPOOL_REQUEUE_ORPHANS` | unset (off) | On startup, return `work/` leftovers to `in/` |

## Operational notes

**Weights are not in the image.** Mount them read-only. Without them every job
fails; the entrypoint warns at startup rather than waiting for the first job.

**GPU.** CUDA comes from conda-forge as a dependency of pytorch, so there is no
toolkit inside the image and only the host driver is needed — run with
`--gpus all`. The entrypoint reports what `torch.cuda.is_available()` says. On
CPU the pipeline is correct but very slow, and `--fp16` must be dropped
(`grid_sample` has no half-precision CPU kernel).

**One model load per job.** Each job is a fresh `demo/inference_w_worker.py`
subprocess, so the SwinB checkpoint and the BERT text encoder are loaded every
time — tens of seconds before frame one. This is a deliberate trade: `Worker`
takes `text_prompt` and calls `load_model` in `__init__`, so a single long-lived
worker cannot serve jobs with different prompts without restructuring
`eval/worker_simple.py`. For a spool of many short clips that overhead
dominates; batch them into longer videos, or lift the model out of `Worker.__init__`.

The `bert-base-uncased` encoder *is* baked into the image at `/opt/hf-cache`, so
that part costs no network at run time and the container works air-gapped.

**Running more than one worker on one spool** is safe for claiming — that is an
atomic rename, so two workers cannot take the same job. It is not safe for
`SPOOL_REQUEUE_ORPHANS`: a restarting worker cannot distinguish its own
abandoned job from another worker's in-flight one, and would yank it back to
`in/` to be processed twice. Leave that off when scaling out, and reconcile
`work/` by hand.

**Shutdown.** On `SIGTERM` the worker stops the running job and renames its
input back to `in/`, so nothing is lost — but it needs longer than Docker's
ten-second default to do it. The compose file sets `stop_grace_period: 60s`;
pass `--stop-timeout 60` with `docker run`. If the container is SIGKILLed
anyway, the input is left in `work/`; see `SPOOL_REQUEUE_ORPHANS`.

**File ownership.** The container runs as root by default, so results are
root-owned on the host. `docker run --user "$(id -u):$(id -g)"` works — the
environment lives at `/app/.pixi` with world-readable permissions, and the only
writes are to `/spool` and `SPOOL_SCRATCH_DIR`.
