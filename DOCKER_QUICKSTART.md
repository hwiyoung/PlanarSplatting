# PlanarSplatting Docker Quickstart

## 1) Build
```bash
docker compose build planarsplat
```

If CUDA extension build fails due to architecture detection in headless build,
set a fixed architecture list explicitly:
```bash
docker compose build planarsplat \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
```

## 2) Enter container
```bash
docker compose run --rm --service-ports planarsplat
```

If your input images are outside this repo, set one extra mount path:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images docker compose run --rm --service-ports planarsplat
```
Then use `/workspace/PlanarSplatting/user_inputs` inside the container.
`PLANARSPLAT_INPUT_DIR`는 `images`의 상위 폴더여도 되고, `images` 폴더 자체여도 됩니다.
`monitor`는 `/images` 유무를 보고 자동으로 데이터 경로를 잡습니다.
VGGT 가중치 다운로드 캐시는 `${PLANARSPLAT_HF_CACHE:-./.cache/huggingface}`로 유지됩니다.
결과 폴더는 `${PLANARSPLAT_OUTPUT_DIR:-./planarSplat_ExpRes}`에 로컬로 바로 저장됩니다.

## 3) Run (inside container)
### A. Interactive Gradio demo
```bash
python run_gradio_demo.py
```
Open: `http://localhost:7860`

### B. CLI demo on your images folder
```bash
python run_demo.py --data_path /workspace/PlanarSplatting/data/my_scene/images
```

If you hit CUDA OOM in VGGT, increase subsampling step:
```bash
python run_demo.py --data_path /workspace/PlanarSplatting/data/my_scene/images --frame_step 3
```

Example with external mounted input:
```bash
python run_demo.py --data_path /workspace/PlanarSplatting/user_inputs/images
```

### C. CLI demo on COLMAP export
```bash
python run_demo_colmap.py -d /workspace/PlanarSplatting/data/my_colmap_scene
```

## 3-1) One-Command Monitor (Train + TensorBoard)
Run training and TensorBoard together:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
docker compose up monitor
```
(`monitor`는 6006 포트를 사용합니다. `planarsplat` 기본 실행은 7860 포트만 사용합니다.)
(`monitor` 기본 실행은 이미지 수가 많으면 자동으로 `frame_step`을 올려서 VGGT OOM을 완화합니다.)
(`monitor`는 학습이 끝나도 TensorBoard를 계속 유지해 결과를 이어서 확인할 수 있습니다.)
(`monitor` 기본 실행은 `plot_freq=200`으로 TensorBoard `Images` 탭에 렌더링 비교 이미지를 주기적으로 기록합니다.)

Then open:
- `http://localhost:6006`
- `http://localhost:${PLANARSPLAT_SYNC_VIEWER_PORT:-18080}` (sync dual viewer port, when you run it)

If host `6006` is already in use, map another host port:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_TB_PORT=16006 docker compose up monitor
```
Open: `http://localhost:16006`

Change auto memory guard threshold:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_MAX_VGGT_IMAGES=16 docker compose up monitor
```

Change local output directory:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_OUTPUT_DIR=/absolute/path/to/planarsplat_outputs \
docker compose up monitor
```

Change training command without editing files:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_TRAIN_CMD='python run_demo.py --data_path /workspace/PlanarSplatting/user_inputs/images' \
docker compose up monitor
```

Tune TensorBoard image update frequency (smaller = more frequent snapshots):
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_PLOT_FREQ=100 docker compose up monitor
```

Avoid run accumulation for same experiment (delete old `exp_example` before run):
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_OVERWRITE_EXP=1 docker compose up monitor
```

Reuse precomputed VGGT/Metric3D data (`data.pth`) when rerunning:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_USE_PRECOMPUTED_DATA=1 docker compose up monitor
```

Example for dataset-style training:
```bash
PLANARSPLAT_TRAIN_CMD='cd planarsplat && python run/runner.py --base_conf confs/base_conf_planarSplatCuda.conf --conf confs/scannetv2_train.conf --gpu 0 --scan_id scene0488_00' \
docker compose up monitor
```

## 4) Input data format
### `run_demo.py`
- `--data_path` can be:
  - an images folder (`.png/.jpg/.jpeg`)
  - a video file (`.mp4/.mov/...`)

### `run_demo_colmap.py`
- folder layout:
  - `images/`
  - `sparse/cameras.bin`
  - `sparse/images.bin`
  - `sparse/points3D.bin`

### Dataset training (`planarsplat/run/runner.py`)
- expected per-scene structure:
  - `image_high/*.png`
  - `intrinsic/intrinsic_depth_high.txt`
  - `pose_unnormalized/*.txt`
- image size currently fixed to `480x640`.

## 5) Where to monitor training
Output root (container):
- `/workspace/PlanarSplatting/planarSplat_ExpRes/`
Output root (host):
- `${PLANARSPLAT_OUTPUT_DIR:-./planarSplat_ExpRes}`

For each run:
- `.../<expname>_<scan_id>/<timestamp>/train.log`
- `.../<expname>_<scan_id>/<timestamp>/plane_plots/`
- `.../<expname>_<scan_id>/<timestamp>/checkpoints/Parameters/latest.pth`

Final mesh:
- `.../<expname>_<scan_id>/<timestamp>/<scan_id>/<scan_id>_planar_mesh.ply`

## 6) Monitoring (TensorBoard-Only Recommended)
You can monitor training in one place with TensorBoard only.

Start TensorBoard for the latest run:
```bash
python tools/run_tensorboard_latest.py
```
Open: `http://localhost:6006`

What you get in one dashboard:
- `Scalars`: loss + plane count + trend signals
- `Images`: rendered-vs-mono snapshots
- `Meshes`: planar splatting mesh snapshots
- `Meshes`: input geometry snapshots (`input/mono_mesh`, `input/vggt_pointcloud`, `input/camera_frustums`, `input/mono_mesh_with_cameras`)
- `Text`: latest status line (`iter`, losses, trend state)

## 6-1) Sync Dual Viewer (Non-TensorBoard)
좌/우 3D 뷰 카메라를 동기화해서 보려면 아래 뷰어를 사용하세요.
- Left: `camera + point cloud`
- Right: `camera + mesh`
- 한쪽을 회전/팬/줌하면 다른쪽도 같이 움직입니다.

Inside container:
```bash
python tools/run_sync_dual_viewer.py --serve --port 18080
```
Open:
- `http://localhost:18080/sync_dual_viewer.html`

특정 run을 직접 지정:
```bash
python tools/run_sync_dual_viewer.py \
  --run_dir /workspace/PlanarSplatting/planarSplat_ExpRes/demo/exp_example/<timestamp> \
  --serve --port 18080
```

좌표 정합 팁:
- 기본값은 `data.pth`에서 point cloud를 다시 만들어 카메라/mesh 좌표계와 맞춥니다.
- `.ply`를 그대로 쓰고 싶으면 `--prefer_saved_pcd`를 추가하세요.

포트 변경:
```bash
PLANARSPLAT_SYNC_VIEWER_PORT=18081 docker compose run --rm --service-ports planarsplat
# container 안에서
python tools/run_sync_dual_viewer.py --serve --port 18080
```
Open:
- `http://localhost:18081/sync_dual_viewer.html`

### Optional debug commands
```bash
# latest run folder
RUN_DIR=$(ls -td planarSplat_ExpRes/expRes/*/* | head -n 1)
echo "$RUN_DIR"

# log tail
tail -f "$RUN_DIR/train.log"

# see newest visualization image path
ls -t "$RUN_DIR/plane_plots"/*.jpg | head -n 5
```

Create/update training curves from `train.log`:
```bash
python tools/plot_training_log.py --run_dir "$RUN_DIR"
```
Output:
- `$RUN_DIR/training_curves.png`
- `$RUN_DIR/training_metrics.csv`

Live-update the curve every 20s:
```bash
python tools/plot_training_log.py --run_dir "$RUN_DIR" --watch 20
```

Run TensorBoard on explicit run dir:
```bash
tensorboard --logdir "$RUN_DIR/tensorboard" --host 0.0.0.0 --port 6006
```
Open: `http://localhost:6006`

Logging frequency controls:
- `train.log_freq`: scalar logging interval
- `train.plot_freq`: rendered image / mesh snapshot interval
- `train.tb_log_mesh`: mesh logging on/off
- `train.tb_log_text`: status text logging on/off
- `train.tb_trend_window`: trend estimation window (iterations)

`run_demo.py` accumulation policy:
- same settings/data still create a new timestamp run each execution (default behavior)
- add `--overwrite_exp` to remove previous same experiment folder before starting
- add `--use_precomputed_data` to reuse existing `out_path/data.pth`

## 7) How to open visualization files
- `*.jpg`: open with any image viewer/browser (not a text editor).
- `*.ply`: open with MeshLab, CloudCompare, or Open3D viewer.
- `training_curves.png`: open with any image viewer/browser.
- TensorBoard:
  - `Scalars`: loss/plane_count trends in real-time
  - `Images`: rendered-vs-mono comparison snapshots
  - `Meshes`: planar splatting primitive mesh snapshots over training
