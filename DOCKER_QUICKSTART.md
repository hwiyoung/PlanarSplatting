# PlanarSplatting Docker 빠른 시작

## 1) 빌드
```bash
docker compose build planarsplat
```

헤드리스 빌드 환경에서 아키텍처 자동 감지 문제로 CUDA 확장 모듈 빌드가 실패하면,
아래처럼 아키텍처 리스트를 명시하세요:
```bash
docker compose build planarsplat \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
```

## 2) 수동 쉘 진입 (선택)
```bash
docker compose run --rm --service-ports shell
```

입력 이미지가 현재 리포지터리 바깥에 있으면, 아래처럼 마운트 경로를 지정하세요:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images docker compose run --rm --service-ports shell
```
컨테이너 내부에서는 `/workspace/PlanarSplatting/user_inputs` 경로를 사용하면 됩니다.
`PLANARSPLAT_INPUT_DIR`는 `images`의 상위 폴더여도 되고, `images` 폴더 자체여도 됩니다.
`planarsplat` 서비스는 `/images` 유무를 보고 자동으로 데이터 경로를 잡습니다.
VGGT 가중치 다운로드 캐시는 `${PLANARSPLAT_HF_CACHE:-./.cache/huggingface}`로 유지됩니다.
결과 폴더는 `${PLANARSPLAT_OUTPUT_DIR:-./planarSplat_ExpRes}`에 로컬로 바로 저장됩니다.

## 3) 실행 (컨테이너 내부)
### A. Gradio 대화형 데모
```bash
python run_gradio_demo.py
```
접속: `http://localhost:7860`

### B. 이미지 폴더 대상 CLI 데모
```bash
python run_demo.py --data_path /workspace/PlanarSplatting/data/my_scene/images
```

VGGT 단계에서 CUDA OOM이 발생하면 `frame_step`을 키우세요:
```bash
python run_demo.py --data_path /workspace/PlanarSplatting/data/my_scene/images --frame_step 3
```

외부 마운트 입력 예시:
```bash
python run_demo.py --data_path /workspace/PlanarSplatting/user_inputs/images
```

### C. COLMAP 데이터 CLI 데모
```bash
python run_demo_colmap.py -d /workspace/PlanarSplatting/data/my_colmap_scene
```

## 3-1) One-Command Planarsplat (학습 + TensorBoard + Sync Viewer)
학습, TensorBoard, sync viewer를 한 번에 실행합니다:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
docker compose up planarsplat
```
(`planarsplat` 기본 실행은 이미지 수가 많으면 자동으로 `frame_step`을 올려서 VGGT OOM을 완화합니다.)
(`frame_step` 계산과 실제 VGGT 입력 모두 이미지 확장자 기준(`jpg/jpeg/png/bmp/tif/tiff/webp`)으로 동작합니다.)
(`planarsplat`는 학습이 끝나도 TensorBoard를 계속 유지해 결과를 이어서 확인할 수 있습니다.)
(`planarsplat` 기본 실행은 `plot_freq=200`으로 TensorBoard `Images` 탭에 렌더링 비교 이미지를 주기적으로 기록합니다.)
(`sync viewer`는 결과 mesh가 생긴 뒤 자동으로 열립니다.)
(`live refresh`는 학습 성능 보호를 위해 기본 비활성화되어 있습니다.)
(`sync viewer`는 최신 run 여러 개를 기본 4개까지 동시에 로드해 체크박스로 on/off 할 수 있습니다.)
(`Runs`는 기본적으로 최신 1개만 활성화되어 시작됩니다. 겹침/파편처럼 보이면 다른 run 체크를 끄세요.)
(`PLANARSPLAT_SYNC_VIEWER_COMPARE_RUNS=1`이면 run 목록도 1개만 로드합니다.)
(`data.pth` 기반 포인트가 너무 희박하면 `input_pointcloud.ply`로 자동 폴백해 밀도를 높입니다.)
(`depth_conf`가 높을수록 포인트가 줄어듭니다. 밀도가 낮으면 `PLANARSPLAT_DEPTH_CONF`를 낮춰 재실행하세요.)

실행 후 접속:
- `http://localhost:6006`
- `http://localhost:${PLANARSPLAT_SYNC_VIEWER_PORT:-18080}/sync_dual_viewer.html`

sync viewer 라이브 업데이트 주기 설정 (선택, CPU/I/O 부하 증가 가능):
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_SYNC_VIEWER_REFRESH_SEC=10 \
PLANARSPLAT_SYNC_VIEWER_POLL_SEC=3 \
PLANARSPLAT_SYNC_VIEWER_COMPARE_RUNS=4 \
PLANARSPLAT_SYNC_VIEWER_POINT_STRIDE=8 \
PLANARSPLAT_SYNC_VIEWER_MAX_POINTS=350000 \
PLANARSPLAT_SYNC_VIEWER_MAX_FACES=500000 \
PLANARSPLAT_SYNC_VIEWER_PREFER_SAVED_PCD=0 \
PLANARSPLAT_DEPTH_CONF=1.0 \
docker compose up planarsplat
```

호스트 `6006` 포트가 이미 사용 중이면, 다른 포트로 매핑하세요:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_TB_PORT=16006 docker compose up planarsplat
```
접속: `http://localhost:16006`

자동 메모리 가드 기준 변경:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_MAX_VGGT_IMAGES=16 docker compose up planarsplat
```

로컬 출력 디렉터리 변경:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_OUTPUT_DIR=/absolute/path/to/planarsplat_outputs \
docker compose up planarsplat
```

파일 수정 없이 학습 명령 변경:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_TRAIN_CMD='python run_demo.py --data_path /workspace/PlanarSplatting/user_inputs/images' \
docker compose up planarsplat
```
(`PLANARSPLAT_TRAIN_CMD`가 비어 있으면 기본 안전 모드(auto)로 실행됩니다.)
(`PLANARSPLAT_TRAIN_CMD`가 설정되면 custom 모드로 실행됩니다.)
(`CUDA_VISIBLE_DEVICES=0 python ...`처럼 선행 env assignment를 포함한 커맨드도 지원합니다.)
(`&&`, `|` 같은 shell 연산자가 포함된 커맨드는 호환을 위해 shell 실행 경로를 사용합니다.)

TensorBoard 이미지 업데이트 빈도 조정 (`작을수록 더 자주 기록`):
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_PLOT_FREQ=100 docker compose up planarsplat
```

동일 실험 run 누적 방지 (`실행 전 기존 exp_example 삭제`):
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_OVERWRITE_EXP=1 docker compose up planarsplat
```

재실행 시 VGGT/Metric3D 사전 계산 데이터(`data.pth`) 재사용:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_USE_PRECOMPUTED_DATA=1 docker compose up planarsplat
```

데이터셋 형식 학습 예시:
```bash
PLANARSPLAT_TRAIN_CMD='cd planarsplat && python run/runner.py --base_conf confs/base_conf_planarSplatCuda.conf --conf confs/scannetv2_train.conf --gpu 0 --scan_id scene0488_00' \
docker compose up planarsplat
```

## 4) 입력 데이터 형식
### `run_demo.py`
- `--data_path`는 아래 형식을 지원합니다:
  - 이미지 폴더 (`.png/.jpg/.jpeg`)
  - 비디오 파일 (`.mp4/.mov/...`)
- 이미지 폴더 입력 시, 지원 확장자(`jpg/jpeg/png/bmp/tif/tiff/webp`)만 사용됩니다.

### `run_demo_colmap.py`
- 폴더 구조:
  - `images/`
  - `sparse/cameras.bin`
  - `sparse/images.bin`
  - `sparse/points3D.bin`

### Dataset training (`planarsplat/run/runner.py`)
- scene 별 기대 구조:
  - `image_high/*.png`
  - `intrinsic/intrinsic_depth_high.txt`
  - `pose_unnormalized/*.txt`
- 현재 이미지 크기는 `480x640` 고정입니다.

## 5) 학습 결과 위치
출력 루트 (컨테이너):
- `/workspace/PlanarSplatting/planarSplat_ExpRes/`
출력 루트 (호스트):
- `${PLANARSPLAT_OUTPUT_DIR:-./planarSplat_ExpRes}`

run 별 주요 파일:
- `.../<expname>_<scan_id>/<timestamp>/train.log`
- `.../<expname>_<scan_id>/<timestamp>/plane_plots/`
- `.../<expname>_<scan_id>/<timestamp>/checkpoints/Parameters/latest.pth`

최종 메시:
- `.../<expname>_<scan_id>/<timestamp>/<scan_id>/<scan_id>_planar_mesh.ply`

## 6) 모니터링 (TensorBoard 권장)
TensorBoard 하나로 학습 경과를 확인할 수 있습니다.

최신 run 기준 TensorBoard 실행:
```bash
python tools/run_tensorboard_latest.py
```
접속: `http://localhost:6006`

한 대시보드에서 확인 가능한 항목:
- `Scalars`: loss + plane count + trend 지표
- `Images`: rendered-vs-mono 스냅샷
- `Meshes`: planar splatting mesh 스냅샷
- `Meshes`: 입력 기하 스냅샷 (`input/mono_mesh`, `input/vggt_pointcloud`, `input/camera_frustums`, `input/mono_mesh_with_cameras`)
- `Text`: 최신 상태 라인 (`iter`, losses, trend state)

## 6-1) Sync Dual Viewer (Non-TensorBoard)
좌/우 3D 뷰 카메라를 동기화해서 보려면 아래 뷰어를 사용하세요.
- 좌측: `camera + point cloud`
- 우측: `camera + mesh`
- 한쪽을 회전/팬/줌하면 다른쪽도 같이 움직입니다.
- 헤더에 `Aligned images: x / N`(정합 이미지 수 / 전체 이미지 수)가 표시됩니다.
- 헤더의 `Data Info`를 열면 현재 시각화 중인 `run_dir / image_dir / mesh / data.pth / pcd`와 포인트/메시 카운트를 확인할 수 있습니다.
- 헤더의 `Layers`에서 표시 대상을 직접 토글할 수 있습니다 (`Point Cloud (Left)`, `Final Mesh (Right)`, `Raw Mesh (Right)`, `Camera Frustums`, `Grid`, `Axes`, `Sync Camera Motion`).
- `Layers > Point Source`에서 좌측 포인트클라우드 소스를 즉시 전환할 수 있습니다 (`Auto`, `Reprojection`, `Saved PCD`).
- `Point Source` 데이터는 필요 시에만 별도 sidecar json을 로드하도록 경량화되어, 메인 viewer json 로딩 부담을 줄입니다.
- 카메라 frustum이 크면 `PLANARSPLAT_SYNC_VIEWER_FRUSTUM_RATIO` 값을 더 낮추세요 (기본값 `0.04`, 예: `0.02`).
- 헤더의 `Runs`에서 TensorBoard처럼 run 체크박스를 선택해 어떤 실험을 표시할지 고를 수 있습니다.

`docker compose up planarsplat` 실행 시 기본적으로 함께 켜집니다:
- `http://localhost:${PLANARSPLAT_SYNC_VIEWER_PORT:-18080}/sync_dual_viewer.html`

수동으로 다시 생성/실행하려면:
```bash
docker compose exec planarsplat \
python tools/run_sync_dual_viewer.py --compare_runs 4 --serve --host 0.0.0.0 --port 18080
```
접속:
- `http://localhost:18080/sync_dual_viewer.html`

특정 run을 직접 지정:
```bash
docker compose exec planarsplat \
python tools/run_sync_dual_viewer.py \
  --run_dir /workspace/PlanarSplatting/planarSplat_ExpRes/demo/exp_example/<timestamp> \
  --compare_runs 4 \
  --serve --port 18080
```

좌표 정합 팁:
- 기본값은 `data.pth`에서 point cloud를 다시 만들어 카메라/mesh 좌표계와 맞춥니다.
- `.ply`를 그대로 쓰고 싶으면 `--prefer_saved_pcd`를 추가하세요.

포트 변경:
```bash
PLANARSPLAT_INPUT_DIR=/absolute/path/to/my_images \
PLANARSPLAT_SYNC_VIEWER_PORT=18081 docker compose up planarsplat
```
접속:
- `http://localhost:18081/sync_dual_viewer.html`

### 선택 디버깅 명령
```bash
# latest run folder
RUN_DIR=$(ls -td planarSplat_ExpRes/demo/*/* | head -n 1)
echo "$RUN_DIR"

# log tail
tail -f "$RUN_DIR/train.log"

# see newest visualization image path
ls -t "$RUN_DIR/plane_plots"/*.jpg | head -n 5
```

`train.log`에서 학습 곡선 생성/업데이트:
```bash
python tools/plot_training_log.py --run_dir "$RUN_DIR"
```
출력:
- `$RUN_DIR/training_curves.png`
- `$RUN_DIR/training_metrics.csv`

20초마다 곡선 라이브 업데이트:
```bash
python tools/plot_training_log.py --run_dir "$RUN_DIR" --watch 20
```

특정 run dir 대상으로 TensorBoard 실행:
```bash
tensorboard --logdir "$RUN_DIR/tensorboard" --host 0.0.0.0 --port 6006
```
접속: `http://localhost:6006`

로깅 빈도 제어 항목:
- `train.log_freq`: scalar 로깅 간격
- `train.plot_freq`: rendered image / mesh 스냅샷 간격
- `train.tb_log_mesh`: mesh 로깅 on/off
- `train.tb_log_text`: 상태 텍스트 로깅 on/off
- `train.tb_trend_window`: trend 추정 윈도우 (iterations)

`run_demo.py` 누적 정책:
- 동일 설정/데이터라도 실행할 때마다 새 timestamp run이 생성됩니다 (기본 동작).
- `--overwrite_exp`를 사용하면 실행 전에 동일 실험 폴더를 삭제합니다.
- `--use_precomputed_data`를 사용하면 기존 `out_path/data.pth`를 재사용합니다.

## 7) 시각화 파일 열기
- `*.jpg`: 일반 이미지 뷰어/브라우저로 열기 (텍스트 편집기 X)
- `*.ply`: MeshLab, CloudCompare, Open3D viewer로 열기
- `training_curves.png`: 일반 이미지 뷰어/브라우저로 열기
- TensorBoard:
  - `Scalars`: loss/plane_count 추이
  - `Images`: rendered-vs-mono 비교 스냅샷
  - `Meshes`: 학습 중 planar splatting primitive mesh 스냅샷

## 8) 런타임 참고 (현재 동작)
- `run_vggt.py`는 `jpg/jpeg/png/bmp/tif/tiff/webp` 확장자 이미지 파일만 로드합니다.
- `depth_conf` 필터링 후 유효 depth가 없으면, 경고 로그와 함께 VGGT scale을 `1.0`으로 폴백합니다.
- `planarsplat` 서비스의 기본 auto 학습 경로는 공백 경로에서도 shell 문자열 파싱 문제 없이 실행됩니다.
- 커스텀 `PLANARSPLAT_TRAIN_CMD`는 선행 env assignment를 지원합니다 (예: `CUDA_VISIBLE_DEVICES=0 python ...`).
