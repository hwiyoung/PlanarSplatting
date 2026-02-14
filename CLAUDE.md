# Semantic PlanarSplatting

## 프로젝트 개요
PlanarSplatting(CVPR 2025)을 확장하여 건물 구성요소 의미론(roof/wall/ground)을 추가하는 박사 연구.
핵심 기여: L_mutual 손실 함수 — 기하학(법선)과 의미론(분류)이 서로를 개선하는 양방향 상호 보강.

## 개발 환경
- 원격 GPU 서버에 ssh 접속
- Docker 컨테이너 내에서 개발 (Dockerfile + docker-compose)
- 코드와 데이터는 호스트에서 볼륨 마운트 (-v)
- **모든 python/pip 명령은 컨테이너 내부에서 실행**
- 새 패키지 설치 시 Dockerfile에도 추가할 것 (컨테이너 재빌드 시 유지)
- TensorBoard는 컨테이너 포트 → 서버 포트 → 로컬 포트로 이중 포워딩
- 원격 서버이므로 GUI 불가. 3D 시각화는 PLY export → scp → 로컬 CloudCompare/MeshLab

### 디렉토리 매핑 (호스트 ↔ 컨테이너)
```
호스트 (원격 서버)                              컨테이너 내부
─────────────────                              ──────────────
./  (프로젝트 루트)                         →  /workspace/PlanarSplatting/        (코드+전체)
./data/                                    →  /workspace/PlanarSplatting/data/   (입력 데이터)
$PLANARSPLAT_OUTPUT_DIR (기본: ./planarSplat_ExpRes)
                                           →  /workspace/PlanarSplatting/planarSplat_ExpRes/  (결과)
$PLANARSPLAT_INPUT_DIR  (기본: ./user_inputs)
                                           →  /workspace/PlanarSplatting/user_inputs/         (사용자 입력)
$PLANARSPLAT_HF_CACHE   (기본: ./.cache/huggingface)
                                           →  /root/.cache/huggingface/                       (모델 캐시)
```

### 자주 쓰는 명령
```bash
# 서버 접속
ssh user@gpu-server

# 컨테이너 시작 (planarsplat 서비스: TensorBoard+SyncViewer 자동 실행)
cd ~/PlanarSplatting && docker compose up -d planarsplat

# 인터랙티브 셸 진입 (shell 서비스)
docker compose run --rm shell

# 또는 기존 planarsplat 컨테이너에 접속
docker compose exec planarsplat bash

# 컨테이너 내부에서 COLMAP 데이터로 학습
python run_demo_colmap.py -d data/colmap_output -o planarSplat_ExpRes/phase0

# 컨테이너 내부에서 이미지로 학습 (VGGT 사용)
python run_demo.py -d data/images -o planarSplat_ExpRes/phase0

# TensorBoard (planarsplat 서비스가 자동 실행, 포트 6006)
# 또는 수동: tensorboard --logdir planarSplat_ExpRes --port 6006 --bind_all

# TensorBoard 접속 (로컬 PC에서)
ssh -L 6006:localhost:6006 user@gpu-server
# → http://localhost:6006

# Sync Viewer 접속 (포트 18080)
ssh -L 18080:localhost:18080 user@gpu-server
# → http://localhost:18080

# PLY export 후 로컬로 복사
# (plane_plots/ 폴더에 iter마다 PLY 자동 저장됨)
# 로컬: scp user@gpu-server:~/PlanarSplatting/planarSplat_ExpRes/demo/exp_*/최신/plane_plots/*.ply ./
```

## 아키텍처
- 각 평면 프리미티브 P_i: c_i(중심), r_i(반경), R_i(회전→법선), color, opacity, **f_i(K=3 의미론적 특징)**
- Θ_geo = {c_i, r_i, R_i, color, opacity}, Θ_sem = {f_i}
- L_total = L_photo + L_depth + L_normal + L_geo + λ_s·L_sem + λ_m·L_mutual

## L_mutual 수식
L_mutual = Σ_i [ p_wall(i)·(n_i·e_z)² + p_roof(i)·(1-(n_i·e_z)²) + p_ground(i)·(1-(n_i·e_z)²) ]
- p_c(i) = softmax(f_i)[c], n_i = R_i에서 유도된 법선, e_z = [0,0,1]
- wall: 법선이 수평(e_z 직교)이면 0 / roof,ground: 법선이 수직(e_z 평행)이면 0
- Warmup: iter < total/3이면 λ_m=0, 이후 선형 증가

## 코드 구조 (실제 프로젝트 기준)
```
PlanarSplatting/                         # 컨테이너: /workspace/PlanarSplatting/
├── run_demo.py                          # VGGT 기반 학습 엔트리포인트
├── run_demo_colmap.py                   # COLMAP 기반 학습 엔트리포인트 ★
├── run_gradio_demo.py                   # Gradio 웹 데모
├── Dockerfile                           # nvidia/cuda:12.1.1 + PyTorch 2.3.1
├── docker-compose.yml                   # planarsplat(학습) + shell(인터랙티브) 서비스
│
├── planarsplat/                         # ★ 핵심 패키지
│   ├── run/
│   │   ├── trainer.py                   # 메인 학습 루프 (PlanarSplatTrainRunner)
│   │   │   # L348: for iter 루프 | L371: forward | L387-388: loss | L394: backward
│   │   │   # L105-131: TB scalar 로깅 | L438-466: TB image/mesh 로깅
│   │   │   # L351-359: density control 호출
│   │   │   # ← warmup, ablation 플래그, L_sem/L_mutual 통합 위치
│   │   ├── net_wrapper.py               # 옵티마이저(Adam, 6 param groups), density control
│   │   │   # L38-48: optimizer 등록 | L90-107: prune_core | L178-211: split
│   │   │   # L154-176: densification_postfix | L299-316: gradient 누적
│   │   │   # ← f_i optimizer 등록, split/prune 시 f_i 처리 추가 위치
│   │   └── runner.py                    # 추론/데모용 러너
│   │
│   ├── net/
│   │   └── net_planarSplatting.py        # 프리미티브 파라미터 정의 + 렌더링 호출
│   │       # L60-65: nn.Parameter 6개 (_plane_center[N,3], _plane_radii_xy_{p,n}[N,2],
│   │       #         _plane_rot_q_normal_wxy[N,3], _plane_rot_q_xyAxis_{w,z}[N,1])
│   │       # L347-376: CUDA rasterizer 호출 → allmap[7ch] 반환
│   │       # ※ color/opacity는 learnable이 아님 (rand/ones로 생성)
│   │       # ← _plane_semantic_feat[N,3] 추가 + semantic 렌더링 추가 위치
│   │
│   ├── utils/
│   │   ├── loss_util.py                 # 손실 함수 2개만 존재
│   │   │   # L4-14:  metric_depth_loss(pred, gt, mask, max_depth=4.0) → L1
│   │   │   # L16-21: normal_loss(pred, gt, mask) → (L1, cosine)
│   │   │   # ← L_sem (CrossEntropy), L_mutual (mutual_loss) 추가 위치
│   │   ├── model_util.py                # split 마스크 계산, 프리미티브 분할 로직
│   │   ├── merge_util.py                # 프리미티브 병합 (후처리)
│   │   ├── mesh_util.py                 # 메시 변환 유틸
│   │   ├── plot_util.py                 # 시각화 유틸 (렌더링 비교 이미지 등)
│   │   ├── graphics_utils.py            # 카메라/투영 변환
│   │   ├── trainer_util.py              # 학습 보조 (decay schedule 등)
│   │   ├── align.py                     # 정렬 유틸
│   │   └── misc_util.py
│   │
│   ├── data_loader/
│   │   ├── scene_dataset.py             # 학습 데이터셋 (ScanNet 형식: image_high/, pose_unnormalized/)
│   │   └── scene_dataset_demo.py        # 데모 데이터셋 (data dict 직접 수신)
│   │
│   ├── data_process/                    # 데이터 전처리
│   │   ├── scannetpp/colmap_io.py       # COLMAP I/O 참고용
│   │   └── scannetv2/
│   │
│   ├── monocues/                        # 단안 깊이/법선 추정 (Metric3D v2, vitL)
│   │   ├── predictor.py
│   │   └── metric3d.py
│   │
│   └── confs/                           # 학습 설정 (.conf 형식, pyhocon)
│       ├── base_conf_planarSplatCuda.conf   # 기본 설정 ★
│       │   # max_total_iters=5000, init_plane_num=2000, log_freq=50
│       │   # weight_mono_normal=5.0, weight_mono_depth=2.0
│       │   # lr_{radii,center,rot_normal,rot_xy}=0.001
│       │   # use_tensorboard=True, tb_log_mesh=True
│       ├── scannetpp_train.conf
│       └── scannetv2_train.conf
│       # ← ablation_{none,full,sem2geo,geo2sem}.conf 추가 위치
│
├── submodules/                          # 외부 의존 모듈 (git submodule)
│   ├── diff-rect-rasterization/         # ★ CUDA 평면 래스터라이저
│   │   ├── cuda_rasterizer/
│   │   │   ├── forward.cu               # alpha 계산: sigmoid(scale - |pos|) * lambda
│   │   │   │   # L381-386: awx/awy sigmoid → G = min(awx,awy)
│   │   │   │   # L398: alpha = min(1, G)
│   │   │   │   # L418: w = alpha * T (blending weight)
│   │   │   │   # ※ 개별 프리미티브 w_i는 외부 미노출 → semantic용 별도 설계 필요
│   │   │   ├── auxiliary.h              # allmap 채널: 0=depth, 1=alpha, 2-4=normal, 5=mid, 6=dist
│   │   │   └── rasterizer_impl.cu
│   │   └── diff_rect_rasterization/     # Python 바인딩
│   ├── Metric3D/                        # 단안 깊이/법선 추정 모델
│   ├── vggt/                            # VGGT (SfM 대안, 정합률 문제 있음)
│   └── quaternion-utils/                # 쿼터니언 CUDA 연산
│
├── tools/                               # 모니터링/시각화 도구
│   ├── run_monitor.sh                   # docker-compose 엔트리 커맨드 (TB+SyncViewer 자동 실행)
│   ├── run_tensorboard_latest.py        # 최신 실험의 TensorBoard 실행
│   ├── run_sync_dual_viewer.py          # 동기화 3D 듀얼 뷰어 (포트 18080)
│   └── plot_training_log.py             # 학습 로그 플롯
│
├── utils_demo/                          # 데모 파이프라인 유틸
│   ├── run_vggt.py                      # VGGT 실행 (dense depth + PCD)
│   ├── run_metric3d.py                  # Metric3D 실행 (mono depth+normal)
│   ├── run_planarSplatting.py           # PlanarSplatting 학습 호출
│   ├── read_write_model.py              # COLMAP 바이너리 I/O (cameras.bin, images.bin)
│   ├── misc.py
│   └── demo.conf                        # 데모용 설정 override
│
├── planarSplat_ExpRes/                  # 실험 결과 출력 (볼륨 마운트)
│   ├── demo/exp_example/날짜/
│   │   ├── checkpoints/Parameters/      # 5000.pth, latest.pth
│   │   ├── tensorboard/                 # TF events (TB 로그)
│   │   ├── plane_plots/                 # PLY+JPG (iter별 시각화)
│   │   └── train.log
│   └── sync_viewer_live/                # 실시간 3D 비교 뷰어 (HTML)
│
├── scripts/                             # ★ 새로 작성할 스크립트 (현재 비어있음)
│   ├── (colmap_to_ps.py)                # [Phase 0] COLMAP → PS 입력 변환
│   ├── (generate_segmentation.py)       # [Phase 2-A] Grounded SAM 2
│   ├── (visualize_primitives.py)        # [Phase 0-Setup] PLY export (--color_by normal/rgb/class)
│   ├── (evaluate.py)                    # [Phase 0-Setup] PSNR, mIoU → JSON
│   ├── (compare_ablation.py)            # [Phase 3-B] Ablation 비교 표
│   └── (export_citygml.py)              # [Phase 4] CityGML LOD2 변환
│
├── data/                                # 입력 데이터 (볼륨 마운트, 현재 비어있음)
└── user_inputs/                         # 사용자 입력 이미지 (볼륨 마운트)
```

### 핵심 파일 매핑 (CLAUDE.md 용어 → 실제 경로)
| 용어 | 실제 파일 | 핵심 라인 |
|------|----------|----------|
| train.py (학습 루프) | `planarsplat/run/trainer.py` | L348 메인루프, L387-391 loss |
| model (프리미티브) | `planarsplat/net/net_planarSplatting.py` | L60-65 파라미터, L347-376 렌더링 |
| renderer (CUDA) | `submodules/diff-rect-rasterization/cuda_rasterizer/forward.cu` | L381-418 alpha계산 |
| loss (손실함수) | `planarsplat/utils/loss_util.py` | L4-14 depth, L16-21 normal |
| optimizer/density | `planarsplat/run/net_wrapper.py` | L38-48 옵티마이저, L90-211 split/prune |
| configs | `planarsplat/confs/*.conf` | pyhocon 형식 |
| 실행 (COLMAP) | `run_demo_colmap.py` | COLMAP→Metric3D→학습 |
| 실행 (VGGT) | `run_demo.py` | VGGT→Metric3D→학습 |

### Docker 환경 현황
| 항목 | 값 |
|------|-----|
| Base image | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` |
| PyTorch | `2.3.1` (cu121) |
| CUDA Archs | `8.0; 8.6; 8.9; 9.0` |
| 빌드된 submodules | diff-rect-rasterization, quaternion-utils, vggt, pytorch3d |
| 서비스 | `planarsplat` (학습, TB:6006, Viewer:18080) + `shell` (인터랙티브, Gradio:7860) |
| SHM | 16GB |
| 엔트리 커맨드 | `bash tools/run_monitor.sh` (TB+SyncViewer 자동 시작) |

### TensorBoard 연동 현황 (이미 구현됨)
`planarsplat/run/trainer.py`에 구현된 항목:
- **Scalars** (매 50 iter): `loss/depth`, `loss/normal_l1`, `loss/normal_cos`, `loss/plane`, `loss/total`, `model/plane_count`, `trend/*`
- **Images** (매 plot_freq iter): `render/vis_compare` (렌더링 vs GT)
- **Meshes** (매 plot_freq iter): `mesh/prim`, `mesh/normal`, `input/camera_frustums`, `input/mono_mesh`
- **Text**: 학습 상태 요약
- **로그 경로**: `{expdir}/tensorboard/`
- **추가 필요**: L_sem, L_mutual, gradient norm, 클래스별 프리미티브 수, Semantic map vs GT

### Semantic 렌더링 구현 시 핵심 제약
- CUDA rasterizer 내부에서 `w_i = alpha_i * T_i` (개별 프리미티브별 weight)가 계산되지만 **외부로 노출되지 않음**
- `allmap[1]` = 누적 alpha (1 - T_final) → 이것은 **합산된 결과**이므로 개별 w_i를 복원할 수 없음
- **해결 옵션**: (a) CUDA 수정하여 semantic feature도 alpha-blend, (b) PyTorch에서 2-pass 렌더링, (c) per-primitive 기여도를 별도 채널로 출력

## 중요 규칙
- **모든 명령은 컨테이너 내부에서.** pip install도 Dockerfile에 반영.
- f_i는 adaptive density control (split/clone/prune) 시 반드시 함께 처리
- L_mutual에서 detach 금지 (양방향 gradient 필수). ablation 시에만 선택적 detach
- semantic 렌더링은 PyTorch 레벨 우선 (CUDA 수정 후순위)
- 기존 기능 보존: --플래그로 새 기능 on/off

---

## Phase 정의

### Phase 0-Setup: 모니터링 + Docker 환경 확인

**목표:** Docker 환경을 확인/보강하고, PLY export·평가 스크립트를 구축한다.

**수행 작업:**
1. Docker 환경 확인:
   - ★ 이미 완료: GPU 설정 (deploy.resources), 포트 6006/18080 매핑, PyTorch 2.3.1+cu121
   - Dockerfile에 open3d (headless 모드), scipy, lxml 설치 여부 확인. 없으면 추가.
   - 컨테이너 재빌드 후 GPU 접근 확인 (python -c "import torch; print(torch.cuda.is_available())")
2. TensorBoard 로깅:
   - ★ 이미 구현됨 (`planarsplat/run/trainer.py` L105-131, L438-466)
   - Scalars: loss/depth, normal, total, plane_count 등 (매 50 iter)
   - Images: render/vis_compare (매 plot_freq iter)
   - Meshes: mesh/prim, mesh/normal
   - 추가 필요 없음 (Phase 0-Setup 요구사항 충족)
3. scripts/visualize_primitives.py: 체크포인트 → PLY export (--color_by normal/rgb, --export_ply)
   - 참고: 기존에 `plane_plots/` 폴더에 iter별 PLY 자동 저장됨
4. scripts/evaluate.py: 체크포인트 → PSNR, Depth MAE → JSON (--compare_with)

**검증:** 컨테이너 내에서 기존 체크포인트(`planarSplat_ExpRes/demo/exp_example/`)로 스크립트 작동 확인. TensorBoard에 ssh 터널링으로 접속 가능 확인.

### Phase 0: SfM/MVS 입력 확보

**목표:** VGGT 10% 정합 문제 해결, 공통 3D 입력 확보.

**입력:** data/images/ (성수동 드론 180장)

**수행 작업:**
1. COLMAP 실행 (컨테이너 내 COLMAP 없음 → 호스트에서 실행 후 결과를 data/colmap_output/에 복사)
   - 필요 출력: images/, sparse/cameras.bin, sparse/images.bin
2. `run_demo_colmap.py` 실행 (COLMAP 경로 전달):
   - 내부적으로: COLMAP 바이너리 읽기(`utils_demo/read_write_model.py`) → Metric3D mono depth/normal → 스케일 정렬 → 학습
   - ※ 별도 scripts/colmap_to_ps.py 불필요 — run_demo_colmap.py가 변환+학습을 일괄 수행
3. 5000 iter 학습 (init_plane_num=3000, base_conf_planarSplatCuda.conf 기반)
- COLMAP 실패 시: Metashape는 호스트/로컬에서 실행 → COLMAP 형식으로 export → 동일 파이프라인

**기대 출력:** data/colmap_output/, planarSplat_ExpRes/phase0/날짜/checkpoints/ (5000.pth)

**Go/No-Go:**
| 지표 | Go | Retry | Switch |
|------|-----|-------|--------|
| 정합 이미지 | ≥ 100장 | 50~99 → 매칭 조정 | < 50 → Metashape |
| 건물 형태 | 식별 가능 | 뭉개짐 → 초기화 조정 | 완전 실패 → 입력 변환 디버깅 |
| PSNR | ≥ 20dB | 18~20 → iter 증가 | < 18 → 입력 문제 |

**확인:** TensorBoard (ssh 터널링), PLY export → scp → CloudCompare, evaluate.py

### Phase 1: MVS Depth Supervision 교체

**목표:** depth supervision을 MVS 깊이로 교체하여 기하 정확도 개선. 기하가 정확해야 Phase 2의 의미론 학습에 올바른 감독 신호 제공.

**입력:** Phase 0 출력

**수행 작업:**
1. `planarsplat/utils/loss_util.py`의 `metric_depth_loss` 수정: MVS depth 지원
   - 현재: L1 loss with max_depth=4.0 masking (이미 절대 L1)
   - 추가: MVS 신뢰도 기반 마스킹
2. `planarsplat/utils/loss_util.py`의 `normal_loss` 수정: MVS depth 유도 법선 지원
3. 데이터 로딩 경로 수정 (run_demo_colmap.py 또는 scene_dataset_demo.py에서 MVS depth 전달)
4. `planarsplat/confs/`에 depth_source = mvs/mono 설정 추가
5. 5000 iter 학습

**기대 출력:** planarSplat_ExpRes/phase1/날짜/checkpoints/5000.pth, evaluation.json

**Go/No-Go:**
| 지표 | Go | Retry |
|------|-----|-------|
| 손실 수렴 | 모두 우하향 | 발산 → λ 조정 |
| Depth MAE | Phase 0 대비 개선/동등 | 악화 → 마스킹 threshold 조정 |
| Normal 시각화 | 벽 수직, 지붕 수평 | 무작위 → normal 유도 코드 확인 |

**확인:** TensorBoard, PLY export (color_by normal), evaluate.py --compare_with phase0

### Phase 2-A: 2D Segmentation 생성

**목표:** Grounded SAM 2로 roof/wall/ground segmentation map 생성.

**입력:** data/images/ (정합된 이미지 전체)

**수행 작업:**
1. Grounded SAM 2 설치 (Dockerfile에 추가, 또는 별도 컨테이너)
2. scripts/generate_segmentation.py 작성
3. "building roof"→1, "building wall"/"facade"→2, "ground"/"road"→3, 나머지→0

**기대 출력:** data/seg_maps/*.png, data/seg_vis/*.png

**Go/No-Go:** 시각적 80%+ → Go / 50~80% → 프롬프트 조정

**확인:** seg_vis/ → scp → 로컬에서 확인

### Phase 2-B: 의미론적 헤드 구현

**목표:** f_i(K=3) 추가, semantic 렌더링, L_sem 구현.

**수행 작업:**
1. `planarsplat/net/net_planarSplatting.py` L60-65 부근에 `_plane_semantic_feat`[N,3] nn.Parameter 추가
2. `planarsplat/run/net_wrapper.py`에서:
   - L40-48: 옵티마이저에 `_plane_semantic_feat` 등록 (별도 lr)
   - L154-176 (`densification_postfix`): split 시 f_i 복사 로직 추가
   - L90-107 (`prune_core`): prune 시 f_i 제거 로직 추가
3. Semantic 렌더링 (핵심 설계 필요):
   - ★ CUDA 내부의 개별 w_i(=alpha*T)는 외부 미노출
   - 옵션 A: CUDA 수정 → semantic feature도 alpha-blend (allmap에 3채널 추가)
   - 옵션 B: PyTorch 2-pass → 느리지만 CUDA 수정 불필요
   - 옵션 C: CUDA rasterizer의 `colors_precomp`에 softmax(f_i) 전달 → rgb 대신 semantic 렌더링
4. `planarsplat/utils/loss_util.py`에 `semantic_loss()` 추가: CrossEntropyLoss(ignore_index=0)
5. `planarsplat/run/trainer.py`에 TensorBoard 로깅 추가:
   - Scalar: `loss/semantic`, `model/class_{roof,wall,ground}_count`
   - Image: `render/semantic_vs_gt`
6. scripts/visualize_primitives.py에 --color_by class 추가
7. scripts/evaluate.py에 --metrics semantic_miou 추가
8. `planarsplat/confs/`에 enable_semantic, lambda_sem = 0.1 설정 추가

**검증:** torch.autograd.grad(L_sem, `_plane_semantic_feat`) non-zero

### Phase 2-C: L_sem 독립 학습

**목표:** L_mutual 없이 baseline 확보.

**수행 작업:**
- `planarsplat/run/trainer.py` L384-391에서 loss_final에 λ_s·L_sem 추가 (λ_m=0)
- 현재 loss: `loss_final = (depth*2.0 + (normal_l1+cos)*5.0) * decay`
- 변경: `loss_final += lambda_sem * L_sem`
- 5000 iter 학습

**기대 출력:** planarSplat_ExpRes/phase2/날짜/checkpoints/5000.pth, evaluation.json

**Go/No-Go:**
| 지표 | Go | Retry |
|------|-----|-------|
| mIoU | ≥ 0.50 | 0.30~0.50 → λ_s 조정 |
| PSNR | Phase 1 대비 ≤ 5% 악화 | > 5% → λ_s 감소 |

**확인:** TensorBoard (L_sem, 클래스 분포, Semantic map), PLY (color_by class) → scp, evaluate.py

### Phase 3-A: L_mutual 구현

**목표:** L_mutual + warmup + ablation 플래그.

**수행 작업:**
1. `planarsplat/utils/loss_util.py`에 `mutual_loss()` 함수 구현 (수식은 상단 참조, detach 금지)
   - 입력: `_plane_semantic_feat` (f_i), `_plane_rot_q_normal_wxy` (법선 유도용)
   - 법선 추출: `net_planarSplatting.py`의 `get_plane_geometry()` 참조
2. `planarsplat/confs/`에 설정 추가: mutual_warmup_ratio=0.33, mutual_mode=full/sem2geo/geo2sem/none, lambda_mutual=0.05
3. `planarsplat/run/trainer.py`에 TensorBoard 로깅 추가:
   - Scalar: `loss/mutual`, `grad/semantic_norm`, `grad/normal_norm`
4. Gradient check: autograd.grad(L_mutual, `_plane_semantic_feat`), autograd.grad(L_mutual, `_plane_rot_q_normal_wxy`)

**검증:** ∂L_mutual/∂f, ∂L_mutual/∂n 모두 non-zero (> 1e-8)

### Phase 3-B: Ablation 4조건 학습

**목표:** 핵심 ablation 결과.

**수행 작업:**
```bash
# 컨테이너 내부에서 순차 실행 (GPU 1개)
# (a) none: Phase 2-C 결과 재사용
# (b) — planarsplat/confs/ablation_full.conf
python run_demo_colmap.py -d data/colmap_output -o planarSplat_ExpRes/phase3_full --conf_path planarsplat/confs/ablation_full.conf
# (c) — planarsplat/confs/ablation_sem2geo.conf
python run_demo_colmap.py -d data/colmap_output -o planarSplat_ExpRes/phase3_sem2geo --conf_path planarsplat/confs/ablation_sem2geo.conf
# (d) — planarsplat/confs/ablation_geo2sem.conf
python run_demo_colmap.py -d data/colmap_output -o planarSplat_ExpRes/phase3_geo2sem --conf_path planarsplat/confs/ablation_geo2sem.conf
```
하나 돌리는 동안 다른 터미널(shell 서비스)에서 Phase 4 코드 작성 가능.

**기대 출력:** planarSplat_ExpRes/phase3_{none,full,sem2geo,geo2sem}/날짜/checkpoints/, ablation_table.csv

**Go/No-Go:** Full mutual이 ≥ 2개 지표에서 개선 → 기여 확인

**확인:**
```bash
# 4조건 TensorBoard 동시 비교 (로그 경로: {expdir}/tensorboard/)
tensorboard --logdir none:planarSplat_ExpRes/phase3_none,full:planarSplat_ExpRes/phase3_full,sem2geo:planarSplat_ExpRes/phase3_sem2geo,geo2sem:planarSplat_ExpRes/phase3_geo2sem --port 6006 --bind_all
# 또는: python tools/run_tensorboard_latest.py
```
compare_ablation.py, PLY 4개 → scp → 비교

### Phase 4: CityGML 변환 + 검증

**목표:** 프리미티브 → CityGML LOD2, val3dity 검증. Phase 3과 병렬 가능.

**수행 작업:**
1. scripts/export_citygml.py (분류 → 병합 → 경계 다각형 → CityGML XML)
2. val3dity 검증 (Dockerfile에 설치 또는 pip)
3. OBJ 생성 (색칠)

**기대 출력:** planarSplat_ExpRes/phase4/building.gml, building_colored.obj, val3dity_report.json

**확인:** OBJ/GML → scp → 로컬 MeshLab/QGIS