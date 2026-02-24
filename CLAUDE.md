# Semantic PlanarSplatting

## 프로젝트 개요
PlanarSplatting(CVPR 2025)을 확장하여 항공 드론 이미지에서 건물 구성요소 의미론(roof/wall/ground)을 추가하는 박사 연구. 핵심 기여는 L_mutual 손실 함수로, 기하학적 법선(R_i에서 유도)과 의미론적 분류(f_i)가 서로를 개선하는 양방향 상호 보강을 구현한다. 최종 출력은 CityGML LOD2 형식의 의미론적 3D 건물 모델.

## 상세 문서 (필요 시 참조)
- `docs/RESEARCH_CONTEXT.md` — 연구 배경, 전체 파라미터 구조, L_mutual 수식/gradient 분석, ablation 설계
- `docs/EXPERIMENT_PLAN.md` — Phase별 목표/입력/출력/Go-No-Go/확인 방법/프롬프트

## 개발 환경: Docker 기반
이 프로젝트의 모든 개발과 실행은 Docker 컨테이너 내부에서 수행한다.

- **Docker 이미지:** `planarsplat:cu121` (nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04, PyTorch 2.3.1)
- **서비스 (docker-compose.yml):**
  - `planarsplat` (컨테이너명: `planarsplat`) — 학습+TensorBoard+웹뷰어 통합. `tools/run_monitor.sh` 실행.
  - `shell` (컨테이너명: `planarsplat-shell`) — 대화형 셸 (Gradio 포트 7860)
- **볼륨 마운트:**
  - `./` → `/workspace/PlanarSplatting` (전체 소스)
  - `./data` → `/workspace/PlanarSplatting/data`
  - `${PLANARSPLAT_OUTPUT_DIR:-./planarSplat_ExpRes}` → `/workspace/PlanarSplatting/planarSplat_ExpRes`
  - `${PLANARSPLAT_INPUT_DIR:-./user_inputs}` → `/workspace/PlanarSplatting/user_inputs`
  - `${PLANARSPLAT_HF_CACHE:-./.cache/huggingface}` → `/root/.cache/huggingface`
- **포트 (docker-compose 기본값 → .env로 변경 가능):**
  - `${PLANARSPLAT_TB_PORT:-6006}:6006` — TensorBoard (현재 호스트: 16006 → `ssh -L 16006:localhost:16006 서버` → `http://localhost:16006`)
  - `${PLANARSPLAT_SYNC_VIEWER_PORT:-18080}:18080` — 3D Sync Dual Viewer (`http://localhost:18080/sync_dual_viewer.html`)
  - `7860:7860` — Gradio (shell 서비스)
- **GPU:** `gpus: all`, `shm_size: 16g`, `ipc: host`
- **모든 python/pip/학습 명령은 반드시 컨테이너 내부에서 실행**
- **패키지 설치:** 컨테이너 내 pip install + Dockerfile에도 추가 (재빌드 시 유지)
- **3D 시각화:** Sync Dual Viewer (Three.js 웹 뷰어, `tools/run_sync_dual_viewer.py`), 또는 PLY export → scp → 로컬 CloudCompare
- **데스크톱 GUI 앱(Open3D GUI, MeshLab 등)은 컨테이너에서 직접 실행 불가** — 웹 뷰어 또는 PLY export로 대체

## 코드 구조
```
PlanarSplatting/
├── run_demo.py                          # VGGT 기반 데모 엔트리
├── run_demo_colmap.py                   # COLMAP 기반 엔트리 (COLMAP→Metric3D→scale align→train)
├── run_gradio_demo.py                   # Gradio 웹 UI
├── Dockerfile                           # nvidia/cuda:12.1.1, PyTorch 2.3.1, CUDA submodule 빌드
├── docker-compose.yml                   # 서비스: planarsplat (학습), shell (대화형)
├── requirements.txt
│
├── planarsplat/                          # 핵심 라이브러리
│   ├── confs/                           # pyhocon 설정 (.conf)
│   │   └── base_conf_planarSplatCuda.conf  # 기본 config (train, dataset, plane_model)
│   ├── run/
│   │   ├── trainer.py                   # PlanarSplatTrainRunner (학습 루프 L348, TensorBoard)
│   │   ├── net_wrapper.py               # PlanarRecWrapper (Adam optimizer, split/prune/density)
│   │   └── runner.py
│   ├── net/
│   │   └── net_planarSplatting.py       # PlanarSplat_Network (7 learnable params: 6 geo + f_i, forward→rasterizer)
│   ├── data_loader/
│   │   ├── scene_dataset.py             # 일반 학습 데이터셋
│   │   └── scene_dataset_demo.py        # 데모 데이터셋 (ViewInfo, SceneDatasetDemo)
│   ├── monocues/
│   │   └── metric3d.py                  # Metric3D mono depth/normal 추론
│   └── utils/
│       ├── loss_util.py                 # 4개 손실: metric_depth_loss, normal_loss, semantic_loss (Phase 2-B), normal_consistency_loss (Phase 2-B)
│       ├── trainer_util.py              # plot_plane_img, save/resume checkpoint
│       ├── model_util.py                # quaternion ops, split mask logic
│       ├── mesh_util.py                 # TSDF mesh (refuse_mesh), depth rendering
│       ├── merge_util.py                # merge_plane (후처리 병합)
│       ├── plot_util.py                 # plot_rectangle_planes → Open3D mesh 생성
│       └── graphics_utils.py            # projection matrix, focal2fov
│
├── submodules/
│   ├── diff-rect-rasterization/         # CUDA 사각형 rasterizer (forward.cu L381-418: alpha, backward.cu: color grad 추가)
│   ├── quaternion-utils/                # CUDA quaternion 연산
│   ├── Metric3D/                        # mono depth/normal 추정 (hubconf.py)
│   └── vggt/                            # VGGT 포인트 클라우드 추정
│
├── tools/
│   ├── run_monitor.sh                   # Docker entrypoint (TensorBoard + 학습 + Sync Viewer)
│   ├── run_tensorboard_latest.py        # 최신 실험 TB 실행
│   ├── run_sync_dual_viewer.py          # Three.js 3D 웹 뷰어 서버 (포트 18080)
│   ├── plot_training_log.py
│   └── sync_viewer_assets/              # Three.js, OrbitControls, TrackballControls
│
├── utils_demo/
│   ├── run_planarSplatting.py           # 학습 실행 래퍼
│   ├── run_metric3d.py                  # Metric3D 실행
│   ├── run_vggt.py                      # VGGT 실행
│   ├── read_write_model.py              # COLMAP binary I/O
│   ├── demo.conf                        # 데모 전용 config override
│   └── misc.py
│
├── scripts/                             # 평가, 시각화, 데이터 변환 스크립트
│   ├── visualize_primitives.py          # 체크포인트 → PLY export (--color_by normal/class)
│   ├── evaluate.py                      # 체크포인트 → Depth MAE, Normal cos, Semantic mIoU → JSON
│   ├── render_views.py                  # 체크포인트 → RGB/Depth/Normal 이미지 렌더링
│   ├── colmap_to_ps.py                  # COLMAP 출력 → PlanarSplatting input_data.pth 변환
│   ├── generate_segmentation.py         # Grounded SAM 2 + MVS normal → seg_maps 생성
│   └── gradient_check_phase2b.py        # Phase 2-B gradient check (6개 자동 테스트)
│
├── planarSplat_ExpRes/                  # 실험 결과 (볼륨 마운트)
│   ├── demo/                            # 실험별 하위: exp_name/timestamp/
│   │   └── exp_*/timestamp/
│   │       ├── tensorboard/             # TensorBoard 이벤트
│   │       ├── checkpoints/Parameters/  # latest.pth, 5000.pth
│   │       ├── plane_plots/             # 시각화 이미지, PLY
│   │       └── input_data.pth           # 저장된 입력 데이터
│   └── sync_viewer_live/               # 웹 뷰어 HTML
│
└── data/                                # 입력 데이터 (볼륨 마운트)
```

### 핵심 데이터 흐름
1. **입력**: 이미지 → COLMAP SfM/MVS(depth, normal) → SceneDatasetDemo(ViewInfo 리스트 + seg_maps)
2. **학습**: PlanarSplatTrainRunner.train() → PlanarSplat_Network.forward() → CUDA rasterizer → rendered_features[4ch] + allmap[7ch]
3. **손실 (기하)**: allmap → depth(ch0), normal(ch2-4) → metric_depth_loss + normal_loss → backward
4. **손실 (의미론, Phase 2-B)**: rendered_features → semantic_loss(vs seg_map GT) + normal_consistency_loss(depth-derived vs rendered normal)
5. **밀도 제어**: net_wrapper.split_plane()/prune_small_plane() — 7 params(6 geo + f_i) 모두 동기 처리
6. **후처리**: merger() → TSDF mesh → merge_plane → PLY 출력

### 프리미티브 파라미터 (6+1 learnable + 2 non-learnable)
| 변수 | 차원 | 의미 | Learnable |
|------|------|------|-----------|
| `_plane_center` | (N,3) | 중심 위치 | ✓ lr=0.001 |
| `_plane_radii_xy_p` | (N,2) | +방향 반경 | ✓ lr=0.001 |
| `_plane_radii_xy_n` | (N,2) | -방향 반경 | ✓ lr=0.001 |
| `_plane_rot_q_normal_wxy` | (N,3) | 법선 회전 quat(w,x,y) | ✓ lr=0.001 |
| `_plane_rot_q_xyAxis_w` | (N,1) | 면내 회전 quat w | ✓ lr=0.001 |
| `_plane_rot_q_xyAxis_z` | (N,1) | 면내 회전 quat z | ✓ lr=0.001 |
| `_plane_semantic_features` | (N,4) | 의미론적 특징 (Phase 2-B) | ✓ lr=0.005 (enable_semantic 시) |
| `colors_precomp` | (N,4) | semantic=f_i / else=random | ✗ (rasterizer 입력) |
| `opacities` | (N,1) | 불투명도 (=1) | ✗ |

### allmap 채널 (rasterizer 출력)
| 채널 | 의미 |
|------|------|
| 0 | depth |
| 1 | alpha (누적 가중치) |
| 2-4 | normal (local 좌표) |
| 5 | mid_depth |
| 6 | distortion |

### 모니터링 스크립트 사용법 (Docker 컨테이너 내부)
```bash
# PLY export (normal 색상)
python scripts/visualize_primitives.py --checkpoint path/to/latest.pth --color_by normal

# PLY export (semantic class 색상: roof=red, wall=blue, ground=gray)
python scripts/visualize_primitives.py --checkpoint path/to/latest.pth --color_by class

# 평가 (depth_mae, normal_cos가 의미 있음. PSNR은 random color 때문에 무의미)
python scripts/evaluate.py --checkpoint path/to/latest.pth --metrics depth_mae normal_cos

# Semantic mIoU 포함 평가 (enable_semantic 학습 후)
python scripts/evaluate.py --checkpoint path/to/latest.pth --metrics depth_mae normal_cos semantic_miou

# 이전 결과와 비교
python scripts/evaluate.py --checkpoint path/to/latest.pth --compare_with prev_results.json
```

### TensorBoard 채널
**이미지 (tb_image_freq=500):**
| 태그 | 내용 | 비고 |
|------|------|------|
| `compare/1_rgb` | GT\|Rendered RGB side-by-side | 색상이 random이므로 참고용 |
| `compare/2_depth` | GT\|Rendered Depth (viridis) | 학습 추적에 유용 |
| `compare/3_normal` | GT\|Rendered Normal ((n+1)/2) | 학습 추적에 유용 |
| `compare/4_semantic` | GT\|Predicted semantic (enable_semantic 시) | Phase 2-B |
| `gt/*`, `render/*` | 개별 채널 (확대용) | |

**스칼라 (enable_semantic 시 추가):**
| 태그 | 내용 | 주기 |
|------|------|------|
| `loss/semantic` | L_sem 값 | log_freq=50 |
| `loss/geo_nc` | L_geo (normal consistency) 값 | log_freq=50 |
| `semantic/class_*_count` | 클래스별 프리미티브 수 (bg/roof/wall/ground) | 100 iter |

## 현재 진행 상태
- [x] Phase 0-Setup: 모니터링 환경 구축
- [x] Phase 0: SfM/MVS 입력 확보 (COLMAP 180장 정합, 100장 학습, Depth MAE=0.067, Normal cos=0.911)
- [x] Phase 1: MVS Depth+Normal Supervision 교체 (재학습 필요 — planar 읽기 버그 수정)
- [x] Phase 2-A: 2D Segmentation 생성 (v10: Confident Labels Only, normal 소스 재검토 필요)
- [x] Phase 2-B: 의미론적 헤드 구현 (f_i parameter, L_sem, L_geo, CUDA backward fix, gradient isolation 검증 완료)
- [ ] Phase 2-C: L_sem 독립 학습 (재학습 필요 — Phase 1 재학습 후)
- [ ] Phase 3-A: L_mutual 구현
- [ ] Phase 3-B: Ablation 7조건 학습
- [ ] Phase 3-C: L_photo 추가 실험 (선택적, core ablation 이후)
- [ ] Phase 4: CityGML 변환 + 검증

## 중요 규칙
- **Docker:** 모든 명령은 컨테이너 내부에서. pip install 시 Dockerfile에도 반영.
- f_i는 adaptive density control (split/clone/prune) 시 반드시 함께 처리 (구현 완료, Phase 2-B)
- L_mutual에서 detach 금지 (양방향 gradient 필수). ablation 시에만 선택적 detach
- **Semantic 렌더링**: Option A 확정 (raw f_i → colors_precomp → CUDA rasterizer alpha-blend → softmax → CE loss)
- **CUDA rasterizer 수정 사항** (Phase 2-B): config.h NUM_CHANNELS=4, backward.cu에 color gradient atomicAdd 추가. Color→alpha gradient path는 의도적 미구현 (L_sem→geometry 격리).
- 기존 PlanarSplatting 기능 보존: `--enable_semantic` 플래그로 새 기능 on/off (default=False)
- 각 Phase 완료 시 results/phaseX/REPORT.md 생성 (정량+정성 결과 포함, 템플릿은 EXPERIMENT_PLAN.md 하단 참조)
- Phase 진행 시 docs/EXPERIMENT_PLAN.md의 해당 Phase를 반드시 읽고 따를 것