# Semantic PlanarSplatting

## 프로젝트 개요
PlanarSplatting(CVPR 2025)을 확장하여 항공 드론 이미지에서 건물 구성요소 의미론(roof/wall/ground)이 통합된 평면 프리미티브를 학습하고, CityGML LOD2 형식의 건물 모델을 자동 생성하는 박사 연구. 기여의 정확한 경계는 실험 결과에 따라 확정하며, 현재 세 가지 방향을 병행 탐색 중:
- **방향 A**: 평면 프리미티브 + 의미론 통합이 CityGML LOD2 생성에 적합함을 실증
- **방향 B**: 의미론-기하학 동시 최적화를 통한 상호보완 (L_mutual)
- **방향 C**: TSDF mesh의 watertight 특성으로 splatting 토폴로지 보정

### 연구 방향 조정 (2026-03-11)
- L_photo를 기본 설정에 포함 (L_photo 미포함 조건은 유지하지 않음)
- Phase 4(CityGML 변환 + mesh 보정)를 최우선으로 진행
- L_mutual은 파이프라인의 한 구성요소 (효과와 한계 분석 대상)
- 기존 Phase 3-B 결과는 "L_photo 미포함 예비 분석"으로 재위치
- Ground class = 지면(terrain), CityGML GroundSurface(건물 바닥면)와 구별. Context class.
- 상세: docs/ADVISOR_FEEDBACK_RESPONSE.md

## 상세 문서 (필요 시 참조)
- `docs/RESEARCH_CONTEXT.md` — 연구 배경, 파라미터 구조, 손실 함수 설계
- `docs/EXPERIMENT_PLAN.md` — Phase별 목표/프롬프트/실험 우선순위
- `docs/ADVISOR_FEEDBACK_RESPONSE.md` — 지도교수 피드백 대응, 기여 방향, 실험 우선순위

## 개발 환경: Docker 기반
모든 개발과 실행은 Docker 컨테이너 내부에서 수행.

- **Docker 이미지:** `planarsplat:cu121` (nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04, PyTorch 2.3.1)
- **서비스:**
  - `planarsplat` (컨테이너: `planarsplat`) — 학습+TensorBoard+웹뷰어. `tools/run_monitor.sh`
  - `shell` (컨테이너: `planarsplat-shell`) — 대화형 셸 (Gradio 7860)
- **볼륨:** `./` → `/workspace/PlanarSplatting`, `./data`, `./planarSplat_ExpRes`, `./user_inputs`, `./.cache/huggingface`
- **포트:** TensorBoard(6006→16006), Sync Viewer(18080), Gradio(7860)
- **GPU:** `gpus: all`, `shm_size: 16g`, `ipc: host`

## 코드 구조
```
PlanarSplatting/
├── planarsplat/
│   ├── confs/base_conf_planarSplatCuda.conf
│   ├── run/trainer.py, net_wrapper.py
│   ├── net/net_planarSplatting.py
│   ├── utils/loss_util.py, mesh_util.py, merge_util.py
├── submodules/diff-rect-rasterization/ (CUDA rasterizer)
├── scripts/ (evaluate.py, visualize_primitives.py, generate_segmentation.py 등)
├── tools/ (run_monitor.sh, run_sync_dual_viewer.py)
└── utils_demo/ (run_planarSplatting.py, demo.conf)
```

### 프리미티브 파라미터 (7 learnable + 2 non-learnable)
| 변수 | 차원 | 의미 | Learnable |
|------|------|------|-----------|
| `_plane_center` | (N,3) | 중심 | ✓ lr=0.001 |
| `_plane_radii_xy_p` | (N,2) | +반경 | ✓ lr=0.001 |
| `_plane_radii_xy_n` | (N,2) | -반경 | ✓ lr=0.001 |
| `_plane_rot_q_normal_wxy` | (N,3) | 법선 회전 | ✓ lr=0.001 |
| `_plane_rot_q_xyAxis_w` | (N,1) | 면내 회전 w | ✓ lr=0.001 |
| `_plane_rot_q_xyAxis_z` | (N,1) | 면내 회전 z | ✓ lr=0.001 |
| `_plane_semantic_features` | (N,4) | 의미론 (bg/roof/wall/ground) | ✓ lr=0.005 |
| `colors_precomp` | (N,4) | rasterizer 입력 | ✗ |
| `opacities` | (N,1) | =1 고정 | ✗ |

### 손실 함수 구조 (L_photo 포함 설계)
```
L_depth  ──→ c_i, r_i, R_i
L_normal ──→ R_i
L_geo    ──→ R_i  (현재 L_nc만 활성)
L_sem    ──→ f_i  (gradient 격리: f_i에만 흐름)
L_photo  ──→ color_i, c_i, R_i, r_i  (Phase 3-B'에서 추가)
L_mutual ──→ R_i + f_i  (양방향, 이 둘만 연결)
```

### 모니터링 스크립트
```bash
python scripts/visualize_primitives.py --checkpoint path --color_by normal|class
python scripts/evaluate.py --checkpoint path --metrics depth_mae normal_cos semantic_miou
```

## 현재 진행 상태
- [x] Phase 0-Setup ~ Phase 3-A: 완료
- [x] Phase 3-B: 7조건 ablation 완료 (L_photo 미포함, **예비 분석으로 재위치**)
  - mIoU -0.0134 (L_mutual 악화), 양방향 시너지 미입증
  - 경로 1(cross-view contamination) / 경로 2(직접 충돌) 진단 → Phase 3-B' 설계 근거
- [x] **Phase 3-B'-Step1**: L_photo 구현 + (c') baseline 확립 (Depth MAE -5.1%, PSNR 15.0dB)
- [ ] **Phase 4 프로토타입**: CityGML 변환 + mesh 보정 ← **다음**
- [ ] Phase 3-B'-Step2: L_mutual 재실험 (L_photo 포함)
- [ ] City3D 비교
- [ ] Phase 4 고도화 (mesh 토폴로지 보정)

## 중요 규칙
- **Docker:** 모든 명령은 컨테이너 내부. pip install 시 Dockerfile에도 반영.
- **L_photo 기본 포함**: 새로운 실험은 L_photo를 기본 설정에 포함. L_photo 미포함 조건은 유지 불요.
- **Ground class**: 지면(terrain). CityGML GroundSurface(건물 바닥면)와 다름. Context class로서 roof/wall 식별 정확도 향상 목적. CityGML 변환 시 필터링.
- f_i는 density control (split/clone/prune) 시 함께 처리 (구현 완료)
- L_mutual에서 detach 금지 (양방향 gradient). ablation 시에만 선택적 detach
- **L_mutual 구조** (3항):
  - `p_wall · L_vert`: 벽 법선 → 수평
  - `p_roof · L_slope`: 지붕이 벽처럼 수평이면 penalty (τ=0.15)
  - `p_ground · L_horiz`: 지면 법선 → 수직
  - `e_gravity = [0,-1,0]` (COLMAP frame)
- **CUDA rasterizer**: NUM_CHANNELS=7 (3 RGB + 4 semantic, Phase 3-B'), backward.cu에 color gradient atomicAdd (Phase 2-B)
- `--enable_semantic` 플래그로 의미론 on/off (default=False)
- 각 Phase 완료 시 results/phaseX/REPORT.md 생성 (EXPERIMENT_PLAN.md 하단 템플릿 준용)
  - **정성적 결과**: 파일 경로 나열이 아니라, 각 이미지를 언급하며 관찰 내용 서술
  - **PLY 필수 산출물**: normal.ply + class.ply + rgb.ply (enable_photo 시). 각 PLY 관찰 서술
- Phase 진행 시 docs/EXPERIMENT_PLAN.md 반드시 참고
