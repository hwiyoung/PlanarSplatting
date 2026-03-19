# Semantic PlanarSplatting

## 프로젝트 개요
PlanarSplatting(CVPR 2025)을 확장하여 항공 드론 이미지에서 건물 구성요소 의미론(roof/wall/terrain)이 통합된 평면 프리미티브를 학습하고, CityGML LOD2 건물 모델을 자동 생성하는 박사 연구. 기여의 경계는 실험 결과에 따라 확정하며, 세 방향을 탐색 중:
- **방향 A**: 평면 프리미티브 + 의미론 통합 → CityGML LOD2 생성 실증
- **방향 B**: 의미론-기하학 동시 최적화를 통한 상호보완 (L_mutual)
- **방향 C**: TSDF mesh watertight 특성으로 splatting 토폴로지 보정

### 연구 방향 조정 (2026-03-19)
- L_photo 기본 포함. L_photo 미포함 조건 유지 불요.
- Phase 4(CityGML + mesh 보정) 최우선.
- **Gravity 보정 발견**: 기존 [0,-1,0]이 부정확. Terrain 프리미티브 법선 평균으로 보정. 기존 Phase 3-B는 gravity 미보정 예비 분석.
- Terrain class = 지면(terrain). CityGML GroundSurface(건물 바닥면)와 구별. Context class.
- 상세: docs/ADVISOR_FEEDBACK_RESPONSE.md

## 상세 문서
- `docs/RESEARCH_CONTEXT.md` — 연구 배경, 파라미터, 손실 함수
- `docs/EXPERIMENT_PLAN.md` — 실험 순서/프롬프트/REPORT 템플릿
- `docs/ADVISOR_FEEDBACK_RESPONSE.md` — 지도교수 피드백 대응

## 개발 환경: Docker
- **이미지:** `planarsplat:cu121` (CUDA 12.1, PyTorch 2.3.1)
- **서비스:** `planarsplat` (학습+TB+뷰어), `shell` (대화형)
- **포트:** TB(6006→16006), Sync Viewer(18080), Gradio(7860)
- **GPU:** `gpus: all`, `shm_size: 16g`

## 코드 구조
```
PlanarSplatting/
├── planarsplat/
│   ├── confs/base_conf_planarSplatCuda.conf
│   ├── run/trainer.py, net_wrapper.py
│   ├── net/net_planarSplatting.py
│   ├── utils/loss_util.py, mesh_util.py, merge_util.py
├── submodules/diff-rect-rasterization/
├── scripts/ (evaluate.py, visualize_primitives.py, export_citygml.py 등)
└── tools/ (run_monitor.sh, run_sync_dual_viewer.py)
```

### 프리미티브 파라미터
| 변수 | 차원 | 의미 | Learnable |
|------|------|------|-----------|
| `_plane_center` | (N,3) | 중심 | ✓ |
| `_plane_radii_xy_p/n` | (N,2) | 반경 | ✓ |
| `_plane_rot_q_normal_wxy` | (N,3) | 법선 회전 | ✓ |
| `_plane_rot_q_xyAxis_w/z` | (N,1) | 면내 회전 | ✓ |
| `_plane_semantic_features` | (N,4) | 의미론 (bg/roof/wall/terrain) | ✓ |
| color (L_photo용) | (N,3) | RGB | ✓ (구현 완료) |

### 손실 함수
```
L_depth  → c_i, r_i, R_i
L_normal → R_i
L_geo    → R_i (L_nc만 활성)
L_sem    → f_i (gradient 격리)
L_photo  → color_i, c_i, R_i, r_i (구현 완료)
L_mutual → R_i + f_i (양방향)
```

### Gravity 추정
- 기존: [0,-1,0] (부정확, Phase 3-B에서 사용)
- 보정: terrain 프리미티브 법선 평균 → UP, gravity = -UP
- 학습 시작 시 1회 계산, 학습 중 고정

### Terrain class (K=4, class=3)
- 실제 지면(도로, 보도). CityGML GroundSurface(건물 바닥면)와 다름.
- 역할: (1) context class (roof/wall 식별 향상), (2) gravity 추정 소스, (3) 지면 높이 참조
- CityGML 변환 시 필터링. GroundSurface = roof 경계를 terrain 높이로 투영하여 생성.

## 현재 진행 상태
- [x] Phase 0~3-A: 완료
- [x] Phase 3-B: 완료 (**예비 분석: L_photo 미포함 + gravity 미보정**)
- [x] Phase 3-B' L_photo 구현: 완료
- [x] Phase 4 프로토타입: Cluster-Intersection 구현. **wall 법선 문제 + gravity 보정 발견**
- [ ] **Phase 3-B'-Step1**: Gravity 보정 + baseline 재학습 ← **현재**
- [ ] Phase 3-B'-Step2: L_mutual 재실험
- [ ] Phase 3-B'-Step3: CityGML 전제 조건 판정
- [ ] Phase 4-Sensitivity: 합성 데이터 민감도 분석
- [ ] Phase 4-Real: CityGML 생성
- [ ] City3D 비교

## 중요 규칙
- Docker 내부에서 모든 명령 실행
- **L_photo 기본 포함**
- **Gravity: terrain 프리미티브 법선 기반 추정** (hardcoded [0,-1,0] 사용 금지)
- **Terrain class**: context class, CityGML 미포함
- f_i density control 동기 처리 (구현 완료)
- L_mutual detach 금지 (ablation 시에만)
- L_geo: L_nc만 활성. L_planar/L_adj 미구현 (val3dity 결과에 따라 검토)
- 각 Phase 완료 시 REPORT.md 생성 (EXPERIMENT_PLAN.md 하단 템플릿 준수)
- **시각적 산출물 필수 생성** (프롬프트에 명시된 항목 모두)
