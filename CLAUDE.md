# Semantic PlanarSplatting → CityGML LOD2

## 프로젝트 개요
PlanarSplatting(CVPR 2025)을 확장하여 항공 드론 이미지에서 건물 구성요소 의미론(roof/wall/terrain)이 통합된 평면 프리미티브를 학습하고, CityGML LOD2 건물 모델을 자동 생성하는 박사 연구. 세 가지 설계 선택의 효과를 실험적으로 검증:
- **설계 선택 1**: 평면 프리미티브 + 의미론 통합 동시 최적화
- **설계 선택 2**: 도메인 지식의 미분 가능 인코딩 (L_mutual)
- **설계 선택 3**: 의미론적 프리미티브에서 폐합 건물 모델 구성 (Stage 3)

## 상세 문서
- `docs/EXPERIMENT_PLAN.md` — 실험 순서, 프롬프트, REPORT 템플릿
- `docs/RESEARCH_CONTEXT.md` — 연구 배경, 파라미터, 손실 함수
- `docs/ADVISOR_FEEDBACK_RESPONSE.md` — 지도교수 피드백 대응

## 파이프라인 (Stage)
- **Stage 1**: SfM/MVS (3D Scene Initialization) — COLMAP + Grounded SAM
- **Stage 2**: Joint Geometric-Semantic Optimization — 미분 가능 렌더링
- **Stage 3**: Structured Building Model Extraction — CityGML LOD2 구성

## 개발 환경: Docker
- 이미지: `planarsplat:cu121` (CUDA 12.1, PyTorch 2.3.1)
- 서비스: `planarsplat` (학습+TB), `shell` (대화형)
- 포트: TB(16006), Sync Viewer(18080), Gradio(7860)
- GPU: `gpus: all`, `shm_size: 16g`

## 프리미티브 파라미터
| 변수 | 차원 | 의미 |
|------|------|------|
| `_plane_center` | (N,3) | 중심 |
| `_plane_radii_xy_p/n` | (N,2) | 반경 |
| `_plane_rot_q_normal_wxy` | (N,3) | 법선 회전 |
| `_plane_rot_q_xyAxis_w/z` | (N,1) | 면내 회전 |
| `_plane_semantic_features` | (N,4) | 의미론 (BG/Roof/Wall/Terrain) |
| color | (N,3) | RGB (L_photo용, 구현 완료) |

## 손실 함수
```
L_depth  → c_i, r_i, R_i         (기하학적 유효성)
L_normal → R_i                    (기하학적 유효성)
L_geo    → R_i (L_nc만 활성)      (기하학적 유효성)
L_sem    → f_i (gradient 격리)    (의미적 정확성)
L_photo  → color_i, c_i, R_i     (기하학적 유효성, 구현 완료)
L_mutual → R_i + f_i (양방향)    (기하학적 유효성 + 의미적 정확성)
```

## L_mutual (4항)
- L_vert: 벽 법선 → 수평 (gravity에 수직)
- L_slope: roof가 벽처럼 수평이면 penalty (τ=0.15)
- L_horiz: terrain 법선 → 수직 (gravity에 평행)
- L_height: roof는 terrain보다 위 (높이 기반)

## Gravity 추정
- **방법**: Grounded SAM 2D GT의 terrain 영역 MVS 법선 평균 = UP → gravity = -UP
- **시점**: Stage 1 출력에서 학습 전 사전 계산 (bootstrap 문제 없음)
- **검증**: roof/terrain 법선이 독립적으로 gravity와 정렬되는지 교차 검증
- Hardcoded [0,-1,0] 사용 금지

## Semantic Class (K=4)
| Index | Class | CityGML | 역할 |
|-------|-------|---------|------|
| 0 | BG | — | ignore_index |
| 1 | Roof | RoofSurface | 직접 매핑 |
| 2 | Wall | WallSurface | 직접 매핑 |
| 3 | Terrain | — (미포함) | Context + gravity + 지면 높이 |

GroundSurface는 학습 class가 아님. Stage 3에서 roof 경계를 terrain 높이로 투영하여 추론 생성.

## 현재 진행 상태
- [x] Stage 1 완료 (SfM/MVS, Grounded SAM)
- [x] Stage 2 예비 실험 완료 (gravity 미보정 + L_photo 미포함)
- [x] Stage 2 L_photo 구현 완료
- [x] Stage 3 알고리즘 (2.5D Hybrid + convex hull fallback + make_valid)
- [x] **Synthetic A (3D BAG)**: 3D BAG 543건물 × 17조건, 완료 (3/27)
  - **데이터**: 3D BAG 3 scene (Amsterdam/Rotterdam/Delft), roof type별 층화 추출
  - **핵심 결과**: 법선이 지배적 요인 (N10°: 90%, N20°: 55%), 나머지 95%+
  - **구조적 품질과 의미론적 정확성은 독립적** — L_mutual 필요성 뒷받침
  - **Slanted roof가 법선에 가장 민감** (N10°: slanted 83% vs horizontal 96%)
  - 복합: N10_worst 80% (단독 90% 대비 -10%p) — 법선이 지배적이나 엄밀한 충분조건은 아님
  - **면적/위치/누락은 영향 미미** (최대에서도 98-99%)
- [ ] **Synthetic B: L_mutual → CityGML 검증** ← **다음**
- [ ] Real: ISPRS + 성수동 + City3D 비교

## 중요 규칙
- Docker 내부에서 모든 명령 실행
- **L_photo 기본 포함**
- **Gravity**: terrain MVS 법선 기반 사전 추정. hardcoded 금지.
- **L_mutual 4항**: L_vert + L_slope + L_horiz + L_height
- **Terrain**: context class. CityGML 미포함. GroundSurface는 Stage 3에서 추론.
- **법선 표현**: "벽의 법선은 수평(gravity에 수직)" 통일
- L_geo: L_nc만 활성. L_planar/L_adj 미구현 (val3dity 결과 따라 검토)
- f_i density control 동기 처리 (구현 완료)
- L_mutual detach 금지 (ablation 시에만)
- 각 Step 완료 시 REPORT.md 생성 (EXPERIMENT_PLAN 템플릿 준수)
- **시각적 산출물 필수 생성** (프롬프트 명시 항목 모두)
- Stage 용어 사용 (Phase 폐지)
