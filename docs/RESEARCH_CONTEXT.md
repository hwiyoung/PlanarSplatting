# 연구 배경 및 설계

## 문제 정의
항공 드론 이미지에서 3D Gaussian Splatting 기반 평면 프리미티브를 학습하여, 건물 구성요소 의미론(roof/wall/ground)이 통합된 3D 표현을 구축하고, 이를 CityGML LOD2 형식의 건물 모델로 변환한다. 최종 출력은 val3dity를 통과하는 CityGML LOD2 모델이며, 건물 에너지 시뮬레이션, 태양광 잠재량 분석 등 도시 스케일 응용에 활용 가능한 데이터를 생성한다.

### 연구 방향 조정 (2026-03-11, 지도교수 피드백)
- L_mutual은 연구의 유일한 핵심 기여가 아님. 전체 파이프라인(항공 이미지 → semantic planar primitive → CityGML LOD2)에서의 한 구성요소.
- L_photo를 기본 설정에 포함. Phase 4(CityGML 변환 + mesh 기반 위상 보정)를 최우선으로 진행.
- 기여의 정확한 경계는 실험 결과(Phase 3-B', Phase 4, City3D 비교) 후 확정.
- 상세: docs/ADVISOR_FEEDBACK_RESPONSE.md

## 기존 연구의 한계
1. 3DGS 기반 건물 재구축(ULSR-GS, AGS 등): 기하학적 표면만 출력, 의미론 없음, CityGML 미연결
2. 이미지 기반 LOD2(PLANES4LOD2 등): CNN→후처리 분리, 3D 표현에 의미론 미통합
3. 기하학적 구조적 재구축(City3D, PolyFit): 순차적 파이프라인의 오차 누적, 사후적 의미론 부여
4. GS4Buildings: 단방향 (외부 LOD2→GS), GS→CityGML 아님
5. Gaussian Splatting 전반: 위상적 불완전성(구멍, 틈, 비밀폐)이 CityGML val3dity의 최대 장벽

## 기여 방향 (실험 결과에 따라 확정)
- **방향 A**: 평면 프리미티브 + 의미론 통합이 CityGML LOD2 생성에 왜 적합한지 실증 (City3D 비교)
- **방향 B**: 의미론-기하학 동시 최적화를 통한 상호보완 (L_mutual의 효과에 의존)
- **방향 C**: TSDF mesh의 watertight 특성으로 splatting의 토폴로지 보정 (기존 연구에 없음)

---

## 프리미티브 파라미터 전체 구조

### PlanarSplatting 원래 파라미터 (Θ_geo)

회전은 분해 쿼터니언: `q_final = normalize(q_normal × q_xyAxis)`

| 파라미터 | 변수명 | 차원 | 의미 | Learnable | lr | 관련 손실 |
|----------|--------|------|------|-----------|-----|-----------|
| c_i | `_plane_center` | (N, 3) | 중심 위치 | ✓ | 0.001 | L_depth |
| r_i+ | `_plane_radii_xy_p` | (N, 2) | +방향 반경 | ✓ | 0.001 | L_depth (간접) |
| r_i- | `_plane_radii_xy_n` | (N, 2) | -방향 반경 | ✓ | 0.001 | L_depth (간접) |
| q_n(w,x,y) | `_plane_rot_q_normal_wxy` | (N, 3) | 법선 회전 quat | ✓ | 0.001 | L_normal, L_mutual |
| q_xy(w) | `_plane_rot_q_xyAxis_w` | (N, 1) | 면내 회전 w | ✓ | 0.001 | L_normal (간접) |
| q_xy(z) | `_plane_rot_q_xyAxis_z` | (N, 1) | 면내 회전 z | ✓ | 0.001 | L_normal (간접) |

### 본 연구에서 추가하는 파라미터

| 파라미터 | 변수명 | 차원 | 의미 | lr | 관련 손실 |
|----------|--------|------|------|-----|-----------|
| f_i | `_plane_semantic_features` | (N, 4) | 의미론적 특징 (bg/roof/wall/ground) | 0.005 | L_sem, L_mutual |
| color_i | (추가 예정, Phase 3-B') | (N, 3) | RGB 색상 | TBD | L_photo |

### 파라미터-손실 매핑 (L_photo 포함 설계)

```
L_depth  ──→ c_i, r_i, R_i      (깊이 정확도)
L_normal ──→ R_i                  (법선 정확도)
L_geo    ──→ R_i                  (법선 일관성, 현재 L_nc만 활성)
L_sem    ──→ f_i                  (의미론 분류)
L_photo  ──→ color_i, c_i, R_i, r_i  (RGB 정합, 추가 예정)
L_mutual ──→ R_i + f_i           (기하-의미론 연결)
```

---

## Ground Class와 CityGML GroundSurface

**중요 구분**:
- 연구의 "ground" (K=4의 3번째 class) = 이미지에 보이는 실제 지면(terrain: 도로, 보도, 주차장). 항공 이미지에서 관측 가능.
- CityGML bldg:GroundSurface = 건물의 하부 폐합면(건물 풋프린트). 항공 이미지에서 관측 불가(건물이 위에 있으므로).

**Ground class의 목적**: 건물 프리미티브(roof/wall)를 더 정확하게 식별하기 위한 배경(context) class. "이것은 지면이지 건물이 아니다"를 학습. CityGML 변환 시 ground 프리미티브는 필터링되며 건물 모델에 포함되지 않음.

**CityGML GroundSurface 생성**: roof 경계를 ground 프리미티브의 평균 높이(z좌표)로 투영하여 별도 생성.

---

## L_sem (의미론적 분류 손실)

CrossEntropyLoss(ignore_index=0). 렌더링된 의미론적 맵 vs Grounded SAM 2 GT.
- 3D 프리미티브에 직접 f_i 부여 → 멀티뷰 의미론적 일관성이 구조적으로 보장
- 2D segmentation 오류는 다수결에 의해 희석
- GT coverage: Roof 5.9%, Wall 23.4%, Ground 19.5% (전체 48.8%), 나머지 51.2%는 BG(ignore)

---

## L_photo (RGB photometric loss) — Phase 3-B'에서 구현 예정

### 설계 변경 (2026-03-11)
기존: L_photo 미포함 (L_mutual ablation을 위한 실험적 편의).
변경: **L_photo를 기본 설정에 포함**. PGSR, 2DGS 등 기하학적으로 우수한 GS 연구들이 모두 L_photo를 사용. 특히 MVS 깊이가 희소한 벽면에서 텍스처 매칭이 기하학 보완 가능.

### 구현 사항 (Phase 3-B'-Step1에서 확인)
- PlanarSplatting 원본: color=random(비학습), SH 없음
- color를 학습 파라미터(N,3)로 추가, SfM 포인트 색상으로 초기화
- L_photo = (1-λ_ssim)*L1 + λ_ssim*(1-SSIM), λ_ssim=0.2
- semantic 렌더링(4ch)과 RGB 렌더링(3ch) 동시 수행 방안 검토 필요
- density control에서 color 동기 처리

### 기존 Phase 3-B 결과의 위상
L_photo 미포함 상태의 7개 ablation은 "예비 분석"으로 재위치. L_photo 포함 시 그대로 유효하다고 가정할 수 없음. 경로 1/2 진단의 논리적 구조는 Phase 3-B' 설계 근거로 유지.

---

## L_geo (기하 정규화)

```
L_geo = λ_p · L_planar + λ_a · L_adj + λ_nc · L_normal_consistency
```

현재 실험: **L_normal_consistency만 활성화**. L_planar, L_adj는 설계 완료, 구현 미사용. 미구현 사유는 코드 수준에서 확인 필요.

| 하위항 | 상태 | 비고 |
|--------|------|------|
| L_normal_consistency | 활성 (Phase 2-B) | depth-derived normal vs rendered normal |
| L_planar | 미사용 | 향후 Phase 4에서 검토 |
| L_adj | 미사용 | 향후 Phase 4에서 검토 |

---

## L_mutual 수식 및 Gradient 분석

### 수식 (3항, Phase 3-B에서 사용)
```
L_mutual = Σ_i [ p_wall(i)·L_vert(n_i) + p_roof(i)·L_slope(n_i) + p_ground(i)·L_horiz(n_i) ]
```
- e_gravity = [0,-1,0] (COLMAP frame)
- L_vert(n) = (n·e_g)² : 벽 법선 → 수평
- L_horiz(n) = (1-|n·e_g|)² : 지면 법선 → 수직
- L_slope(n) = relu(τ-(n·e_g)²)², τ=0.15 : roof가 벽처럼 수평이면 penalty

### 4항 (L_height, 실험 미수행)
```
L_height = p_ground(i)·relu(h_i-h_high)² + p_roof(i)·relu(h_low-h_i)²
```
L_height는 Phase 3-B' 결과에 따라 추가 여부 결정.

### L_mutual의 위치 (연구 방향 조정 후)
L_mutual은 전체 파이프라인의 한 구성요소. "의미론-기하학 상호보완"(방향 B)의 구체적 메커니즘이며, 효과는 L_photo 포함 조건에서 재검증. 효과가 확인되면 기여, 제한적이면 "작동 조건 분석" 수준의 기여로 조정.

### Gradient 방향
- ∂L/∂R_i (의미론→기하): p_wall이 크면 n_i를 수평으로 교정
- ∂L/∂f_i (기하→의미론): n_i가 수평이면 f_i를 wall로 교정
- L_sem만으로는 gradient 격리(f_i only)에 의해 의미론→기하 피드백 불가. L_mutual이 있어야 R_i에도 gradient.

---

## Warmup 전략

- 초기 (0~N/3): λ_m=0, 기하+의미론 독립 안정화
- 중기 (N/3~2N/3): λ_m 점진 증가
- 후기 (2N/3~N): λ_m 목표값

---

## Phase 3-B Ablation 설계

### 기존 (Phase 3-B, L_photo 미포함, 예비)
| 조건 | 구성 | 결과 (vs (c)) |
|------|------|-------------|
| (a) Geo Only | L_d+L_n+L_geo | — |
| (b) Sem Only | L_d+L_n+L_sem | — |
| (c) Independent | L_d+L_n+L_geo+L_sem | baseline |
| (d) Joint | +L_mutual full | Normal +0.0008, mIoU **-0.0134** |
| (e) Sem→Geo | softmax detach | mIoU -0.0025 |
| (f) Geo→Sem | n_i detach | mIoU -0.0163 |
| (g) No Warmup | no warmup | mIoU -0.0190 |

### Phase 3-B' (L_photo 포함, 본 실험)
| 조건 | 구성 |
|------|------|
| (c') Independent+Photo | L_d+L_n+L_geo+L_sem+L_photo (baseline) |
| (d') Joint+Photo | +L_mutual full |
| (d'-masked) | L_mutual을 GT 있는 영역에만 |
| (d'-small) | λ_m=0.01 |

---

## Phase 4: CityGML 변환 + Mesh 기반 위상 보정

### TSDF Mesh 활용 방향
mesh를 CityGML의 중간 표현으로 쓰는 것이 아니라, 프리미티브는 유지하면서 TSDF mesh의 watertight 특성으로 프리미티브 사이의 위상적 문제(틈, 겹침)를 보정.
- refuse_mesh()로 TSDF mesh 추출
- 프리미티브-mesh 대응 분석: 미커버 영역(틈) 식별
- 틈 영역에서 mesh 정보로 프리미티브 경계 확장/보충

### Building Instance 분리 (전략 2)
1. Roof 중심 XZ 투영 → connected component → building_id
2. Wall → 최근접 roof cluster 귀속
3. Ground 프리미티브 필터링 (CityGML에 미포함)
4. GroundSurface: roof 경계를 ground 높이로 투영하여 별도 생성

---

## 데이터
- 성수동 드론 이미지 180장 (oblique, 70m, 2048x1365, COLMAP 100장 학습)
