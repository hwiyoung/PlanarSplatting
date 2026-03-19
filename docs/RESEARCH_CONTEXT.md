# 연구 배경 및 설계

## 문제 정의
항공 드론 이미지에서 의미론적 평면 프리미티브를 학습하여 CityGML LOD2 건물 모델을 자동 생성한다. CityGML 품질(기하학적 유효성, 의미적 정확성, 위상적 일관성)을 달성하기 위해, 파이프라인의 각 설계 선택이 어떻게 기여하는지를 실험으로 검증한다.

### 연구 방향 조정 이력
- 2026-03-11: 지도교수 피드백 반영. L_photo 기본, Phase 4 우선, L_mutual 상대화.
- 2026-03-19: Gravity 보정 발견. 기존 Phase 3-B를 예비 분석으로 재위치.

## 파이프라인
1. **SfM/MVS** — COLMAP으로 포즈, 포인트 클라우드, depth/normal 추출. Grounded SAM 2로 2D seg GT 생성.
2. **Joint Geometric-Semantic Optimization** — 평면 프리미티브에 의미론 통합, 미분 가능 렌더링으로 동시 최적화.
3. **Structured Building Model Extraction** — 의미론적 프리미티브에서 CityGML LOD2 구성.

---

## 프리미티브 파라미터

| 파라미터 | 변수명 | 차원 | 관련 손실 |
|----------|--------|------|-----------|
| c_i | `_plane_center` | (N,3) | L_depth, L_photo |
| r_i | `_plane_radii_xy_p/n` | (N,2)×2 | L_depth |
| R_i | `_plane_rot_q_normal_wxy` + `xyAxis` | (N,3)+(N,1)×2 | L_normal, L_mutual, L_photo |
| f_i | `_plane_semantic_features` | (N,4) | L_sem, L_mutual |
| color_i | (구현 완료) | (N,3) | L_photo |

---

## Semantic Class (K=4)

| Index | Class | CityGML 대응 | 역할 |
|-------|-------|-------------|------|
| 0 | BG | — | ignore_index, 학습 제외 |
| 1 | Roof | bldg:RoofSurface | 직접 매핑 |
| 2 | Wall | bldg:WallSurface | 직접 매핑 |
| 3 | Terrain | — (CityGML 미포함) | Context class |

**Terrain의 역할:**
1. 건물/비건물 경계 정의 → roof/wall 분류 정확도 향상
2. Gravity 추정: terrain 프리미티브 법선 평균 = UP → gravity = -UP
3. 지면 높이 참조: GroundSurface 추론 시 활용

**CityGML GroundSurface**: 학습된 class가 아님. Stage 3에서 roof 경계를 terrain 높이로 투영하여 추론 생성.

---

## Gravity 추정

**문제**: COLMAP 좌표계에서 gravity 방향은 데이터마다 다름. 기존 [0,-1,0] 가정이 부정확.
**방법**: terrain 프리미티브(class=3) 법선 평균 = UP → gravity = -UP, normalize.
**검증**: roof/terrain 법선이 독립적으로 추정 gravity와 93% 정렬, 두 클래스 간 cos=0.9986.
**적용**: 학습 시작 시 1회 계산, 학습 중 고정. L_mutual의 e_gravity에 사용.

---

## 손실 함수

```
L = λ_d·L_depth + λ_n·L_normal + λ_g·L_geo + λ_s·L_sem + λ_p·L_photo + λ_m·L_mutual
```

| 손실 | 작용 대상 | CityGML 품질 대응 |
|------|----------|-----------------|
| L_depth | c_i, r_i, R_i | 기하학적 유효성 |
| L_normal | R_i | 기하학적 유효성 (면 방향) |
| L_geo (L_nc) | R_i | 기하학적 유효성 (내부 일관성) |
| L_sem | f_i | 의미적 정확성 |
| L_photo | color_i, c_i, R_i, r_i | 기하학적 유효성 (벽면 보완) |
| L_mutual | R_i + f_i | 기하학적 유효성 + 의미적 정확성 |

### L_mutual 수식
```
L_mutual = Σ_i [ p_wall·L_vert(n_i) + p_roof·L_slope(n_i) + p_terrain·L_horiz(n_i) ]
```
- L_vert = (n·e_g)² : wall 법선 → 수직(gravity에 수직)
- L_horiz = (1-|n·e_g|)² : terrain 법선 → 수평(gravity에 평행)
- L_slope = relu(τ-(n·e_g)²)², τ=0.15 : roof가 wall처럼 수평이면 penalty
- e_g: gravity 방향 (terrain 법선 기반 추정)

**사후 휴리스틱과의 차이**: City3D는 동일 지식을 재구축 후 적용(법선 교정 불가). L_mutual은 최적화 중 적용(양방향 교정, 수천 iter).

### L_geo
L_nc만 활성. L_planar/L_adj 설계 완료, 미구현. val3dity 결과에 따라 추가 검토.

### Warmup
- 0~N/3: λ_m=0 (기하+의미론 독립 안정화)
- N/3~2N/3: λ_m 점진 증가
- 2N/3~N: λ_m 목표값

---

## Phase 3-B 예비 분석 결과

L_photo 미포함 + gravity 미보정([0,-1,0]) 조건에서 7개 ablation 수행.
- (c) Independent: Normal cos=0.7816, mIoU=0.7859
- (d) Joint: Normal cos +0.0008, mIoU **-0.0134**
- 경로 1(cross-view contamination), 경로 2(직접 충돌) 진단 → Phase 3-B' 설계 근거
- **gravity 미보정이 L_mutual 악화의 원인일 가능성 → gravity 보정 후 재실험**

---

## Phase 4 관찰

Cluster-Intersection 기반 CityGML 변환 시도:
- 합성 데이터(noise 0~0.3): 정상 작동
- Real data: wall 법선 51%만 수직, 관측되지 않은 면 누락 → 유효 CityGML 생성 한계
- Gravity [0,-1,0] → [0.12, 0.37, 0.92] 보정 발견

---

## 데이터
성수동 드론 이미지 180장 (oblique, 70m, 2048x1365, COLMAP 100장)
향후 nadir multi-view 추가 예정
