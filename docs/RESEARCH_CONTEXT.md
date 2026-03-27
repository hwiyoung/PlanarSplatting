# 연구 배경 및 설계

## 문제 정의
항공 드론 이미지에서 의미론적 평면 프리미티브를 학습하여 CityGML LOD2 건물 모델을 자동 생성한다. CityGML 품질(기하학적 유효성, 의미적 정확성, 위상적 일관성)을 달성하기 위해 세 가지 설계 선택과 그 효과를 실험으로 검증한다.

## 파이프라인
- **Stage 1: SfM/MVS** — COLMAP 포즈/포인트/depth/normal + Grounded SAM 2D seg GT
- **Stage 2: Joint Geometric-Semantic Optimization** — 평면 프리미티브에 의미론 통합, 미분 가능 렌더링 동시 최적화
- **Stage 3: Structured Building Model Extraction** — 의미론적 프리미티브에서 CityGML LOD2 구성

---

## 프리미티브 파라미터

| 파라미터 | 변수명 | 차원 | 관련 손실 |
|----------|--------|------|-----------|
| c_i | `_plane_center` | (N,3) | L_depth, L_photo |
| r_i | `_plane_radii_xy_p/n` | (N,2)×2 | L_depth |
| R_i | `_plane_rot_q_normal_wxy` + `xyAxis` | (N,3)+(N,1)×2 | L_normal, L_mutual, L_photo |
| f_i | `_plane_semantic_features` | (N,4) | L_sem, L_mutual |
| color_i | (구현 완료) | (N,3) | L_photo |

**Multi-primitive 특성**: PlanarSplatting은 장면을 수천 개의 작은 프리미티브로 표현. 건물면 하나당 수십~수백 개. Stage 3의 핵심 과제는 이 다수 프리미티브를 건물면 단위로 클러스터링하여 CityGML gml:Polygon으로 변환하는 것.

---

## Semantic Class (K=4)

| Index | Class | CityGML 대응 | 역할 |
|-------|-------|-------------|------|
| 0 | BG | — | ignore_index |
| 1 | Roof | RoofSurface | 직접 매핑 |
| 2 | Wall | WallSurface | 직접 매핑 |
| 3 | Terrain | — (미포함) | Context class |

**Terrain 역할 3가지:**
1. 건물/비건물 경계 정의 → roof/wall 분류 정확도 향상. 벽 하단이 terrain과 만나는 지점이 건물 수직 범위 결정.
2. Gravity 추정: terrain 프리미티브 법선 평균 = UP → gravity = -UP
3. 지면 높이 참조: GroundSurface 추론 시 활용

**CityGML GroundSurface**: 학습 class가 아님. Stage 3에서 roof 경계를 terrain 높이로 투영하여 추론 생성.

---

## Gravity 추정

**방법**: Grounded SAM 2D GT의 terrain 영역에 해당하는 MVS 법선 평균 = UP → gravity = -UP.
**Bootstrap 해결**: 학습 전 Stage 1 출력에서 계산 가능. terrain 프리미티브가 아직 분류되지 않은 학습 초기에도 사용 가능.
**검증**: roof/terrain 법선이 독립적으로 추정 gravity와 정렬되는지 교차 검증.
**이력**: 기존 예비 실험에서 [0,-1,0] 사용 → 부정확 발견 → terrain 법선 기반 보정.

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

### L_mutual (4항)
```
L_mutual = Σ_i [ p_wall·L_vert + p_roof·L_slope + p_terrain·L_horiz + L_height ]
```
- L_vert = (n·e_g)²: 벽 법선 → 수평 (gravity에 수직)
- L_horiz = (1-|n·e_g|)²: terrain 법선 → 수직
- L_slope = relu(τ-(n·e_g)²)², τ=0.15: roof가 벽처럼 수평이면 penalty
- L_height = p_terrain·relu(h_i-h_high)² + p_roof·relu(h_low-h_i)²: roof > terrain 높이 제약

**사후 휴리스틱 vs L_mutual**: 같은 지식, 다른 적용. 사후=재구축 후 레이블링(교정 불가), L_mutual=최적화 중 양방향 교정(수천 iter).

### L_geo
L_nc만 활성. L_planar/L_adj 설계 완료, 미구현. val3dity 결과에 따라 검토.

### Warmup
- 0~N/3: λ_m=0 (기하+의미론 독립 안정화)
- N/3~2N/3: λ_m 점진 증가
- 2N/3~N: λ_m 목표값

---

## Stage 3: Structured Building Model Extraction

### 방법론 (6단계)
1. **프리미티브 분류/필터링**: argmax(softmax(f_i)) → roof/wall/terrain/BG. Terrain+BG 필터링.
2. **Multi-primitive 클러스터링**: 동일 class + 법선 유사도(cos>0.95) + 공간 근접(중심거리<2×r_i) → 건물면 클러스터. 대표 평면 = 소속 프리미티브 법선/중심 가중 평균.
3. **평면 교차 → 다면체**: 인접 클러스터의 대표 평면 교차 → 모서리/꼭짓점 → 폴리곤. 인접성: 프리미티브 최소 거리 < threshold + 비평행(법선 각도>10°).
4. **Building Instance 분리**: Roof 중심 XZ 투영 → connected component → building_id. Wall → 최근접 roof 클러스터 귀속.
5. **GroundSurface 생성**: Roof 경계를 terrain 높이로 투영 (항공 관측 본질적 한계).
6. **CityGML + val3dity**: gml:Polygon + BoundarySurface 매핑 + val3dity 검증.

### 현재 상태
- 합성 데이터(noise=0~0.3): 정상 작동
- Real data: wall 법선 51%만 수직(gravity 미보정 시), 면 누락 → 유효 CityGML 생성 한계
- TSDF mesh 활용: 미구현, 향후 개선 방향

---

## 합성 데이터 실험 설계

### GT 소스
공개 벤치마크 CityGML(뉴욕, 베를린 등) 건물 20개. 형태 다양 + 성수동 유사.

### 실험 A — 프리미티브 수준 노이즈 (Stage 3 단독)
- GT CityGML → 프리미티브 변환 (면당 10, 30, 50개)
- 노이즈: 법선(2°~20°), 위치(0.1~1.0m, 3축+수직/수평), 분류(5~30%), 면 누락(10~50%), 아웃라이어(1~10%)
- Stage 3 → val3dity + 면 IoU + 의미론 accuracy + Hausdorff
- 한계: 인공 프리미티브 ≠ 실제 Stage 2 출력

### 실험 B — 감독 신호 수준 노이즈 (Stage 2+3 통합)
- GT CityGML → 합성 렌더링 → 감독 신호 노이즈 (clean/depth/seg/뷰 누락)
- Stage 2 학습 → Stage 3 → GT 비교
- 한계: L_photo 행동 차이

---

## 예비 실험 결과 (Pilot, gravity 미보정 + L_photo 미포함)

- Pilot-Baseline (Independent): Normal cos=0.7816, mIoU=0.7859
- Pilot-Joint: Normal cos +0.0008, mIoU **-0.0134**
- 경로 1(cross-view contamination), 경로 2(직접 충돌) 진단
- **Gravity 미보정이 L_mutual 악화 원인일 가능성 → gravity 보정 후 재실험**

---

## 데이터
성수동 드론 이미지 180장 (oblique, 70m, 2048x1365, COLMAP 100장)
향후 nadir multi-view 추가 예정
