# 연구 배경 및 설계

## 문제 정의
항공 드론 이미지에서 3D Gaussian Splatting 기반으로 건물을 재구축할 때, 기하학적 재구축과 의미론적 분류를 동시에 수행하되, 두 과제가 서로를 개선하는 양방향 상호 보강을 실현한다.

## 기존 연구의 한계
1. 3DGS 기반 건물 재구축(ULSR-GS, AGS 등): 기하학적 표면만 출력, 의미론 없음
2. 이미지 기반 LOD2(PLANES4LOD2 등): CNN→후처리 분리, end-to-end 아님
3. GS4Buildings: 단방향 (외부 LOD2→GS), 양방향 아님
4. Multi-task learning(PCGrad, CAGrad): gradient 충돌 해결에 집중, 과제 간 구조적 연계 미활용

---

## 프리미티브 파라미터 전체 구조

### PlanarSplatting 원래 파라미터 (Θ_geo) — Phase 0-Setup에서 코드 확인 완료

PlanarSplatting은 L_photo 없이 기하학적 손실(L_depth, L_normal)만으로 최적화한다. **color는 매 forward에서 random 생성, opacity는 고정값 1.0**으로, 둘 다 학습 파라미터가 아니다. SH 계수도 없다.

회전은 **분해 쿼터니언 시스템**: `q_final = normalize(q_normal × q_xyAxis)` 로, 법선 방향 회전(azimuth)과 면내 회전(twist)을 분리하여 관리한다.

| 파라미터 | 변수명 | 차원 | 의미 | Learnable | lr | 관련 손실 |
|----------|--------|------|------|-----------|-----|-----------|
| c_i | `_plane_center` | (N, 3) | 중심 위치 | ✓ | 0.001 | L_depth |
| r_i+ | `_plane_radii_xy_p` | (N, 2) | +방향 반경 (x+, y+) | ✓ | 0.001 | L_depth (간접) |
| r_i- | `_plane_radii_xy_n` | (N, 2) | -방향 반경 (x-, y-) | ✓ | 0.001 | L_depth (간접) |
| q_n(w,x,y) | `_plane_rot_q_normal_wxy` | (N, 3) | 법선 회전 quat (w,x,y) | ✓ | 0.001 | L_normal, **L_mutual** |
| q_xy(w) | `_plane_rot_q_xyAxis_w` | (N, 1) | 면내 회전 quat w | ✓ | 0.001 | L_normal (간접) |
| q_xy(z) | `_plane_rot_q_xyAxis_z` | (N, 1) | 면내 회전 quat z | ✓ | 0.001 | L_normal (간접) |
| color | `torch.rand_like(center)` | (N, 3) | 렌더링 색상 | ✗ (random) | — | 없음 |
| opacity | `torch.ones(N, 1)` | (N, 1) | 불투명도 | ✗ (고정=1) | — | 없음 |

**쿼터니언 재구성 과정:**
- `q_normal = normalize([q_n_wxy, 0])` — z=0 고정 (3 DoF → 법선 방향)
- `q_xyAxis = normalize([q_xy_w, 0, 0, q_xy_z])` — x,y=0 고정 (2 DoF → 면내 twist)
- `q_final = normalize(q_normal × q_xyAxis)` → rotation matrix → **법선 n_i 유도**

**L_mutual이 작동하는 지점:** `q_n(w,x,y)` 가 법선 n_i를 결정하므로, L_mutual의 ∂L/∂R_i gradient는 주로 이 3개 파라미터를 통해 흐른다.

**Density control 시 동기 처리 필요:**
`net_wrapper.py`의 `prune_core()`, `densification_postfix()`, `split_planes_via_radii_grad()` 에서 6개 파라미터 + optimizer state를 모두 동기 처리한다. **f_i 추가 시 이 함수들에 반드시 포함시켜야 한다.**

### 본 연구에서 추가하는 파라미터 (Θ_sem)

| 파라미터 | 차원 | 의미 | 관련 손실 |
|----------|------|------|-----------|
| **f_i** | **4** | **의미론적 특징 (bg/roof/wall/ground)** | **L_sem, L_mutual** |

### 파라미터-손실 매핑 (기본 설계: L_photo 미포함)

```
L_depth  ──→ c_i, r_i, R_i      (깊이 정확도)
L_normal ──→ R_i                  (법선 정확도)
L_geo    ──→ c_i, r_i, R_i      (기하 정규화: L_nc + L_planar + L_adj)
L_sem    ──→ f_i                  (의미론 분류)
L_mutual ──→ R_i + f_i           (★ 이 둘만 연결)
```

### L_mutual이 연결하는 것과 연결하지 않는 것
- **연결하는 것:** R_i(→법선 n_i)와 f_i(→클래스 확률 p_c). 이 두 파라미터만.
- **연결하지 않는 것:** c_i(위치), r_i(크기), 기타. 이들은 L_mutual과 무관.
- "양방향"이란: L_mutual의 gradient가 R_i 방향과 f_i 방향 둘 다로 흐른다는 뜻.

---

## L_sem (의미론적 분류 손실)

CrossEntropyLoss(ignore_index=0)로, 렌더링된 의미론적 맵을 Grounded SAM 2 생성 GT와 비교.
- 3D 프리미티브에 직접 f_i를 부여하므로, **여러 뷰에서 번갈아 L_sem을 최적화하면 멀티뷰 의미론적 일관성이 구조적으로 보장**된다.
- 2D segmentation 오류가 있더라도 동일 프리미티브가 여러 뷰에서 L_sem을 받으므로 **다수결에 의해 오류 희석**.
- L_mutual의 기하학-의미론 교차 검증이 추가적인 보정 역할.

---

## L_photo 포함 여부: 실험으로 확인할 사항

### PlanarSplatting의 설계
PlanarSplatting은 L_photo 없이 MVS depth/normal 기반 기하학적 손실만으로 최적화한다.

### 기본 설계: L_photo 미포함 (이론적 근거)
1. **핵심 기여와 독립:** L_mutual은 R_i와 f_i만 연결. color/L_photo와 무관하므로, 없어도 핵심 메커니즘 성립.
2. **Ablation 명확성:** L_photo 추가 시 R_i에 대한 gradient source가 증가 (L_depth, L_normal, L_geo, L_mutual에 더해 L_photo). L_mutual의 고유 효과가 희석되어 "법선 개선이 L_mutual 덕인지 L_photo 덕인지" 분리 어려움.
3. **설계 일관성:** PlanarSplatting이 L_photo 없이 기하 재구축을 달성하므로 추가 필요성 없음.

### 추가 실험: L_photo 포함 시 효과 (Phase 3-B 이후)
L_photo 추가가 전체 품질을 높일 가능성은 있으므로, core ablation 완료 후 확인:
- color를 학습 파라미터로 추가하고 L_photo를 포함한 실험
- L_photo 유무에 따른 기하 지표 변화
- L_photo 유무에 따른 L_mutual ablation 효과 변화
- 수렴 안정성 비교

---

## L_geo (기하 정규화)

프리미티브가 "건물다운" 기하학적 조건을 만족하도록 강제하는 정규화 항. 세 가지 하위항으로 구성된다.

```
L_geo = λ_p · L_planar + λ_a · L_adj + λ_nc · L_normal_consistency
```

### L_normal_consistency (법선 일관성)
렌더링된 법선 맵이 깊이 맵으로부터 유도된 법선과 일치해야 한다는 제약. 깊이와 법선이 독립적으로 업데이트될 때 발생하는 모순을 방지. 2DGS, PGSR 등에서 검증된 기법.
```
L_nc = Σ_p [ 1 - n_render(p) · n_depth_derived(p) ]
```
- n_render: allmap[2:5] (rasterizer 출력)
- n_depth_derived: allmap[0] (depth)에서 finite difference로 유도
- **구현**: PyTorch 레벨, CUDA 수정 불필요 → Phase 2-B에서 구현

### L_planar (평면성 제약)
하나의 프리미티브에 속하는 모든 3D 점들이 실제로 하나의 평면 위에 놓여야 한다는 제약.
```
L_planar = Σ_i [ (1/|S_i|) · Σ_{x_j ∈ S_i} (n_i · x_j + d_i)² ]
```
- n_i · x_j + d_i: 점 x_j에서 프리미티브 i 평면까지의 부호 거리
- 렌더링된 깊이 맵에서 역투영한 3D 점이 해당 프리미티브의 평면 방정식과 일치하는지 검증
- **의존성**: 픽셀별 프리미티브 ID 필요 → CUDA rasterizer에 primitive ID 채널 추가 필요
- **구현**: Phase 4에서 CUDA 수정 시 함께 구현

### L_adj (인접 정합성)
인접한 두 프리미티브가 만나는 경계에서 틈(gap)이나 격침(overlap)이 없어야 한다는 제약. CityGML은 건물이 닫힌 다면체를 형성하도록 요구.
```
L_adj = Σ_{(i,j) ∈ Neighbors} [ (1/|B_ij|) · Σ_{x_k ∈ B_ij} ( |n_i·x_k+d_i| + |n_j·x_k+d_j| )² ]
```
- B_ij: 프리미티브 i와 j의 경계 영역 점 집합
- 인접성은 렌더링된 2D 맵에서 프리미티브 ID의 경계로 판단
- **의존성**: L_planar와 동일 (픽셀별 프리미티브 ID 필요)
- **구현**: Phase 4에서 CUDA 수정 시 함께 구현

### 구현 단계 요약
| 하위항 | Phase | CUDA 수정 | 비고 |
|--------|-------|-----------|------|
| L_normal_consistency | 2-B | 불필요 | allmap[0]과 allmap[2:5]에서 계산 |
| L_planar | 4 | 필요 (primitive ID 채널) | val3dity 검증과 직결 |
| L_adj | 4 | 필요 (primitive ID 채널) | CityGML 닫힌 다면체 요구 |

※ allmap[6](distortion)은 별도의 floater 방지 정규화로, L_geo의 하위항은 아니지만 추가 사용 가능.
※ `regularize_plane_shape()`은 반경 대칭화 hard constraint로, L_geo와 역할이 다름.

---

## L_mutual 수식 및 Gradient 분석

### 수식
```
L_mutual = Σ_i [ p_wall(i) · L_vert(n_i) + p_roof(i) · L_slope(n_i) + p_ground(i) · L_horiz(n_i) ]
```
- p_c(i) = softmax(f_i)[c] : 프리미티브 i의 클래스 c 확률
- n_i : R_i에서 유도된 법선 벡터
- e_z = 중력 방향 단위벡터 (COLMAP world frame에서 gravity ≈ -Y → e_z ≈ [0, -1, 0]. **[0,0,1]이 아님!**)
- L_vert(n) = (n · e_z)² : 법선이 수평이면 0 → 벽면에 적합 (벽 법선은 수평이어야)
- L_horiz(n) = (1 - |n · e_z|)² : 법선이 수직이면 0 → 지면에 적합 (지면 법선은 수직이어야)
- L_slope(n) = relu(τ − (n · e_z)²)² : **단측 wall exclusion**. τ=0.15.
  - roof로 분류된 프리미티브의 법선이 지나치게 수평(wall-like)이면 penalty
  - flat roof (수직 법선, (n·e_z)²≈1)에는 penalty 없음 → flat roof와 ground의 법선이 동일한 경우에도 L_mutual이 무해하게 퇴화
  - flat roof vs ground 구분은 L_sem이 담당 (2D GT + multi-view consistency)

### L_mutual은 렌더링과 독립
L_mutual은 **per-primitive softmax(f_i)**와 **per-primitive n_i**를 직접 사용하여 계산된다. rasterizer를 통한 렌더링(alpha-blending)은 거치지 않으므로, 렌더링 순서나 occlusion에 무관하다. 반면 L_sem은 colors_precomp에 raw f_i를 전달하여 rasterizer가 alpha-blend한 결과에 softmax → CrossEntropyLoss를 적용한다. 즉 L_sem만 렌더링 경로를 사용하고, L_mutual은 프리미티브 레벨에서 직접 작동한다.

### Gradient 방향 1: ∂L_mutual/∂R_i (의미론 → 기하)
"wall 확률이 높은 프리미티브의 법선을 수평으로 밀어라"
- p_wall이 크면 L_vert(n_i)의 가중치가 커지고, R_i를 수정하여 n_i를 수평으로 회전시키는 gradient 발생
- 의미론적 분류(p_wall)가 기하학적 법선(n_i)을 교정하는 메커니즘

### Gradient 방향 2: ∂L_mutual/∂f_i (기하 → 의미론)
"법선이 수평인 프리미티브를 wall로 분류하라"
- n_i가 이미 수평이면 L_vert(n_i)≈0, L_horiz(n_i)≈1
- p_wall을 키우면 L_mutual이 줄어드는 방향 → f_i를 wall 쪽으로 밀어주는 gradient 발생
- 이것이 기하학적 법선(n_i)이 의미론적 분류(f_i)를 보조하는 메커니즘

### 전체 손실 (기본 설계)
```
L_total = λ_d·L_depth + λ_n·L_normal + λ_g·L_geo + λ_s·L_sem + λ_m·L_mutual
```
※ L_geo는 Phase 2-B에서 L_normal_consistency만 포함. L_planar, L_adj는 Phase 4(CUDA 수정 후) 추가.
※ L_photo는 기본 설계에서 미포함. "L_photo 포함 여부" 섹션 참조.

---

## Warmup 전략 (Curriculum Learning)

학습 초기에 기하학과 의미론 모두 수렴하지 않은 상태에서 L_mutual을 활성화하면, 부정확한 의미론이 기하학을 오염시킬 수 있다. 시간적 분리로 이를 방지:

- **초기 (0 ~ N/3):** λ_m = 0. L_depth + L_normal + L_geo로 기하 안정화, L_sem으로 의미론 대략 학습.
- **중기 (N/3 ~ 2N/3):** λ_m를 0에서 목표값까지 점진적 증가. 상호 보강 시작.
- **후기 (2N/3 ~ N):** λ_m 목표값 유지. 상호 보강 완전 작동.

PCGrad/CAGrad 같은 gradient surgery 대신 시간 축 분리로 충돌 회피. 구현이 단순하면서 adaptive density control(split/prune)과 호환.

## Trivial Solution 방지
- 위험: L_mutual 최소화를 위해 p_wall을 줄이는 것이 n_i를 고치는 것보다 쉬움 (semantic evasion)
- 방어: L_sem이 독립적으로 2D GT와의 일치를 강제 → p_wall이 부적절하게 줄면 L_sem이 복원
- 추가: entropy regularization, class-balanced weighting (필요 시)

### 클래스 불균형 (seg_maps 실측 — DJI pitch 기반 최종)
현재 seg_maps coverage (180장 평균): Roof 5.1%, Wall 23.0%, Ground 19.7%, Background 52.3%.
- Background(=ignore) 비율이 높아 유효 라벨 밀도가 낮음 (47.7% coverage)
- Roof가 특히 적음 (전체의 5.1%, 유효 라벨 중 ~11%)
  - 원인: oblique view에서 MVS 퇴화 법선(|dot|≈|sin(pitch)|=0.755)이 ambiguous zone → 보수적 threshold(0.85) 필요
  - Roof 부족은 L_mutual이 기하→의미론 방향으로 보완하도록 설계됨
- **Phase 2-C에서 class-balanced weighting** (inverse frequency 또는 focal loss) 검토 필요
- 근본 원인: MVS normal 부재 이미지(80/180)는 text-only fallback (roof/wall 구분 불가)

### Oblique View 한계와 대응 전략

**한계:** Oblique view(pitch≈-49°)에서 roof가 적게 보이는 것은 촬영 조건의 본질적 한계.
- Roof 2D label이 부족 → L_sem의 roof supervision 약함
- 퇴화 MVS normal이 모든 view에서 발생 → multi-view consistency로도 교정 한계
- Wall은 "seg label 정확(ambiguous→wall) + L_mutual 교정" 경로가 살아있으나, Roof는 두 경로 모두 약함

**학습 단계 대응 (Phase 2-C~3):**
1. Class-balanced weighting (inverse frequency): Roof의 적은 label에 높은 가중치
2. L_mutual의 기하→의미론 방향: 수평 법선을 가진 primitive를 roof로 분류하도록 유도
3. L_geo (L_normal_consistency): depth 유도 normal로 퇴화 MVS normal을 간접 교정

**Seg map 개선 대안 (Phase 2-C에서 mIoU 미달 시):**
1. Text-only 80장에 multi-prompt voting (roof/wall 별도 검출 후 score 합산)
2. Pitch-adaptive threshold: 이미지별 |sin(pitch)| + margin으로 degenerate 경계 조정
3. Nadir view 혼합 촬영 (향후 데이터 확장 시)

**논문 기술 방향:** "oblique 전용 데이터셋에서의 한계 + nadir 혼합 시 개선 가능성"으로 기술.

---

## Ablation 설계 — 통합 번호 체계 (a)~(i)

### Core Ablation (a)~(d): 전체 접근법 검증 (스케치 4.4.1)

| 조건 | 손실 구성 | 의미론 헤드 | 검증하려는 것 |
|------|----------|-----------|-------------|
| **(a)** Geo Only | L_depth + L_normal + L_geo | 없음 | 기하 전용 baseline |
| **(b)** Sem Only | L_depth + L_normal + L_sem | 있음 | L_geo 없이 의미론만 추가 효과 |
| **(c)** Independent | L_depth + L_normal + L_geo + L_sem (λ_m=0) | 있음 | 독립 최적화 baseline |
| **(d)** Joint | L_depth + L_normal + L_geo + L_sem + L_mutual | 있음 | 양방향 상호 보강 효과 |

핵심 비교: **(c) vs (d)** — 유일한 차이가 L_mutual이므로, (d)가 (c)보다 우수하면 양방향 상호 보강 효과 직접 입증.
추가 비교: (a) vs (c) — 의미론 헤드 추가가 기하에 미치는 영향 (간접 효과).

### Directional Ablation (e)~(f): 방향별 메커니즘 검증 (기여2 핵심)

| 조건 | 설정 | 검증하려는 것 |
|------|------|-------------|
| **(e)** Sem→Geo only | softmax(f_i).detach() in L_mutual | R_i만 gradient. 의미론→기하 단방향 |
| **(f)** Geo→Sem only | n_i.detach() in L_mutual | f_i만 gradient. 기하→의미론 단방향 |

핵심 비교: **(d) vs (e) vs (f)** — 양방향이 각 단방향보다 우수한지. (d)의 개선 > (e) + (f)이면 시너지.
※ 스케치 원본에서 (e),(f)는 L_photo 실험이었으나 본 설계에서는 Directional ablation으로 재배치 (기여2 "양방향"의 핵심 증거이므로 우선순위 상향).

### Warmup Ablation: (d) vs (g)

| 조건 | 설정 | 검증하려는 것 |
|------|------|-------------|
| **(d)** Joint (기본) | warmup 적용 (3단계 curriculum) | baseline (위 Core의 (d)와 동일) |
| **(g)** No-warmup | warmup 없이 λ_m 즉시 적용 | warmup 필요성 검증 |

### L_photo Ablation (h)~(i): 선택적 추가 (스케치 4.4.2 → core 이후)

| 조건 | 설정 | 검증하려는 것 |
|------|------|-------------|
| **(h)** Photo+Indep | (c) + L_photo (λ_m=0) | 색상 추가 시 독립 기하/의미론 |
| **(i)** Photo+Joint | (d) + L_photo | 색상+양방향 시너지 |

핵심 비교: **(h) vs (i)** — L_photo 존재 하에서도 L_mutual 효과 유지되는지.
※ (h),(i)는 Phase 3-C에서 수행 (core ablation (a)~(g) 이후).

### 기대 및 대응

기대: (d)가 (c) 대비 기하+의미론 모두 개선, (d)의 개선 > (e) 단독 + (f) 단독 → 시너지

결과가 기대와 다를 경우:
- 효과 없음 → "조건과 한계" 분석 (MVS depth가 이미 충분히 정확하여 L_mutual의 추가 보강 여지가 적음 등)
- 한 방향만 효과 → "항공 데이터에서는 Sem→Geo가 더 중요" 등 방향 분석
- Trivial solution → entropy reg, class-balanced weighting, L_sem weight 증가

## 데이터
- 성수동 드론 이미지 180장 (oblique, 70m, 원본 8192x5460 → 2048x1365 리사이즈)