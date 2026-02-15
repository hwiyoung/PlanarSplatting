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
| **f_i** | **3** | **의미론적 특징** | **L_sem, L_mutual** |

### 파라미터-손실 매핑 (기본 설계: L_photo 미포함)

```
L_depth  ──→ c_i, r_i, R_i      (깊이 정확도)
L_normal ──→ R_i                  (법선 정확도)
L_geo    ──→ c_i, r_i, R_i      (기하 정규화)
L_sem    ──→ f_i                  (의미론 분류)
L_mutual ──→ R_i + f_i           (★ 이 둘만 연결)
```

### L_mutual이 연결하는 것과 연결하지 않는 것
- **연결하는 것:** R_i(→법선 n_i)와 f_i(→클래스 확률 p_c). 이 두 파라미터만.
- **연결하지 않는 것:** c_i(위치), r_i(크기), 기타. 이들은 L_mutual과 무관.
- "양방향"이란: L_mutual의 gradient가 R_i 방향과 f_i 방향 둘 다로 흐른다는 뜻.

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

## L_mutual 수식 및 Gradient 분석

### 수식
```
L_mutual = Σ_i [ p_wall(i) · L_vert(n_i) + p_roof(i) · L_horiz(n_i) + p_ground(i) · L_horiz(n_i) ]
```
- p_c(i) = softmax(f_i)[c] : 프리미티브 i의 클래스 c 확률
- n_i : R_i에서 유도된 법선 벡터
- e_z = [0, 0, 1] : 수직(중력) 방향
- L_vert(n) = (n · e_z)² : 법선이 수평이면 0 → 벽면에 적합
- L_horiz(n) = 1 - (n · e_z)² : 법선이 수직이면 0 → 지붕/지면에 적합

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
※ L_photo는 기본 설계에서 미포함. "L_photo 포함 여부" 섹션 참조.

---

## Warmup 전략

- iter < total/3 : λ_m = 0 (Θ_geo와 Θ_sem이 독립적으로 안정화)
- iter ≥ total/3 : λ_m를 선형 증가 (안정화 후 커플링)
- PCGrad/CAGrad 같은 gradient surgery 대신 시간적 분리로 충돌 회피

## Trivial Solution 방지
- 위험: L_mutual 최소화를 위해 p_wall을 줄이는 것이 n_i를 고치는 것보다 쉬움 (semantic evasion)
- 방어: L_sem이 독립적으로 2D GT와의 일치를 강제 → p_wall이 부적절하게 줄면 L_sem이 복원
- 추가: entropy regularization, class-balanced weighting (필요 시)

---

## Ablation 설계

| 조건 | 설정 | 검증하려는 것 |
|------|------|-------------|
| (a) No mutual | λ_m = 0 | baseline |
| (b) Full mutual | λ_m with warmup | 양방향 상호 보강 효과 |
| (c) Sem→Geo only | softmax(f_i).detach() in L_mutual | R_i만 gradient 받음. 의미론→기하 단방향 |
| (d) Geo→Sem only | n_i.detach() in L_mutual | f_i만 gradient 받음. 기하→의미론 단방향 |

기대: (b)가 (a) 대비 기하+의미론 모두 개선, (b)의 개선 > (c) 단독 + (d) 단독 → 시너지

결과가 기대와 다를 경우:
- 효과 없음 → "조건과 한계" 분석 (MVS depth가 이미 충분히 정확하여 L_mutual의 추가 보강 여지가 적음 등)
- 한 방향만 효과 → "항공 데이터에서는 Sem→Geo가 더 중요" 등 방향 분석
- Trivial solution → entropy reg, class-balanced weighting, L_sem weight 증가

## 데이터
- 성수동 드론 이미지 180장 (oblique, 70m, 원본 8192x5460 → 2048x1365 리사이즈)