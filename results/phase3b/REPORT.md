# Phase 3-B: Ablation Study 결과 보고

## 수행 일시
2026-03-05

## 수행 작업 요약

Phase 3-B는 L_mutual 손실 함수의 효과를 체계적으로 검증하는 ablation study이다. Phase 3-A에서 L_mutual을 구현하고 gradient check(14/14 PASS)로 양방향 gradient 흐름을 확인하였으므로, 이제 실제 학습에서 L_mutual이 기하학적 법선(Normal cos)과 의미론적 분류(mIoU)에 미치는 영향을 정량적으로 분리한다.

이전 Phase에서 확보한 전제조건:
- Phase 2-B: gradient 격리 검증 (L_sem→f_i only, L_depth/L_normal→geometry only)
- Phase 2-C: L_sem 독립 학습 baseline (Depth MAE=0.027, Normal cos=0.782, mIoU=0.810)
- Phase 3-A: L_mutual gradient check 14/14 PASS, warmup curriculum 구현

검증 대상:
1. **L_mutual 전체 효과**: (d) Joint vs (c) Independent
2. **방향별 기여**: (d) vs (e) Sem→Geo, (d) vs (f) Geo→Sem
3. **양방향 시너지**: (d) > max((e),(f)) 여부
4. **Warmup 필요성**: (d) vs (g) No-warmup

## 정량 지표

### 7조건 × 3지표 비교표 (GT: MVS depth/normal, seg_maps v10)

| 조건 | 구성 | Depth MAE | Δ(c) | Normal cos | Δ(c) | mIoU | Δ(c) |
|------|------|-----------|-------|-----------|-------|------|-------|
| **(a)** Geo Only | L_d+L_n+L_geo | 0.0275 ± 0.017 | +0.0000 | 0.7806 ± 0.033 | -0.0009 | — | — |
| **(b)** Sem Only | L_d+L_n+L_sem | 0.0266 ± 0.016 | -0.0009 | 0.7802 ± 0.032 | -0.0014 | 0.7835 ± 0.100 | -0.0024 |
| **(c)** Independent | L_d+L_n+L_geo+L_sem | **0.0274** ± 0.016 | 0 | **0.7816** ± 0.032 | 0 | **0.7859** ± 0.092 | 0 |
| **(d)** Joint | +L_mutual(full) | 0.0279 ± 0.017 | +0.0005 | 0.7823 ± 0.033 | **+0.0008** | 0.7725 ± 0.086 | -0.0134 |
| **(e)** Sem→Geo | +L_mutual(s→g) | 0.0272 ± 0.014 | -0.0003 | 0.7811 ± 0.033 | -0.0004 | 0.7834 ± 0.093 | -0.0025 |
| **(f)** Geo→Sem | +L_mutual(g→s) | 0.0279 ± 0.016 | +0.0005 | **0.7830** ± 0.033 | **+0.0014** | 0.7696 ± 0.097 | -0.0163 |
| **(g)** No Warmup | +L_mutual(no-wu) | 0.0285 ± 0.017 | +0.0011 | 0.7817 ± 0.033 | +0.0001 | 0.7669 ± 0.095 | -0.0190 |

### Per-class IoU

| 조건 | IoU Roof | IoU Wall | IoU Ground |
|------|----------|----------|-----------|
| (b) Sem Only | 0.6393 | 0.8176 | 0.8935 |
| (c) Independent | 0.6466 | 0.8190 | 0.8920 |
| (d) Joint | 0.6309 | 0.8080 | 0.8788 |
| (e) Sem→Geo | 0.6383 | 0.8203 | 0.8917 |
| (f) Geo→Sem | 0.6276 | 0.8082 | 0.8730 |
| (g) No Warmup | 0.6208 | 0.8044 | 0.8756 |

### Primitive 수

| 조건 | Planes |
|------|--------|
| (a) Geo Only | 2689 |
| (b) Sem Only | 2668 |
| (c) Independent | 2605 |
| (d) Joint | 2657 |
| (e) Sem→Geo | 2612 |
| (f) Geo→Sem | 2608 |
| (g) No Warmup | 2601 |

## 핵심 해석

### 1. L_mutual 전체 효과: (d)-(c)

- **Normal cos: +0.0008** — L_mutual이 법선 정확도를 미세하게 개선. 의미론적 클래스 확률이 기하학적 법선 방향 최적화에 약한 guidance를 제공.
- **mIoU: -0.0134** — 의미론적 분류 성능이 하락. L_mutual의 기하학적 prior가 2D GT 레이블과 충돌하는 경우가 있음. 특히 roof IoU(-0.0157)와 ground IoU(-0.0132) 하락이 두드러짐.
- **Depth MAE: +0.0005** — 보조 참고. c_i는 L_mutual 수식에 없으므로 간접 효과이며, 변화 폭이 std(0.017) 대비 매우 작아 유의미하지 않음.

**해석**: L_mutual은 Normal cos 개선 방향으로는 작동하나, 현재 설정에서 mIoU를 희생하는 trade-off가 관찰됨. 이는 L_mutual의 기하학적 prior(벽=수직, 지면=수평)가 seg_maps GT의 confident-labels-only 전략(ambiguous→BG)과 상충하기 때문으로 분석됨 — L_mutual이 ambiguous 영역의 프리미티브를 강하게 분류하려 하지만, GT에서는 해당 영역이 BG(ignore)이므로 L_sem과 상충.

### 2. 의미론→기하 방향 기여: (d)-(e) Normal cos

- **(d)-(e) Normal cos = +0.0012** — 양방향(d)이 단방향 Sem→Geo(e)보다 Normal cos가 높음. 의미론적 확률이 detach되지 않을 때 법선에 추가적인 gradient가 흐르는 효과.
- 그러나 (f) Geo→Sem의 Normal cos가 +0.0014로 (d)보다 높음 — 의미론을 detach하고 법선만 gradient를 받을 때가 양방향보다 법선에 더 유리.

### 3. 기하→의미론 방향 기여: (d)-(f) mIoU

- **(d)-(f) mIoU = +0.0029** — 양방향(d)이 단방향 Geo→Sem(f)보다 mIoU가 소폭 높음. 법선이 detach되지 않을 때 의미론에 약간의 보조 효과.
- 그러나 두 조건 모두 (c) Independent보다 mIoU가 낮아, L_mutual 자체가 mIoU에 부정적.

### 4. 양방향 시너지: (d) > max((e),(f))

- **Normal cos**: 0.7823 < max(0.7811, 0.7830) = 0.7830 → **NO** (f가 더 높음)
- **mIoU**: 0.7725 < max(0.7834, 0.7696) = 0.7834 → **NO** (e가 더 높음)

**양방향 시너지는 현재 실험 조건에서 입증되지 않음.** 이는 다음 가능성을 시사:
1. λ_mutual=0.05가 과도하여 기하-의미론 간 gradient 간섭 발생
2. Confident-labels-only GT에서 L_mutual과 L_sem 간 목표 불일치
3. 5000 iterations가 양방향 조화에 불충분 (warmup 이후 실질 joint training은 ~1650 iters)

### 5. Warmup 필요성: (g)-(d)

- **Normal cos: -0.0007** — warmup 없이 즉시 L_mutual 적용 시 법선 품질이 소폭 하락
- **mIoU: -0.0056** — warmup 없이 mIoU가 더 크게 하락
- **Depth MAE: +0.0007** — depth도 소폭 악화

**Warmup이 학습 안정성에 기여**: 초기에 기하학적 구조가 충분히 수렴하기 전에 L_mutual을 적용하면 gradient 간섭이 더 심해짐. (d) 대비 (g)의 전 지표 하락이 이를 확인.

### 6. L_geo 효과: (c)-(b) — 부수 관찰

- **Normal cos: +0.0014** — L_geo(normal consistency) 추가가 법선 품질을 개선
- **mIoU: +0.0024** — L_geo가 기하학적 일관성을 강화하여 의미론 분류에도 간접적 이득

## 정성적 결과

각 조건별 9개 뷰(View 0, 12, 25, 37, 50, 62, 75, 87, 99)에 대해 Depth, Normal, Semantic(해당 시) 렌더링을 `images/{a~g}/` 에 저장하였다. PLY 파일은 `ply/{a~g}_{normal,class}.ply`에 저장.

### 주요 관찰 (렌더링 이미지 기반)

1. **Depth 맵**: 모든 조건에서 유사한 품질. MAE 차이가 0.002 범위로 시각적 구분 어려움.
2. **Normal 맵**: (d) Joint와 (f) Geo→Sem에서 건물 벽면의 법선이 (c)보다 약간 더 균일 — L_mutual의 "벽=수직" prior 효과. 그러나 차이가 미세함.
3. **Semantic 맵**: (c) Independent가 GT와 가장 유사. (d) Joint에서 roof-wall 경계 영역에서 일부 오분류 증가 — L_mutual이 기하학적으로 ambiguous한 경사면을 wall로 강제하는 경향.
4. **3D PLY**: (d)의 class PLY에서 wall 프리미티브의 법선 일관성이 (c)보다 개선된 것이 확인됨. 그러나 roof 영역에서 wall로 오분류된 프리미티브가 관찰됨.

### 렌더링 파일 경로

- Depth/Normal 비교: `images/{c,d,f}/depth_view{00-08}.png`, `images/{c,d,f}/normal_view{00-08}.png`
- Semantic 비교: `images/{c,d,e,f}/semantic_view{00-08}.png`
- GT RGB 참고: `images/{a-g}/rgb_view{00-08}.png`
- 3D Class PLY: `ply/{c,d,e,f}_class.ply`

## Go/No-Go 판단

- [ ] Go / [x] Retry / [ ] Switch

### 판단 근거

**Go 기준**: "Joint(d)가 Independent(c) 대비 Normal cos **또는** mIoU 중 하나 이상에서 개선, 나머지 지표 유의미한 악화 없음."

- Normal cos: +0.0008 (개선, 그러나 std=0.033 대비 매우 작아 통계적 유의성 부족)
- mIoU: -0.0134 (유의미한 악화, Phase 2-C의 0.810 대비 하락)
- 양방향 시너지 미입증: (d) < max((e),(f))

**엄밀한 Go 기준을 충족하지 못함.** Normal cos 개선이 관찰되나, mIoU 악화와 시너지 미입증으로 인해 현재 상태에서 논문 핵심 클레임("양방향 상호 보강")을 지지하기 어려움.

### Retry 방향

1. **λ_mutual 조정**: 현재 0.05 → 0.01~0.02로 감소. L_mutual의 기하학적 prior 강도를 낮추어 L_sem과의 충돌 완화.
2. **Iteration 증가**: 5000 → 10000. Warmup 이후 실질 joint training 시간 확보.
3. **L_slope τ 조정**: 현재 0.15 → 0.3. 경사 지붕에 대한 penalty 완화.
4. **GT 개선**: confident-labels-only의 ambiguous 영역(51.2%)이 L_mutual 효과를 제한할 가능성. Coverage가 높은 GT를 사용하면 L_mutual의 교정 효과가 더 명확히 드러날 수 있음.

## 생성/수정 파일 목록

| 파일 | 유형 | 핵심 변경 |
|------|------|----------|
| `utils_demo/ablation_{a..g}_*.conf` | 수정 | expname을 조건별 고유명으로 업데이트 |
| `scripts/compare_ablation.py` | 신규 | 여러 eval JSON → 비교 표(CSV+터미널) + 시너지 검증 |
| `scripts/eval_phase3b.sh` | 신규 | 전체 조건 일괄 평가 (evaluate + render + PLY) |
| `scripts/run_phase3b_ablation.sh` | 수정 | checkpoint 경로 수정 ({expname}_{scan_id} 패턴) |
| `scripts/render_views.py` | 수정 | Semantic 맵 렌더링 추가 (enable_semantic 시) |
| `results/phase3b/eval/abl_{a..g}.json` | 생성 | 조건별 정량 평가 결과 |
| `results/phase3b/images/{a..g}/` | 생성 | 조건별 렌더링 이미지 (9 views × Depth/Normal/Semantic/RGB) |
| `results/phase3b/ply/{a..g}_{normal,class}.ply` | 생성 | 조건별 3D 프리미티브 export |
| `results/phase3b/ablation_comparison.csv` | 생성 | 비교 표 CSV |

## 이슈 및 해결

1. **Checkpoint 경로 문제**: `run_demo_colmap.py`가 `{expname}_{scan_id}/{timestamp}/` 구조로 디렉토리를 생성하나, 초기 run script에서 `{expname}/` 만으로 검색하여 checkpoint를 찾지 못함. `_${SCAN_ID}` 접미사를 추가하여 해결.
2. **L_mutual과 mIoU 충돌**: L_mutual의 기하학적 prior가 confident-labels-only GT와 상충. 이는 설계상의 이슈로, λ_mutual 튜닝 또는 GT 전략 변경이 필요.

## TensorBoard 비교 방법

```bash
# 모든 ablation 조건의 TensorBoard를 동시에 비교
tensorboard --logdir_spec=\
a:planarSplat_ExpRes/phase3b/abl_a_geo_only_example/*/tensorboard,\
b:planarSplat_ExpRes/phase3b/abl_b_sem_only_example/*/tensorboard,\
c:planarSplat_ExpRes/phase3b/abl_c_independent_example/*/tensorboard,\
d:planarSplat_ExpRes/phase3b/abl_d_joint_example/*/tensorboard,\
e:planarSplat_ExpRes/phase3b/abl_e_sem2geo_example/*/tensorboard,\
f:planarSplat_ExpRes/phase3b/abl_f_geo2sem_example/*/tensorboard,\
g:planarSplat_ExpRes/phase3b/abl_g_no_warmup_example/*/tensorboard \
--port 6006
```

`loss/total`, `loss/semantic`, `loss/mutual`을 overlay하면 조건 간 수렴 속도와 최종 손실 비교 가능. `compare/4_semantic`에서 의미론 맵 품질 시각 비교.

## 다음 Phase

### 즉시 실행: Retry (하이퍼파라미터 조정)

현재 결과는 L_mutual의 메커니즘이 작동함을 보여주지만 (Normal cos 개선 방향), λ_mutual과 학습 스케줄 튜닝이 필요하다. 다음을 순차적으로 시도:

1. **λ_mutual = 0.01** (현재 0.05의 1/5): L_mutual의 강도를 줄여 L_sem과의 균형 개선
2. **max_total_iters = 10000**: warmup 이후 충분한 joint training 확보
3. 위 조정 후 (d) Joint만 재실행하여 (c)와 비교

Retry 결과가 Go 기준을 충족하면 Phase 3-C (L_photo 추가) 또는 Phase 4 (Building Grouping + CityGML)로 진행.
