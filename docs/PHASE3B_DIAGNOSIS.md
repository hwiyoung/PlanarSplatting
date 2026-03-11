# Phase 3-B 결과 진단 및 연구 방향 재검토

## 1. 실험 결과 요약

Phase 3-B ablation study에서 L_mutual(양방향 기하-의미론 상호 보강 손실)의 효과를 7개 조건으로 검증하였으나, 핵심 가설인 **"양방향 시너지"**가 입증되지 않았다.

### 정량 결과 (baseline = (c) Independent)

| 조건 | Normal cos | Δ(c) | mIoU | Δ(c) |
|------|-----------|-------|------|-------|
| (c) Independent (L_d+L_n+L_geo+L_sem) | 0.7816 | — | 0.7859 | — |
| (d) Joint (+L_mutual full) | 0.7823 | +0.0008 | 0.7725 | **-0.0134** |
| (e) Sem→Geo (detach semantics) | 0.7811 | -0.0004 | 0.7834 | -0.0025 |
| (f) Geo→Sem (detach geometry) | 0.7830 | +0.0014 | 0.7696 | -0.0163 |
| (g) No Warmup | 0.7817 | +0.0001 | 0.7669 | -0.0190 |

- **Normal cos**: L_mutual로 미세 개선 (+0.0008) 가능하나, std=0.033 대비 유의미하지 않음
- **mIoU**: 모든 L_mutual 조건에서 하락 (최대 -0.019)
- **시너지 검증**: (d) > max((e),(f)) → **양쪽 모두 NO**

---

## 2. 근본 원인 분석

### 원인 1: L_mutual이 해결할 오류가 없다

| 과제 | 현재 supervision | 수렴 수준 | L_mutual 기여 여지 |
|------|-----------------|----------|-------------------|
| 기하학 (법선) | MVS normal (1차 측정, 91% coverage) | Normal cos 0.782 | **거의 없음** — MVS가 이미 강하게 supervision |
| 의미론 (분류) | L_sem (3D multi-view consistency) | mIoU 0.786 | **거의 없음** — multi-view vote가 이미 교정 |

L_mutual의 "벽=수평" prior는 MVS normal supervision과 **중복**. L_mutual의 "수평이면 벽" prior는 L_sem의 multi-view 교정보다 **약함**.

→ **L_mutual은 정보 위계에서 최하위**에 위치하여, 개선보다 간섭을 유발.

### 원인 2: GT coverage 구조적 충돌

```
L_mutual: 모든 N개 프리미티브에 작용 (per-primitive, 마스크 없음)
L_sem:    GT가 있는 49%의 영역에만 작용 (ignore_index=0, 51% ambiguous→BG)
```

- L_mutual이 ambiguous 영역(51%)의 프리미티브를 특정 클래스로 push
- L_sem은 해당 영역에서 아무 feedback도 못 줌
- 결과: L_mutual이 일방적으로 의미론을 왜곡 → mIoU 하락의 직접 원인

### 원인 3: 양방향 시너지 성립 전제조건 미충족

양방향 시너지가 작동하려면:
1. **기하 오류를 의미론이 교정 가능해야 함** → MVS supervision이 워낙 강해서 의미론의 약한 prior가 영향을 주지 못함
2. **의미론 오류를 기하가 교정 가능해야 함** → L_sem + multi-view가 이미 충분히 교정
3. **두 방향의 교정 빈도가 비슷해야 시너지** → 현재는 양쪽 다 교정할 것이 없음

---

## 3. 현재 가설과 핵심 기여 재검토

### 원래 가설
> "기하학적 법선(R_i)과 의미론적 분류(f_i)가 L_mutual을 통해 서로를 개선하는 양방향 상호 보강"

### 가설의 전제 (implicit assumptions)
1. 기하학적 supervision이 불완전하여 의미론이 보완할 여지가 있다
2. 의미론적 supervision이 불완전하여 기하가 보완할 여지가 있다
3. L_mutual이 이 두 가지 보완을 동시에 달성할 수 있다

### 실험이 보여준 것
- 전제 1 불성립: MVS depth+normal이 이미 강한 supervision
- 전제 2 부분 성립: ambiguous 영역(51%)에서 의미론 GT 부재 → 기하 기반 pseudo-labeling 가능
- 전제 3 부분 성립: 단, 현재 구현에서는 GT 부재 영역에서의 기여와 GT 존재 영역에서의 간섭이 상쇄

### 실제로 확인된 기여
- **L_geo(normal consistency)의 효과**: (c)-(b) Normal cos +0.0014, mIoU +0.0024 ← 실질적 기여
- **3D primitive + multi-view L_sem**: mIoU 0.786 달성 ← 구조적 장점
- **L_mutual의 기하 방향**: (f) Normal cos +0.0014 ← 약한 신호 존재

---

## 4. 대응 방향 (2026-03-11 확정)

지도교수 피드백에 따라 연구 방향을 조정하였다. 상세: `docs/ADVISOR_FEEDBACK_RESPONSE.md`

### 핵심 결정
1. **L_photo를 기본 설정에 포함**: 기존 Phase 3-B는 "L_photo 미포함 예비 분석"으로 재위치
2. **Phase 4(CityGML 변환 + mesh 보정) 최우선**: 파이프라인 완성이 핵심 목표
3. **L_mutual은 파이프라인의 한 구성요소**: 효과와 한계를 분석하는 대상
4. **기여 방향 3가지 병행 탐색** (실험 결과에 따라 확정):
   - 방향 A: 평면 프리미티브 + 의미론 통합 → CityGML LOD2 (City3D 비교)
   - 방향 B: 의미론-기하학 동시 최적화 (L_mutual, L_photo 포함 조건 재실험)
   - 방향 C: TSDF mesh watertight 특성 → splatting 토폴로지 보정

### 본 진단의 활용
- 원인 1~3 분석은 Phase 3-B' 재실험 설계의 근거로 유지
- 특히 원인 2(GT coverage 충돌) → (d'-masked) 조건 설계에 반영
- 원인 1(해결할 오류 부재) → L_photo 추가가 기하 supervision 경관을 바꿀 가능성 검증

---

## 5. 관련 파일 목록
- 실험 결과: `results/phase3b/REPORT.md`, `results/phase3b/eval/abl_*.json`
- 연구 설계: `docs/RESEARCH_CONTEXT.md`, `docs/EXPERIMENT_PLAN.md`
- 방향 조정: `docs/ADVISOR_FEEDBACK_RESPONSE.md`
- L_mutual 구현: `planarsplat/utils/loss_util.py` (mutual_loss, L56-114)
- Trainer 통합: `planarsplat/run/trainer.py` (L555-572)
- Segmentation GT: `scripts/generate_segmentation.py` (confident-labels-only 전략)
- 전체 진행: `results/SUMMARY.md`, `CLAUDE.md`
