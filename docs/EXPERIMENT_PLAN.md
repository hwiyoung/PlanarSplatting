# 실험계획 (v6 — Synthetic/Real 구조, 3D BAG 기반)

## 전체 구조

```
Phase 1: Synthetic (3D BAG, GT 있음, 제어 가능)
  ├── Synthetic A: Stage 3 단독 검증
  └── Synthetic B: Stage 2+3 통합 검증 (L_mutual ablation)

Phase 2: Real (ISPRS + 성수동, GT 제한적)
  ├── 전체 파이프라인 적용
  └── City3D 비교 (동일 데이터에서 공정 비교)
```

### Phase 간 관계

```
Synthetic A: "Stage 3이 요구하는 프리미티브 품질은?"
  → 각 요인의 허용 범위 도출 + 가장 민감한 요인 식별
  → Stage 3 독립 검증 (설계 선택 3)
    ↓ (Stage 2의 목표 설정 + B 실패 시 원인 진단 기준)
Synthetic B: "L_mutual이 그 품질을 달성하여 CityGML 생성을 가능하게 하는가?"
  → Stage 2 학습 + L_mutual ablation → Stage 3 end-to-end 검증
  → 출력 프리미티브의 품질을 Synthetic A 임계값과 대조
    ↓
Real: "실제 데이터에서도 작동하는가?"
  → ISPRS (학술 표준) + 성수동 (연구 대상 지역)
  → City3D와 동일 데이터에서 공정 비교
```

### 논문 설계 선택과의 연결

| 설계 선택 | 검증 실험 |
|----------|---------|
| 1: 평면 프리미티브 + 의미론 통합 동시 최적화 | Synthetic B (Joint vs Baseline) |
| 2: 도메인 지식의 미분 가능 인코딩 (L_mutual) | Synthetic B (±L_mutual → CityGML 품질) |
| 3: 의미론적 프리미티브에서 폐합 건물 모델 구성 | Synthetic A (프리미티브 → CityGML 허용 범위) |

## 현재 상태

| 단계 | 상태 | 비고 |
|------|------|------|
| Stage 1 (SfM/MVS) | 완료 | COLMAP 180→100장, Grounded SAM GT |
| Stage 2 예비 실험 | 완료 | gravity 미보정 + L_photo 미포함 |
| Stage 2 L_photo 구현 | 완료 | 구현 완료 |
| Stage 3 알고리즘 (2.5D Hybrid) | 완료 | convex hull fallback + make_valid 개선 |
| **Synthetic A (3D BAG)** | **완료** | 543건물 × 17조건, 법선 지배적 요인 확인 |
| **Synthetic B** | **다음** | L_mutual → CityGML 검증 |

## 평가 체계 (ISO 19157 기반)

### CityGML LOD2 품질 측면

| 품질 측면 (ISO 19157) | 정의 | 평가 지표 |
|---------------------|------|---------|
| **구조적 품질** (기하학적 유효성 + 위상적 일관성) | 폐합된 유효한 3D solid인가, 면들이 올바르게 연결되는가 | **val3dity 통과율** |
| **형상 정확도** (positional accuracy) | GT 형상과 얼마나 가까운가 | **Chamfer distance** |
| **의미론적 정확성** (thematic accuracy) | 면 라벨(Roof/Wall/Ground)이 올바른가 | **Semantic accuracy** |
| **완전성** (completeness) | scene 내 건물 중 몇 %가 성공적으로 생성되었는가 | **Stage 3 성공률** |

- **val3dity**: ISO 19107 기반. 기하학적 유효성(폐합, 평면성, 면 방향)과 위상적 일관성(manifold, edge 공유)을 동시에 검사. 분리 측정 불가.
- **Chamfer distance**: 양방향 평균 표면 거리. Hausdorff(최악 거리) 대비 이상치에 안정적. GT와 동일 형상이면 0.
- **Semantic accuracy**: GT 면과 법선+중심 유사도로 매칭 후 라벨 일치율.

### 노이즈 요인과 품질 측면의 관계

노이즈 요인은 CityGML 품질 측면에서 **"이것에 영향을 줄 수 있는 프리미티브 속성은 무엇인가?"**로 도출:

```
구조적 품질 ← 법선(면 방향), 위치(면 연결), 누락(면 형성), 아웃라이어(잘못된 면), 면적(plane eq. 가중치)
의미론적 정확성 ← 분류(면 라벨)
```

각 노이즈는 **여러 품질 측면에 동시에 영향** 가능 (다:다 관계). 실험에서는 각 노이즈를 부여하고 **모든 품질 지표를 측정**하여 교차 영향을 파악한다.

---

## Phase 1: Synthetic (3D BAG)

### Synthetic A: Stage 3 단독 검증

**목적**: Stage 3 알고리즘이 어떤 유형/수준의 프리미티브 오류까지 유효한 CityGML을 생성할 수 있는가? 이 결과로 Stage 2의 최적화 목표를 수치로 설정한다.

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Synthetic A를 진행해줘. 컨테이너 내부에서 작업이야.

목표: Stage 3 알고리즘의 입력 허용 범위 파악.
3D BAG 실제 건물에서 어떤 유형/수준의 프리미티브 오류까지 유효한 CityGML을 생성할 수 있는가?

=== 실행 ===
scripts/stage3_synthetic/run_3dbag_experiment.py 실행.
- 3D BAG 3개 scene (Amsterdam/Rotterdam/Delft) 에서 roof type별 층화 추출 (~610 건물)
- 17개 노이즈 조건 (단일 14 + 복합 2 + clean)
- 평가: val3dity + Chamfer + Semantic Accuracy
- 결과: results/stage3_synthetic_a/3dbag_results.json

=== 분석 ===
1. 노이즈 × 품질 교차표: 어떤 노이즈가 어떤 품질에 영향?
2. 허용 범위 도출: 각 노이즈의 임계값
3. 민감도 순위: 가장 영향력 있는 요인
4. 복합 검증: 가장 민감한 요인만 관리하면 충분한가 (충분조건)?
5. 건물 유형별 차이: roof type에 따라 허용 범위가 다른가?
6. Stage 2 최적화 목표 설정

=== 산출물 ===
1. 3D BAG scene 전체 뷰 (XY footprint + 3D 샘플)
2. 노이즈 × 품질 교차표 (val3dity / Chamfer / Semantic Acc)
3. 민감도 순위 그래프
4. 복합 vs 단일 비교 ("충분조건 검증")
5. 건물 유형별 비교
6. 대표 건물 GT vs 결과 비교 (CityJSON Ninja)

results/stage3_synthetic_a/REPORT.md 작성. CLAUDE.md 업데이트.
```

#### 데이터: 3D BAG

**3D BAG** (네덜란드 전국 LOD2 CityJSON, TU Delft). val3dity 검증된 GT.

선택 이유:
- **CityJSON 형식**: 파이프라인과 호환 (CityJSON Ninja로 시각화 가능)
- **val3dity 검증된 GT**: TU Delft가 품질 관리 → GT로 신뢰 가능
- **건물 유형 메타데이터**: roof type, 높이, 면적 등 포함 → 유형별 분석 가능
- **대규모**: ~1,000만 건물 → 필요한 만큼 샘플링 가능

3개 scene (도시 유형별 다양성 확보):
- **Amsterdam Jordaan**: 역사적 canal house, 경사 지붕 다수
- **Rotterdam Center**: 현대 상업 건물, 평지붕/대형 건물
- **Delft Residential**: 주거 단지, row house

건물 필터링 과정:

| 단계 | Amsterdam | Rotterdam | Delft | 합계 | 설명 |
|------|----------|----------|-------|------|------|
| 타일 내 CityObjects | 5,926 | 2,048 | 9,840 | 17,814 | Building + BuildingPart |
| Building 객체 | 2,956 | 1,020 | 4,914 | **8,890** | BuildingPart는 하위 요소 |
| 면적 ≥ 10m² | -34 | -47 | -1,382 | -1,463 | 매우 작은 구조물 제외 |
| 면수 ≤ 200 | -32 | -30 | -17 | -79 | 극도로 복잡한 건물 제외 |
| **파싱 통과** | **2,890** | **943** | **3,515** | **7,348** | |
| Stage 3 clean 성공 (추정) | ~2,600 (90%) | ~810 (86%) | ~3,440 (98%) | **~6,850** | ~93% |

- **8,890 → 7,348 (파싱 필터)**: 극소 구조물(< 10m²)과 극복잡 건물(> 200면)을 제외. 이는 Stage 2에서도 프리미티브를 생성하기 어려운 건물.
- **7,348 → ~6,850 (Stage 3 성공)**: ~7% 실패는 Stage 3 알고리즘의 한계 (좁은 건물, 복잡한 dormer/다단 지붕). 기존 CityGML 생성 방법(외부 footprint 제공 시 ~98%)과 비교하여, 외부 데이터 없이 프리미티브만으로 자동 생성하는 trade-off.
- **완전성**: clean에서 Stage 3이 성공하는 비율(~93%)은 **알고리즘 자체의 완전성**. 노이즈 실험에서는 clean 성공 건물만 대상으로 하므로 clean 완전성은 100%. 노이즈로 인해 Stage 3이 **추가 실패**하는 비율이 실험에서 측정하는 완전성.

#### 샘플링

모집단(~6,850 valid 건물) 전체 테스트 시 ~35시간 소요. **roof type별 층화 추출 (~610개)**로 유형별 분석이 가능한 수준의 결과 확보.

샘플링 방법:

1. **모집단 선정**: 7,348개 파싱 통과 건물 중 clean(n=30)에서 Stage 3 + val3dity 성공하는 건물만 대상. 실패 건물은 노이즈와 무관하게 실패하므로 노이즈 실험 대상에서 제외.

2. **층화 추출 (roof type별)**: 노이즈 허용 범위가 건물 유형에 따라 다를 수 있으므로 (경사 지붕은 ridge 계산 필요 → 법선에 더 민감할 수 있음), 유형별로 충분한 수를 확보.

| Roof type | 모집단 (파싱) | Valid 추정 | 샘플 | 95% CI (비율 85% 가정) |
|-----------|------------|----------|------|---------------------|
| slanted | 6,044 | ~5,600 | 200 | ±5.0% |
| horizontal | 1,074 | ~1,000 | 200 | ±5.0% |
| multiple horizontal | 226 | ~210 | **전부 (~210)** | 수가 적으므로 전부 사용 |
| **합계** | 7,348 | ~6,850 | **~610** | |

3. **랜덤 추출**: 각 roof type의 valid 건물에서 목표 수만큼 랜덤 샘플링 (seed=42, 재현 가능). 3 scene에서 고르게 분포.

4. **결과 보고**: 유형별 결과를 각각 보고하고, 전체 통과율은 모집단 비율로 **가중 평균** (slanted 82% : horizontal 15% : multiple 3%).

근거: 이항분포 정규근사, `CI = p ± 1.96 × √(p(1-p)/n)`. 유형별 200개이면 ±5.0%로, 유형 간 차이(예: slanted 85% vs horizontal 95%)를 식별 가능.

#### 실험 논리

```
1. Clean → baseline (알고리즘 자체 성능)

2. 단일 요인 (14개)
   → CityGML 품질에 영향 가능한 프리미티브 속성을 개별 노이즈로 테스트
   → 각 노이즈에 대해 모든 품질 지표 교차 측정 (다:다 관계)
   → "어떤 요인이 가장 민감한가?" 식별

3. 복합 (2개)
   → 사전조건: 단일 요인 결과에서 가장 민감한 요인 식별
   → 해당 요인의 임계 수준을 고정, 나머지 전부 최악
   → "해당 요인만 관리하면 충분한가 (충분조건)?"
```

#### 노이즈 조건 (17개)

##### 단일 요인 (14개)

노이즈 요인은 CityGML 품질에 영향 가능한 프리미티브 속성에서 도출.
각 노이즈는 **여러 품질 측면에 동시에 영향 가능**하므로,
모든 품질 지표를 교차 측정하여 실제 영향을 파악.

| 요인 | 조건 | 구체적 내용 | 실험 의도 |
|------|------|-----------|---------|
| **법선** | normal_2° | 접선 평면에서 σ=2° 등방 회전 | 법선 민감도 하한 |
| | normal_10° | σ=10° | Stage 2 목표 수준 (임계점 탐색) |
| | normal_20° | σ=20° | 실패 수준 확인 |
| **위치** | pos_iso_0.5m | center XYZ 각축 σ=0.5m Gaussian | 위치 오차 영향 |
| | pos_iso_1.0m | σ=1.0m | |
| **분류** | cls_15% | 프리미티브 15%의 class 랜덤 교체 | 분류 오류 영향 |
| | cls_30% | 30% | |
| **누락** | missing_30% | 프리미티브 30% 랜덤 제거 | 커버리지 부족 영향 |
| | missing_50% | 50% | |
| **아웃라이어** | outlier_5% | 5%의 법선/위치를 극단적 변경 | 이상치 강건성 |
| | outlier_10% | 10% | |
| **면적** | area_30% | 각 프리미티브 area에 σ=30% log-normal 스케일 | 면적 가중치 오차 영향 |
| | area_50% | σ=50% | |
| | area_100% | σ=100% | |

##### 복합 (2개)

단일 요인에서 가장 민감한 요인(가설: 법선)의 임계 수준을 고정하고,
나머지 요인을 **전부 최대(최악)** 수준으로 설정.

| 조건 | 구성 | 실험 의도 |
|------|------|---------|
| **N10_worst** | N10°+P1.0+C30+M50+O10+A100 | 법선 10°가 **충분조건**인가? 나머지 전부 최악에서도 결과 유지? |
| **N2_worst** | N2°+P1.0+C30+M50+O10+A100 | 법선 2°에서도 동일한가? (보강) |

결과 해석:
- N10_worst ≈ normal_10° 단독 → "법선만 관리하면 충분" (충분조건 확인)
- N10_worst << normal_10° → "법선 외 추가 관리 필요" (필요조건만)

#### 평가 지표 (ISO 19157 기반)

각 노이즈 조건 × 모든 평가 지표를 교차 측정:

| 품질 측면 | 지표 | 측정 방법 |
|----------|------|---------|
| **구조적 품질** | val3dity 통과율 | ISO 19107 기하학적+위상적 검증 (동시 검사, 분리 불가) |
| **형상 정확도** | Chamfer distance | 양방향 평균 표면 거리 (GT 동일 형상 = 0m) |
| **의미론적 정확성** | Semantic accuracy | GT 면과 법선+중심 매칭 후 라벨 일치율 |
| **완전성** | Stage 3 성공률 | 전체 건물 중 CityGML 생성 성공 비율 |

#### 실행

```bash
# 컨테이너 내부에서 실행
docker exec -it planarsplat bash -c \
  "cd /workspace/PlanarSplatting && python3 scripts/stage3_synthetic/run_3dbag_experiment.py"

# 진행 확인
docker exec planarsplat tail -f /tmp/3dbag_experiment.log
```

**실행 완료**: 2026-03-27, 543건물 × 17조건 = 9,231회, ~23분 소요.

#### 산출물

- `results/stage3_synthetic_a/3dbag_results.json`: 전체 결과 (543건물 × 17조건 = 9,231 entries)
- `results/stage3_synthetic_a/3dbag_sampled_buildings.json`: 샘플링된 건물 목록
- `results/stage3_synthetic_a/REPORT.md`: 분석 보고서
- 시각적 산출물 (`results/stage3_synthetic_a/images/`):
  1. `noise_quality_cross.png`: 노이즈 × 품질 교차 분석 (val3dity / Chamfer / Semantic Acc)
  2. `sensitivity_ranking.png`: 민감도 순위
  3. `combined_vs_single.png`: 복합 vs 단일 비교
  4. `roof_type_comparison.png`: 건물 유형별 비교
  5. `structure_vs_semantic.png`: 구조적 품질 vs 의미론적 정확성 독립성
- Scene 뷰: `scene_amsterdam_jordaan_full.png`, `scene_rotterdam_center_full.png`, `scene_delft_wijk_full.png`

---

### Synthetic B: Stage 2+3 통합 검증

**목적**: L_mutual이 CityGML 생성을 가능하게 하는가? L_mutual 없이는 Synthetic A의 품질 임계값을 달성할 수 없고, L_mutual이 있으면 달성하여 유효한 CityGML이 생성됨을 검증한다.

**데이터**: Synthetic A와 동일한 3D BAG scene.

**선행 조건**: Synthetic A 완료 (허용 범위 파악).

#### Stage 2 실험 조건

| 이름 | 구성 | 의미 |
|------|------|------|
| **Baseline** | L_d + L_n + L_geo + L_sem + L_photo (λ_m=0) | 독립 최적화 |
| **Joint** | Baseline + L_mutual 4항, warmup | 공동 최적화 |
| **Joint-GTOnly** | Joint, GT 영역 프리미티브만 | 경로 1 제거 (조건부) |
| **Joint-Weak** | Joint, λ_m=0.01 | gradient 크기 진단 (조건부) |

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Synthetic B를 진행해줘. 컨테이너 내부에서 작업이야.

=== Part A: 합성 렌더링 ===
Synthetic A의 3D BAG scene을 가상 카메라로 렌더링:
1. 카메라 배치: 항공 oblique 시뮬레이션 (고도 ~70m, 다양한 각도)
2. 렌더링 출력: depth map, normal map, segmentation map
   - RGB는 불필요 (L_photo 제외, L_mutual 검증에 불필요)
   - depth/normal/seg는 geometry + semantic에서 직접 생성
3. COLMAP 형식으로 변환 (Stage 2 입력 호환)

=== Part B: Stage 2 학습 ===
2×2 ablation:
| 조건 | L_mutual | 감독 신호 |
|------|---------|---------|
| (a) Baseline + Clean | OFF | clean |
| (b) Joint + Clean | ON | clean |
| (c) Baseline + Noisy | OFF | noisy |
| (d) Joint + Noisy | ON | noisy |

감독 신호 노이즈: 다양한 depth noise 수준으로 Stage 2를 학습하고,
출력 프리미티브의 법선 σ를 측정하여 Synthetic A의 임계값과 대조.
(감독 신호 → 프리미티브 품질의 관계는 사전에 알 수 없으므로, 실험적으로 파악.)

각 조건에서 Stage 2 학습 → 프리미티브 출력

=== Part C: Stage 3 실행 ===
각 조건의 프리미티브 → Stage 3 → CityGML 생성

=== Part D: 평가 ===
1차 (렌더링 지표): Depth MAE, Normal cos, mIoU
2차 (프리미티브 지표): wall 법선 σ → Synthetic A 임계값 대조
3차 (CityGML 지표): val3dity, Chamfer, Semantic Acc

핵심 비교:
- (b) vs (a): L_mutual의 추가 가치 (clean 조건)
- (d) vs (c): L_mutual의 필수성 (noisy 조건) — 이것이 핵심
- (c) vs (a): 감독 신호 노이즈의 영향
- (d) vs (b): L_mutual이 노이즈를 얼마나 흡수하는가

=== 시각적 산출물 ===
1. 2×2 ablation 결과 비교표 + 그래프
2. L_mutual ON vs OFF CityGML side-by-side
3. wall 법선 분포 히스토그램 (Synthetic A 임계선 표시)
4. 전체 파이프라인 시각화: scene → depth/seg → 프리미티브 → CityGML

results/synthetic_b/REPORT.md 작성. CLAUDE.md 업데이트.
```

---

## Phase 2: Real (ISPRS + 성수동)

**목적**: Synthetic에서 검증된 파이프라인이 실제 데이터에서도 작동하는가?

**데이터**:
- **ISPRS benchmark** (Vaihingen/Potsdam): 학술 표준, 다른 방법과 공정 비교
- **성수동**: 연구 대상 지역, 드론 항공 이미지

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 2 (Real)를 진행해줘. 컨테이너 내부에서 작업이야.

=== Part A: ISPRS 데이터 준비 ===
1. ISPRS Vaihingen/Potsdam 데이터 다운로드
2. 항공 이미지 → COLMAP SfM/MVS
3. Grounded SAM segmentation 생성
4. Stage 2 입력 형식으로 변환

=== Part B: Stage 2 학습 ===
Synthetic B에서 확정된 최선 조건으로 학습:
- ISPRS 데이터
- 성수동 데이터

=== Part C: Stage 3 실행 ===
Stage 2 출력 → Stage 3 → CityGML 생성

=== Part D: 평가 ===
GT CityGML이 없으므로 Chamfer distance / Semantic accuracy 직접 측정 불가.
대신:
- val3dity 통과율 + 완전성 (건물 단위)
- 가용 외부 참조와 비교:
  · ISPRS: GT 건물 모델 (제공되는 경우, 있으면 Chamfer/Semantic 측정)
  · 성수동: 국토부 footprint IoU + 건축물대장 높이 비교
- Synthetic A 예측과 실측 비교: Stage 2 프리미티브 품질 → Synthetic A 결과표에서 예상 val3dity 확인

=== Part E: City3D 비교 ===
동일 COLMAP MVS 점군에서 City3D 실행 → 동일 평가 지표로 비교.
City3D는 점군 → CityGML 파이프라인이므로, 같은 입력에서 공정 비교 가능.
1. COLMAP dense point cloud → City3D 입력
2. City3D CityGML 출력
3. val3dity + 외부 참조 + 처리 시간 비교
4. 정성적 비교: 의미론 통합(우리) vs 사후 부여(City3D)

=== 시각적 산출물 ===
1. 입력 이미지 → 프리미티브 → CityGML 3단계 비교
2. CityGML 3D 뷰 (class별 색상)
3. val3dity 결과 하이라이트
4. City3D vs 제안 방법 side-by-side
5. Synthetic 예측 vs Real 실측 비교표
6. 처리 시간 비교표

results/real/REPORT.md 작성. CLAUDE.md 업데이트.
```

---

## REPORT.md 템플릿

```markdown
# [실험 이름] 결과 보고

## 수행 일시
YYYY-MM-DD

## 목적
<!-- 이 실험이 답하는 질문. 전체 실험 흐름에서의 위치. -->

## 핵심 결론
<!-- 1-2문장 + 핵심 정량 수치. 바로 답을 봄. -->

## 실험 설계
<!-- 가설, 데이터, 노이즈 조건, 평가 지표 -->

## 결과
<!-- 교차표 + 그래프. 가설 vs 실제 비교. -->

## Stage 2 반영 / 다음 단계
<!-- 이 결과가 다음 실험에 어떻게 연결되는지 -->

## 알려진 한계

## 시각적 산출물
<!-- 인라인 이미지 + CityJSON Ninja 파일 목록 -->

## 생성/수정 파일
```
