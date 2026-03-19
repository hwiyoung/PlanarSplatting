# 실험계획 (v3)

## 현재 상태

| Phase | 상태 | 비고 |
|-------|------|------|
| Phase 0~2-C | 완료 | 유효 |
| Phase 3-A | 완료 | L_mutual 구현 검증 |
| Phase 3-B | 완료 | L_photo 미포함 + gravity 미보정. **예비 분석** |
| Phase 3-B' L_photo | 완료 | **L_photo 구현 완료. gravity 미보정.** |
| Phase 4 프로토타입 | 진행 중 | Cluster-Intersection 구현. wall 법선 문제, gravity 보정 발견 |

**재시작: Phase 3-B'-Step1 gravity 보정부터.**

## 실험 순서

```
Step1: Gravity 보정 + baseline 재학습
  ↓
Step2: L_mutual 재실험
  ↓
Step3: CityGML 전제 조건 판정 (Stage 2 실제 품질 수치 측정)
  ↓ (병행)
Phase 4-Sensitivity: 합성 데이터 입력 민감도 분석
  ↓
Phase 4-Real: CityGML 생성 + val3dity
  ↓
City3D 비교
```

---

## Phase 3-B'-Step1: Gravity 보정 + Baseline 재학습

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 3-B'-Step1을 진행해줘. 컨테이너 내부에서 작업이야.

목표: gravity 보정 + L_photo 포함 baseline 재학습.
L_photo 구현은 이미 완료. gravity 보정만 추가.

=== Part A: Gravity 보정 ===
1. loss_util.py에서 e_gravity 값 찾기
2. hardcoded → 데이터 자동 추정으로 변경:
   - 학습 시작 시 terrain class(class=3) 프리미티브 법선 평균 계산
   - gravity = -(terrain 법선 평균), normalize
   - 학습 중 고정
   - fallback: terrain 부족 시 [0,-1,0] + 경고
3. 추정 gravity를 TensorBoard + 콘솔 로깅
4. 검증: roof/terrain 법선과 gravity 정렬도(dot>0.7 비율) 출력

=== Part B: Baseline 재학습 ===
(c') Independent+Photo: L_d + L_n + L_geo + L_sem + L_photo (λ_m=0)
- gravity: Part A 추정값, 5000 iter
- 평가: Depth MAE, Normal cos, mIoU, per-class IoU, PSNR
- CityGML 전제: wall 법선 수직도(gravity 수직 10도 이내), roof 법선 정렬도

=== Part C: 비교 ===
(c') gravity 보정 vs 기존 gravity 미보정: 효과 분리
wall 법선 수직도 변화에 주목

=== 시각적 산출물 (필수) ===
1. 프리미티브 PLY: class별 색상 (roof=red, wall=blue, terrain=gray)
2. 프리미티브 PLY: 법선 방향 색상 (gravity 수직 10도 이내=green, 초과=red)
3. 렌더링 이미지: Depth/Normal/Semantic/RGB 각 4장 (다양한 뷰)
4. gravity 방향 시각화: 좌표축 + gravity 벡터를 PLY에 오버레이

results/phase3b_prime/step1_gravity_REPORT.md 작성 (하단 REPORT 템플릿 준수).
CLAUDE.md 업데이트.
```

---

## Phase 3-B'-Step2: L_mutual 재실험

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 3-B'-Step2를 진행해줘. 컨테이너 내부에서 작업이야.

목표: gravity 보정 + L_photo 상태에서 L_mutual 효과 재검증.

=== 실험 조건 (모두 L_photo + gravity 보정) ===
- (c'): Step1 완료 (재사용)
- (d') Joint+Photo: + L_mutual full, warmup
- (d'-masked): L_mutual을 GT 있는 영역 프리미티브에만
- (d'-small): λ_m = 0.01

=== 평가 ===
기본: Depth MAE, Normal cos, mIoU, per-class IoU, PSNR
CityGML 전제: wall 법선 수직도, roof 법선 정렬도, roof-wall 혼동률

=== 해석 ===
- (d') vs (c'): L_mutual → wall 법선 수직도 개선?
- (d'-masked) vs (c'): 경로 1 제거 효과
- Phase 3-B(gravity 미보정) vs 3-B'(보정): gravity 보정이 L_mutual에 미치는 영향

=== 시각적 산출물 (필수) ===
1. 조건별 프리미티브 PLY: class별 색상 (c', d', d'-masked, d'-small)
2. 조건별 wall 법선 수직도 히스토그램 (gravity 대비 각도 분포)
3. (c') vs (d') 동일 뷰 렌더링 비교: Normal/Semantic side-by-side
4. Phase 3-B(예비) vs Phase 3-B'(본실험) 비교 표 + 막대그래프

results/phase3b_prime/step2_REPORT.md 작성.
CLAUDE.md 업데이트.
```

---

## Phase 3-B'-Step3: CityGML 전제 조건 판정

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 3-B'-Step3을 진행해줘. 컨테이너 내부에서 작업이야.

목표: Phase 3 최선 조건이 Stage 3에 충분한지 판정 + 실제 품질 수치 측정.

=== 판정 기준 ===
1. Wall 법선 수직도 ≥ 80% (gravity 수직 10도 이내)
2. Roof 법선 정렬도 ≥ 90%
3. mIoU ≥ 0.80
4. Roof-Wall 혼동률 < 10%
5. 건물 표면 프리미티브 커버리지: 시각적 확인

=== 작업 ===
1. Step2 모든 조건에서 5개 기준 측정 → 충족 표
2. 최선 조건 선정
3. 미충족 기준 원인 분석
4. Phase 4-Sensitivity용 실제 품질 수치:
   - wall 법선의 gravity 수직 편차 분산 (σ_normal)
   - 분류 오류율 (전체 + class별)
   - 면 누락률 추정 (가능하면)

=== 시각적 산출물 (필수) ===
1. 판정 기준 충족 표 (조건 × 기준, 색상 코딩: 초록=충족, 빨강=미충족)
2. 최선 조건의 3D 시각화: building별 색상 PLY
3. 커버리지 시각화: 건물 표면에 프리미티브가 없는 영역 하이라이트
4. wall 법선 분포 히스토그램 (σ_normal 표시)

results/phase3b_prime/step3_REPORT.md 작성.
CLAUDE.md 업데이트.
```

---

## Phase 4-Sensitivity: 합성 데이터 입력 민감도 분석

**프레이밍**: "강건성 벤치마크"가 아니라 "입력 민감도 분석". 상대적 민감도가 핵심.

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 4-Sensitivity를 진행해줘. 컨테이너 내부에서 작업이야.

목표: Stage 3 알고리즘의 입력 민감도 분석.
어떤 유형의 입력 오류가 CityGML 품질에 가장 큰 영향을 미치는가?

=== Part A: 합성 데이터 생성 ===
scripts/generate_synthetic_buildings.py:
- box (6면), gable roof (7면), L자형 (10면)
- 이상적 프리미티브 → noise=0 → Stage 3 → val3dity 통과 확인

=== Part B: 노이즈 부여 ===
scripts/add_noise_to_primitives.py:

Step3에서 측정된 실제 품질에서 역산:
- σ_real(wall 법선 분산) → 노이즈: 0.5σ, σ, 1.5σ, 2σ
- e_real(분류 오류율) → 노이즈: 0.5e, e, 1.5e, 2e
- 측정 전이면 임시: 법선 5°,10°,15°,20° / 분류 5%,10%,15%,20%

노이즈 유형 (각각 독립):
1. 법선 노이즈: Gaussian 회전
2. 위치 노이즈: center 변위 (0.1m, 0.3m, 0.5m, 1.0m)
3. 분류 오류: class 랜덤 교체
4. 면 누락: 프리미티브 랜덤 제거 (10%, 20%, 30%, 50%)

=== Part C: Stage 3 실행 + 평가 ===
각 조건: val3dity 통과율, 오류 유형 분포

=== Part D: 분석 ===
1. 민감도 순위: 어떤 오류 유형이 가장 큰 영향?
2. Phase 3 실제 출력이 어느 노이즈 수준에 해당 → Stage 3 예상 결과
3. Stage 2 개선 우선순위 도출

=== 시각적 산출물 (필수) ===
1. noise=0 합성 건물 시각화: 프리미티브 + CityGML 결과 side-by-side
2. 노이즈 유형별 val3dity 통과율 그래프 (x=노이즈 수준, y=통과율, 4개 선)
3. 노이즈 유형별 대표 CityGML 결과 시각화 (정상 vs 오류 비교)
4. Phase 3 실제 품질 위치를 그래프에 세로선으로 표시
5. 민감도 순위 막대그래프

results/phase4_sensitivity/REPORT.md 작성.
CLAUDE.md 업데이트.
```

---

## Phase 4-Real: CityGML 생성

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 4-Real을 진행해줘. 컨테이너 내부에서 작업이야.

목표: Phase 3 최선 조건 → CityGML LOD2 + val3dity.

입력: Step3 최선 조건 checkpoint. Gravity 보정값.
Terrain: CityGML 미포함. GroundSurface = roof 경계를 terrain 높이로 투영.

Stage 3 실행 → val3dity → Sensitivity 예측 비교 → 시각화.

=== 시각적 산출물 (필수) ===
1. CityGML 모델 3D 시각화: class별 색상 (roof=red, wall=blue, ground=green)
2. val3dity 결과 시각화: 오류 위치를 모델 위에 하이라이트
3. 건물별 품질 표: building_id, 면 수, val3dity 결과, 주요 오류
4. 입력 이미지 → 프리미티브 → CityGML 3단계 비교 이미지
5. Phase 4-Sensitivity 예측 vs 실제 결과 비교 표

results/phase4_real/REPORT.md 작성.
CLAUDE.md 업데이트.
```

---

## City3D 비교

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 City3D 비교를 진행해줘. 컨테이너 내부에서 작업이야.

1. City3D 설치 + COLMAP dense cloud → City3D 입력
2. 성수동 데이터에서 City3D 실행
3. City3D도 val3dity 검증
4. 비교: val3dity, 의미론, 기하학, 처리 시간

=== 시각적 산출물 (필수) ===
1. 동일 건물 side-by-side: City3D vs 제안 방법 (3D 모델, class 색상)
2. val3dity 통과율 비교 막대그래프
3. 처리 시간 비교 표
4. 의미론 분류 비교: City3D 사후 휴리스틱 vs 제안 방법 (오분류 하이라이트)

results/comparison_city3d/REPORT.md 작성.
```

---

## REPORT.md 템플릿

각 Phase/Step 완료 시 해당 디렉토리에 REPORT.md를 생성한다. 하나의 파일에 누적하지 않고 Phase마다 독립 파일.

```markdown
# [Phase/Step 이름] 결과 보고

## 수행 일시
YYYY-MM-DD

## 수행 작업 요약
<!-- 1. 이 단계가 전체 연구에서 어디에 위치하는지 1문단 서술
     2. 이전 단계에서 확보한 전제조건
     3. 이 단계에서 검증하려는 것
     4. CityGML 품질 기준과의 연결 -->

## 정량 지표
<!-- Phase 간 비교 시 GT 기준이 다르면 명시. 변화의 원인 1-2문장 해석. -->
| 지표 | 값 | 이전 | 변화 | GT 기준 |
|------|-----|------|------|---------|

### CityGML 전제 조건 (해당 시)
| 기준 | 목표 | 실측 | 충족 |
|------|------|------|------|

## 정성적 결과
<!-- 8-9장 이상. 선별 기준: 건물 다양(고층/저층, flat/경사 roof), 카메라 다양.
     각 이미지에 구체적 캡션: 어떤 구조물인지, 어떤 결과인지, 이전과 달라진 점.
     이미지는 results/[phase]/images/에 저장, 상대 경로 참조. -->

### 시각적 산출물 체크리스트
<!-- 프롬프트에 명시된 시각적 산출물이 모두 생성되었는지 체크 -->
- [ ] 산출물 1: ...
- [ ] 산출물 2: ...

## Go/No-Go 판단
<!-- 숫자 기준 충족 + 다음 단계 전제조건 만족 논증.
     불합격 시 Retry/Switch 경로와 근거. -->
- [ ] Go / [ ] Retry / [ ] Switch
- 근거: ...

## 생성/수정 파일 목록
| 파일 | 유형 | 핵심 변경 |
|------|------|----------|

## 이슈 및 해결

## 다음 단계
<!-- 이 결과가 다음 단계에 어떤 전제를 제공하는지 -->
```

※ 이미지는 `results/[phase]/images/`에 저장, 상대 경로 참조.
※ 시각적 산출물은 프롬프트에 명시된 항목을 반드시 모두 생성.
