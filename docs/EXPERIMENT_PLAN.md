# 실험 계획 (v2 — 지도교수 피드백 반영)

## 이 문서의 역할
이 문서는 Claude Code가 자동으로 읽지 않는다 (자동으로 읽히는 것은 CLAUDE.md뿐).
각 Phase의 프롬프트를 Claude Code 채팅창에 복붙하여 사용한다.

## 연구 방향 조정 요약 (2026-03-11)
지도교수 피드백에 따라 연구의 무게 중심을 조정:
- **L_photo를 기본 설정에 포함**: 최종 재구축 품질 최대화. L_photo 미포함 조건은 유지하지 않음.
- **Phase 4(CityGML 변환)를 최우선**: Phase 4까지 끝까지 가보는 것이 핵심.
- **L_mutual은 파이프라인의 한 구성요소**: 효과와 한계를 분석하는 대상.
- **City3D 비교 추가**: splatting 기반 접근의 장점 실증.
- **기존 Phase 3-B**: L_photo 미포함 예비 분석. 경로 1/2 진단은 재실험 설계 근거로 유지.
- **Ground class**: 지면(terrain), CityGML GroundSurface(건물 바닥면)와 구별. Context class.
상세: docs/ADVISOR_FEEDBACK_RESPONSE.md

## Phase 의존성 (조정 후)

```
Phase 0-Setup → ... → Phase 3-B (예비, 완료)
                                     ↓
                              Phase 3-B'-Step1 (L_photo 구현 + (c') baseline)
                                     ↓
                          ┌──────────┴──────────┐
                          ↓                     ↓
                   Phase 4 프로토타입     Phase 3-B'-Step2
                   ((c') checkpoint →     (L_mutual 재실험)
                    CityGML + mesh 보정)        ↓
                          ↓              City3D 비교
                   Phase 4 고도화 ←──────┘
```

## 데이터셋 / 비교 방법 / 평가 지표

| 데이터셋 | 용도 | 비고 |
|----------|------|------|
| 성수동 드론 이미지 | 주요 실험 | 180장, oblique, 70m, COLMAP 100장 |
| UrbanScene3D | 향후 일반화 | GT 메쉬 포함 |

| 비교 방법 | 목적 | 시점 |
|----------|------|------|
| City3D | 순차적 파이프라인 vs splatting | 2순위 |
| PlanarSplatting 원본 | 의미론 통합 효과 | Phase 1 결과 활용 |
| (c') vs (d') ablation | L_mutual 효과 (L_photo 포함) | 1순위 |

| 범주 | 지표 | 선정 이유 | 관련 연구 |
|------|------|----------|----------|
| 기하학 | Depth MAE | 프리미티브 위치 정확도 | PlanarSplatting, PGSR, 2DGS |
| 기하학 | Normal cos | 법선 정확도; CityGML 분류가 면 방향에 의존 | PlanarSplatting, PGSR, 2DGS |
| 의미론 | mIoU / per-class IoU | 구성요소 분류; class별 편향 식별 | AlignGS, NeRBuilder |
| 진단 | Wall 법선 정확도 | L_mutual 핵심 메커니즘 검증 | 본 연구 고유 |
| 렌더링 | PSNR | RGB 품질 (L_photo 포함 후) | 3DGS, PGSR |
| 실용 | val3dity 통과율 | CityGML 유효성 | PLANES4LOD2 |
| 실용 | 처리 시간 | City3D 비교용 | |

---

## 완료된 Phase 요약 (Phase 0 ~ Phase 3-B)

Phase 0-Setup ~ Phase 3-A: 기존대로 수행 완료. 각 REPORT.md 참조.

### Phase 3-B 결과 (완료, 예비 분석으로 재위치)
- (c) Independent: Normal cos=0.7816, mIoU=0.7859
- (d) Joint: Normal cos +0.0008, mIoU **-0.0134**
- 양방향 시너지 미입증. 경로 1(cross-view contamination) / 경로 2(직접 충돌) 진단.
- L_photo 미포함 상태의 결과이므로, L_photo 포함 시 유효성 보장 불가. 경로 분석 논리는 유지.

---

## 1순위: Phase 3-B' + Phase 4 프로토타입 (병행)

### Phase 3-B'-Step1: L_photo 구현 및 baseline 확립

**목표**: L_photo를 PlanarSplatting에 추가, L_photo 포함 baseline 확립.

**사전 확인**: color=random(비학습), CUDA rasterizer NUM_CHANNELS=4 (semantic용). L_photo를 위해 color를 학습 파라미터로 변경하고, RGB 렌더링 경로를 확보해야 함.

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 3-B'-Step1을 진행해줘. 컨테이너 내부에서 작업이야.
docs/RESEARCH_CONTEXT.md와 docs/ADVISOR_FEEDBACK_RESPONSE.md도 참고해줘.

목표: PlanarSplatting에 L_photo를 추가하고 baseline 확립.

Step 1: L_photo 구현 가능성 분석
- 현재 color 처리 방식 확인 (random 생성, 비학습)
- CUDA rasterizer RGB 렌더링 경로 확인 (NUM_CHANNELS, forward.cu)
- Phase 2-B의 color gradient backward가 L_photo에 활용 가능한지 확인
- semantic 렌더링(f_i, 4ch)과 RGB 렌더링(color, 3ch) 동시 수행 방안 검토:
  방안 A: 렌더링 2회 (semantic용 + RGB용)
  방안 B: 채널 확장 (4ch semantic + 3ch RGB = 7ch)
  방안 C: 기타 (코드 구조에 따라)
- 각 방안 장단점 분석 후 선택

Step 2: L_photo 구현
- color를 학습 파라미터 (N,3)로 추가. SfM 포인트 색상으로 초기화.
- L_photo = (1-0.2)*L1(rendered_rgb, gt_rgb) + 0.2*(1-SSIM(rendered_rgb, gt_rgb))
- --enable_photo --lambda_photo 플래그
- density control에서 color split/prune 동기
- enable_semantic과 enable_photo 독립 on/off

Step 3: (c') Independent+Photo 학습
- L_depth + L_normal + L_geo + L_sem + L_photo (λ_m=0), 5000 iter
- 평가: Depth MAE, Normal cos, mIoU, PSNR

Step 4: (c') vs (c) 비교 — L_photo 추가 효과

results/phase3b_prime/step1_REPORT.md 작성. CLAUDE.md 업데이트.
```

### Phase 3-B'-Step2: L_mutual 재실험 (L_photo 포함)

**목표**: L_photo 포함 baseline에서 L_mutual 효과 재검증.

**선행**: Step1 완료

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 3-B'-Step2를 진행해줘. 컨테이너 내부에서 작업이야.

목표: L_photo 포함 상태에서 L_mutual 효과 재검증.

실험 (모두 L_photo 포함):
- (c') Independent+Photo: Step1에서 완료 (재사용)
- (d') Joint+Photo: + L_mutual full, warmup
- (d'-masked): L_mutual을 GT가 있는 영역 프리미티브에만 적용
- (d'-small): λ_m = 0.01

평가: Depth MAE, Normal cos, mIoU, per-class IoU, PSNR, wall 법선 정확도

해석:
- (d') vs (c'): L_photo 포함 시 L_mutual 효과
- (d'-masked) vs (c'): 경로 1 제거 효과
- (d'-small) vs (c'): gradient 크기 진단
- Phase 3-B(예비) vs Phase 3-B'(본실험) 비교

results/phase3b_prime/step2_REPORT.md 작성. CLAUDE.md 업데이트.
```

### Phase 4 프로토타입: CityGML 변환 + mesh 기반 위상 보정

**목표**: 프리미티브 → CityGML LOD2 + val3dity. TSDF mesh로 토폴로지 보정. Phase 4까지 끝까지 가보는 것이 핵심.

**선행**: Phase 3-B'-Step1 완료. (c') Independent+Photo checkpoint 사용.

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 4 프로토타입을 진행해줘. 컨테이너 내부에서 작업이야.
docs/ADVISOR_FEEDBACK_RESPONSE.md의 Q3, 2절도 참고해줘.

목표: 프리미티브 → CityGML LOD2 + val3dity. (c') Independent+Photo checkpoint 사용.
Checkpoint: planarSplat_ExpRes/phase3b_prime/cprime_independent_photo_example/2026_03_11_14_06_52/checkpoints/Parameters/latest.pth

중요 — Ground class:
- "ground" = 실제 지면(terrain). CityGML GroundSurface(건물 바닥면)와 다름.
- Ground 프리미티브 → CityGML에 포함 안 함. 건물/비건물 분리용 context class.
- GroundSurface = roof 경계를 지면 높이로 투영하여 별도 생성.

Part 1: 프리미티브 분석
scripts/analyze_primitives.py:
1. Checkpoint에서 추출 (center, normal, radii, semantic logits)
2. argmax(softmax(f_i)) → roof/wall/ground/bg
3. 통계: class별 수, 법선 분포, 높이 분포
4. Ground/BG 필터링 → 건물 프리미티브(roof+wall) 추출
5. 시각화: class별 PLY

Part 2: Building Instance 분리
scripts/building_grouping.py:
1. Roof 중심 → XZ 투영
2. Distance threshold connected component → building_id
3. Wall → 최근접 roof cluster 귀속
4. 시각화: building_id별 PLY

Part 3: TSDF Mesh 추출 + 위상 분석
1. refuse_mesh()로 TSDF mesh 추출
2. Watertight 확인 (trimesh: is_watertight, holes)
3. 프리미티브-mesh 대응: vertex→프리미티브 매핑, 미커버 영역(틈) 식별
4. 위상 문제 진단: 틈의 위치/크기/빈도

Part 4: CityGML 변환
scripts/export_citygml.py (per-building):
1. 프리미티브 병합: 같은 class + cos(normal)>0.95 + 중심거리<2*r_i
2. 경계 다각형 (alpha shape / convex hull)
3. GroundSurface: roof 경계를 ground 프리미티브 평균 높이로 투영
4. CityGML XML + OBJ 생성

Part 5: val3dity 검증 + 오류 분류

Part 6: Mesh 기반 위상 보정 (프로토타입)
- Part 3 결과 기반: 틈 영역에서 mesh 정보로 프리미티브 경계 확장
- 보정 후 CityGML 재생성 + val3dity 재검증

results/phase4/REPORT.md 작성. CLAUDE.md 업데이트.
```

---

## 2순위: City3D 비교

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 City3D 비교를 진행해줘. 컨테이너 내부에서 작업이야.

1. City3D 설치 (https://github.com/tudelft3d/City3D)
2. COLMAP dense cloud → City3D 입력 변환
3. 성수동 데이터에서 City3D 실행
4. 비교: 기하학적 정확도, 의미론(사후 휴리스틱 vs 통합 학습), val3dity, 처리 시간
5. 동일 건물 시각적 비교

results/comparison_city3d/REPORT.md 작성.
```

---

## 3순위: 확장 및 보완

- Mesh 토폴로지 보정 고도화
- L_mutual 확장 (Phase 3-B' 결과 따라): L_height, confidence-aware 등
- 추가 데이터셋 (UrbanScene3D) 일반화 검증

---

## 결과 기록 REPORT.md 템플릿

```markdown
# Phase X: [이름]

## 수행 일시 / 수행 작업 요약

## 정량 지표
| 지표 | 값 | 이전 | 변화 | GT 기준 |

## 정성적 결과
<!-- 작성 기준:
- 구체적 뷰/이미지를 언급하며 관찰 내용을 서술 (파일 경로 나열 금지)
- 각 이미지에 대해: 무엇이 보이는지, 어떤 의미인지, 이전 Phase와 달라진 점
- 카테고리별 (RGB, Depth, Normal, Semantic) 분류하여 작성
- 8~9장 이상, results/phaseX/images/ 저장 -->

## 3D PLY 시각화
<!-- 필수 산출물 (Phase 2-B 이후):
- *_normal.ply: 법선 색상
- *_class.ply: 클래스 색상 (roof=red, wall=blue, ground=gray)
- *_rgb.ply: 학습된 RGB 색상 (enable_photo 시)
각 PLY에 대해 관찰 내용 1~2문장 서술 -->

## Go/No-Go

## 생성/수정 파일 목록

## 이슈 및 해결 / 다음 Phase
```
