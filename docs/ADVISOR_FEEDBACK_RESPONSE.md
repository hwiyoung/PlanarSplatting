# 지도교수 피드백 대응 및 연구 방향 조정안 (v5)

> 최종 수정: 2026-03-20 (Stage 통일, gravity bootstrap, 합성 실험 A/B 반영)

---

## 1. 지도교수 질문 대응

### Q1. 왜 L_photo를 고려하지 않는가?
**대응**: L_photo를 기본 설정에 포함. 구현 완료. PGSR, 2DGS 등 기하학적으로 우수한 GS 연구들이 모두 사용. 특히 MVS 깊이가 희소한 벽면에서 텍스처 매칭이 기하학을 보완하며, CityGML WallSurface 품질에 직접 기여.

### Q2. L_mutual이 CityGML에 중요한가?
**대응**: L_mutual을 유일한 핵심 기여로 두지 않음. 파이프라인의 한 구성요소. 효과는 Baseline vs Joint ablation으로 검증. 기존 예비 실험의 mIoU 악화는 gravity 미보정이 원인일 가능성 → gravity 보정 + L_photo 포함 + L_height 추가(4항) 상태에서 재실험.

### Q3. Stage 3에서 mesh를 활용하면?
**대응**: TSDF mesh watertight 특성으로 프리미티브 토폴로지 보정. 향후 개선 방향으로 검토. 현재 스케치에는 미구현으로 명시.

### Q4. Splatting이 기존 방법보다 좋은 점?
**대응**: (1) 평면 프리미티브가 CityGML gml:Polygon과 구조적 대응 — 다단계 변환 불필요, (2) 학습 가능 파라미터이므로 의미론 동시 최적화 가능 — RANSAC에서는 불가, (3) Stage 2 최적화 내 오차 전파 방지(단, Stage 1/3 오차는 별도). City3D 비교로 실증.

---

## 2. Terrain Class와 CityGML GroundSurface

- **Terrain** (class=3) = 실제 지면. CityGML 미포함. Context class.
- **GroundSurface** = roof 경계를 terrain 높이로 투영하여 Stage 3에서 추론 생성.
- Terrain 역할: (1) context — 건물/비건물 경계 정의, wall 수직 범위 결정, (2) gravity 추정 소스, (3) 지면 높이 참조.
- K=4는 CityGML LOD2의 3 BoundarySurface에 대응하는 최소 필요 클래스 집합.

---

## 3. Gravity 추정

- **방법**: Grounded SAM 2D GT terrain 영역의 MVS 법선 평균 = UP → gravity = -UP.
- **Bootstrap 해결**: Stage 1 출력에서 학습 전 사전 계산. terrain 프리미티브 미분류 시에도 사용 가능.
- **검증**: roof/terrain 법선이 독립적으로 추정 gravity와 93% 정렬, cos=0.9986.
- **이력**: 기존 [0,-1,0] → 부정확 발견 → terrain 법선 기반 보정.
- Gravity 효과 분리 실험: 불필요 (잘못 계산한 것을 바로잡은 것).

---

## 4. 기여 방향

CityGML LOD2 자동 생성이라는 목표를 위한 통합적 방법론. 세 가지 설계 선택:
- **설계 선택 1**: 평면 프리미티브 + 의미론 → 기하학적 유효성 + 의미적 정확성
- **설계 선택 2**: L_mutual 4항(도메인 지식 미분 가능 인코딩) → 기하학적 유효성 + 의미적 정확성
- **설계 선택 3**: Structured Building Model Extraction → 위상적 일관성 + 의미적 정확성

---

## 5. 실험 우선순위

1. Stage 2-Step 1: Gravity 보정 + Baseline
2. Stage 2-Step 2: L_mutual ablation (Joint, 조건부 Joint-GTOnly/Weak)
3. Stage 2-Step 3: CityGML 전제 조건 판정
4. Stage 3-Synthetic: 합성 데이터 실험 A(프리미티브 노이즈) + B(감독 신호 노이즈)
5. Stage 3-Real: CityGML 생성 + val3dity
6. City3D 비교

---

## 6. 합성 데이터 실험

- **GT**: 벤치마크 CityGML(뉴욕, 베를린) 건물 20개
- **실험 A**: GT → multi-primitive(면당 10/30/50) → 노이즈(법선/위치/분류/누락/아웃라이어) → Stage 3 → GT 비교
- **실험 B**: GT → 합성 렌더링 → 감독 신호 노이즈(clean/depth/seg/뷰) → Stage 2+3 → GT 비교
- **목적**: Stage 3 알고리즘 검증 + 파이프라인 노이즈 전파 파악 + Stage 2 품질이 Stage 3에 미치는 영향 규명
