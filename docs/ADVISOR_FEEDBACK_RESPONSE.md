# 지도교수 피드백 대응 및 연구 방향 조정안 (v4)

> 최종 수정: 2026-03-19 (gravity 보정 발견 반영)

---

## 1. 지도교수 질문 대응

### Q1. 왜 L_photo를 고려하지 않는가?
**대응**: L_photo를 기본 설정에 포함. 구현 완료.

### Q2. L_mutual이 CityGML에 중요한가?
**대응**: L_mutual을 유일한 핵심 기여로 두지 않음. 효과는 실험으로 확인, 결과에 따라 기여 범위 결정. **추가 발견: 기존 Phase 3-B에서 L_mutual이 악화를 보인 것은 gravity 미보정이 원인일 가능성. gravity 보정 후 재실험.**

### Q3. Phase 4에서 mesh를 활용하면?
**대응**: TSDF mesh watertight 특성으로 프리미티브 토폴로지 보정. 향후 개선 방향으로 검토.

### Q4. Splatting이 기존 방법보다 좋은 점?
**대응**: 재구축 과정 내 오차 전파 방지 + 기하/의미론 동시 최적화 + 확장성. City3D 비교로 실증.

---

## 2. Terrain Class와 CityGML GroundSurface

- Terrain (class=3) = 실제 지면. CityGML 미포함. Context class.
- GroundSurface = roof 경계를 terrain 높이로 투영하여 추론 생성.
- Terrain 역할: (1) context, (2) gravity 추정, (3) 지면 높이 참조.

---

## 3. Gravity 보정 (2026-03-19 발견)

- 기존: [0,-1,0] (camera -Y axis 평균). **부정확.**
- 보정: terrain 프리미티브 1039개 법선 평균 → gravity = -UP.
- 검증: roof/terrain 독립 93% 정렬, cos=0.9986.
- **영향**: 기존 Phase 3-B의 L_mutual 결과가 무효화될 가능성. gravity 보정 후 재실험 필수.

---

## 4. 기여 방향 (실험 결과 따라 확정)

- **A**: 평면 프리미티브 + 의미론 → CityGML LOD2 실증 (City3D 비교)
- **B**: 의미론-기하학 상호보완, L_mutual (gravity 보정 후 재검증)
- **C**: TSDF mesh 토폴로지 보정 (향후)

---

## 5. 실험 우선순위

1. Gravity 보정 + baseline 재학습
2. L_mutual 재실험 (gravity + L_photo)
3. CityGML 전제 조건 판정
4. Phase 4 합성 민감도 분석 + Real CityGML
5. City3D 비교
