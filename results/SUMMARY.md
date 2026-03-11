# Semantic PlanarSplatting — 전체 Phase 진행 현황

> 모든 수치는 MVS depth/normal 기준으로 통일. Phase 0의 Metric3D 기준 수치는 별도 행으로 표기.

## Phase별 핵심 지표 요약

| Phase | Depth MAE | Normal cos | mIoU | GT 기준 | 학술적 의의 |
|-------|-----------|-----------|------|---------|------------|
| **0** (Metric3D 기준)¹ | 0.0670 | 0.9110 | — | Metric3D | 모니터링 인프라 구축, 코드 구조 파악 |
| **1** MVS Native | 0.0229 | 0.7811 | — | MVS | MVS depth가 Metric3D 대비 정확 (-66%), normal은 GT 차이로 직접 비교 불가 |
| **2-B** Semantic Head | — | — | — | — | f_i 파라미터, L_sem, CUDA backward fix, gradient 격리 검증 (구현 Phase) |
| **2-C** L_sem 독립 | 0.0271 | 0.7820 | 0.8096 | MVS + seg_v10 | Gradient 격리 실험적 확인 — L_mutual 귀속 근거 확보 |
| **3-A** L_mutual 구현 | — | — | — | — | gradient check 14/14 PASS, warmup curriculum (구현 Phase) |
| **3-B** Ablation | 아래 표 참조 | | | MVS + seg_v10 | L_mutual 효과 정량 분리, 양방향 시너지 미입증 (Retry) |

¹ Phase 0은 Metric3D를 GT로 사용하므로 Phase 1+ (MVS GT)와 직접 비교 불가.

## Phase 3-B Ablation 상세

| 조건 | 설명 | Depth MAE | Normal cos | mIoU | Δ(c) Normal | Δ(c) mIoU |
|------|------|-----------|-----------|------|-------------|-----------|
| (a) Geo Only | L_d+L_n+L_geo | 0.0275 | 0.7806 | — | -0.0009 | — |
| (b) Sem Only | L_d+L_n+L_sem | 0.0266 | 0.7802 | 0.7835 | -0.0014 | -0.0024 |
| **(c) Independent** | L_d+L_n+L_geo+L_sem | **0.0274** | **0.7816** | **0.7859** | **baseline** | **baseline** |
| (d) Joint | +L_mutual(full) | 0.0279 | 0.7823 | 0.7725 | +0.0008 | **-0.0134** |
| (e) Sem→Geo | +L_mutual(s→g) | 0.0272 | 0.7811 | 0.7834 | -0.0004 | -0.0025 |
| (f) Geo→Sem | +L_mutual(g→s) | 0.0279 | 0.7830 | 0.7696 | +0.0014 | -0.0163 |
| (g) No Warmup | +L_mutual(no-wu) | 0.0285 | 0.7817 | 0.7669 | +0.0001 | -0.0190 |

### 핵심 결론
- **L_geo 효과 확인**: (c)-(b) Normal cos +0.0014, mIoU +0.0024
- **L_mutual Normal cos 개선**: (d)-(c) +0.0008, (f) +0.0014가 최고
- **L_mutual mIoU 악화**: 모든 L_mutual 조건에서 mIoU 하락 (기하 prior와 GT 충돌)
- **양방향 시너지 미입증**: (d) < max((e),(f)) in both metrics
- **Warmup 유효**: (g) < (d) in mIoU (-0.0056) and Normal cos (-0.0007)
- **Go/No-Go**: Retry — λ_mutual 감소(0.05→0.01), iteration 증가(5K→10K) 후 재실험

## Phase 2-A Segmentation 현황
- Grounded SAM 2 (Confident Labels Only, v10)
- Coverage: 48.8% (Roof 5.9%, Wall 23.4%, Ground 19.5%)
- Ambiguous 51.2% → BG (ignore) — L_mutual 효과 제한 요인

## 생성 산출물 경로

| 파일 | 내용 |
|------|------|
| `results/phase0/REPORT.md` | Phase 0 결과 (모니터링 구축) |
| `results/phase1/REPORT.md` | Phase 1 결과 (MVS 전환) |
| `results/phase2a/REPORT.md` | Phase 2-A 결과 (Segmentation) |
| `results/phase2b/REPORT.md` | Phase 2-B 결과 (Semantic Head 구현) |
| `results/phase2c/REPORT.md` | Phase 2-C 결과 (L_sem 독립 학습) |
| `results/phase3a/REPORT.md` | Phase 3-A 결과 (L_mutual 구현) |
| `results/phase3b/REPORT.md` | Phase 3-B 결과 (Ablation Study) |
| `results/phase3b/ablation_comparison.csv` | Ablation 비교 표 |
| `results/phase3b/eval/abl_{a..g}.json` | 조건별 평가 JSON |
| `results/phase3b/images/{a..g}/` | 조건별 렌더링 이미지 |
| `results/phase3b/ply/{a..g}_*.ply` | 조건별 3D PLY |
