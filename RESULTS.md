# ML-1M 신규 사용자 성별 공정성 실험 결과

기록일: **2026-08-13**

## 한 줄 결론

`MultVAE + GRL + soft GBS`는 naive MultVAE보다 **장르 노출 공정성을 일부 개선한 후보**입니다. 배포 seed 42에서는 대부분의 결과 공정성 지표가 좋아졌고, 3-seed 평균에서도 장르 노출 L1과 preference residual은 감소했습니다. 그러나 성별 NDCG gap은 전체적으로 일관되게 줄지 않았고 post-hoc gender probe는 오히려 악화했습니다. 따라서 이 결과를 “성별에 완전히 공정한 모델” 또는 “성별 정보 제거 성공”으로 해석하면 안 됩니다.

Canonical 원본 결과:

- [`artifacts/metrics/comparison.json`](artifacts/metrics/comparison.json): 사이트에 연결된 seed 42 전체 test 결과
- [`artifacts/metrics/multiseed_summary.json`](artifacts/metrics/multiseed_summary.json): seed 42/43/44 paired 요약
- `comparison_softgbs_seed43.json`, `comparison_softgbs_seed44.json`: 반복별 상세 결과
- `baseline_test.json`, `fair_test.json`: 배포 checkpoint별 상세 평가
- `comparison_grl_only_seed*.json`: GRL-only 폐기 ablation

## 연구 선택

문헌과 로컬 실험을 바탕으로 다음 후보를 동일 user-disjoint protocol에서 비교했습니다.

| 후보 | 신규 사용자 입력 | 역할 |
| --- | --- | --- |
| Popularity | 필요 없음 | 비개인화 sanity baseline |
| ItemKNN | 좋아한 영화 | 고전적인 item-to-item inductive baseline |
| MultVAE | 좋아한 영화 multi-hot | naive neural baseline |
| Adv-MultVAE + soft GBS | 좋아한 영화 multi-hot | 최종 공정성 정규화 후보 |

MF, NeuMF, LightGCN의 전형적인 user-ID embedding 구현은 새로운 user row를 바로 encode하지 못하므로 중심 후보에서 제외했습니다. RecSys 2023의 unseen-user VAE와 SIGIR 2022의 Adv-MultVAE가 이 문제 설정에 가장 직접적으로 맞았습니다.

주요 근거:

- [Unlearning Protected User Attributes in Recommendations with Adversarial Training](https://arxiv.org/abs/2206.04500), SIGIR 2022; [공식 코드](https://github.com/CPJKU/adv-multvae)
- [Providing Previously Unseen Users Fair Recommendations Using Variational Autoencoders](https://arxiv.org/abs/2308.15230), RecSys 2023; [공식 코드](https://github.com/BjornarVass/fair-vae-rec)
- [Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness](https://arxiv.org/abs/2502.17921), WWW 2025

현재 구현은 어느 논문의 exact reproduction도 아닙니다. 세 연구 아이디어를 ML-1M few-shot unseen-user protocol에 맞춰 결합한 **명시적 adaptation**입니다.

## 최종 정규화

Naive arm은 MultVAE reconstruction과 KL만 사용합니다. Fairness arm은 같은 backbone에 다음 두 항을 더합니다.

1. `GRL gender adversary`: posterior mean `mu`에서 성별을 분류하는 adversary의 encoder gradient를 반전합니다.
2. `soft GBS`: minibatch M/F의 예상 장르 노출 분포 차이를 줄입니다.

~~~text
p_ui = softmax(mask_seen_and_cold(logits_u) / T)_i
e_ug = normalize(sum_i p_ui * movie_genre[i, g])
L_soft_GBS = sum_g | mean_M(e_ug) - mean_F(e_ug) |

L_fair = L_reconstruction + beta * KL
       + 1.0 * CE(gender_adversary(GRL_lambda(mu)), gender)
       + 1.0 * L_soft_GBS
~~~

최종값은 `GRL lambda=1`, `soft GBS weight=1`, `T=1`입니다. 성별은 이 batch-level 정규화와 offline audit에만 쓰며 encoder/decoder의 입력이나 사이트 inference API에는 들어가지 않습니다. 학습 surrogate는 soft exposure이고, 평가는 별도의 **hard Top-10** 장르 노출로 수행했습니다.

## Protocol

- Dataset: MovieLens 1M, 평점 4 이상을 positive로 사용
- Split: gender-stratified user-disjoint train/validation/test = 4,832/604/604 users
- Model input: user ID 없이 고정 catalog의 positive multi-hot vector
- Training input: 매 epoch random support 3/5/10/20, 10% full profile; target은 full positive profile
- Evaluation support: 최초 warm positive 1/3/5/10/20개
- Query: 마지막 support 뒤의 positive, 이미 본 영화와 train-cold item 제외
- Candidates: train positive가 존재하는 3,506개 warm item; 377개 cold 좌표 제외
- Checkpoint selection: validation only
  - naive: NDCG@10 최대
  - fair: 최고 NDCG의 3% 이내에서 NDCG gender gap → hard genre exposure L1 → support-conditioned residual → probe leakage 순으로 최소화
- Test: 모델/weight/epoch 확정 후 reporting에만 사용

배포 checkpoint:

| 역할 | seed | epoch | SHA-256 |
| --- | ---: | ---: | --- |
| Naive MultVAE | 42 | 10 | `d7b9e857943cbf5c8adbdd16c4b0f71eeb56b8bdd848106c24d32b7cc41b0e79` |
| SoftGBS+GRL | 42 | 27 | `fd4f7b459ff820d483b68686c799e8a4bceee9b532b79dd7f2a866f375481cfa` |

## 배포 seed 42의 전체 test 결과

괄호 안은 fair 모델의 naive 대비 상대 변화입니다. NDCG와 coverage는 높을수록 좋고, gap/L1은 낮을수록 좋습니다.

| Support | Model | NDCG@10 | 성별 NDCG gap | 장르 노출 L1 | Preference residual L1 | Probe AUC | Coverage |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Naive | 0.3585 | 0.0677 | 0.1185 | 0.1715 | 0.5507 | 0.0525 |
| 1 | Fair | 0.3358 (-6.32%) | 0.0654 (-3.37%) | 0.1123 (-5.18%) | 0.1968 (+14.77%) | 0.5674 | 0.0710 |
| 3 | Naive | 0.3726 | 0.0594 | 0.1989 | 0.1083 | 0.6405 | 0.0479 |
| 3 | Fair | 0.3616 (-2.94%) | 0.0708 (+19.26%) | 0.1592 (-19.94%) | 0.1023 (-5.47%) | 0.6447 | 0.0696 |
| 5 | Naive | 0.3744 | 0.0631 | 0.2261 | 0.1330 | 0.6890 | 0.0511 |
| 5 | Fair | 0.3663 (-2.15%) | 0.0649 (+2.90%) | 0.1988 (-12.07%) | 0.0900 (-32.33%) | 0.6950 | 0.0782 |
| 10 | Naive | 0.3456 | 0.0737 | 0.3035 | 0.1224 | 0.7374 | 0.0673 |
| 10 | Fair | 0.3404 (-1.53%) | 0.0523 (-28.97%) | 0.2687 (-11.49%) | 0.1063 (-13.19%) | 0.7503 | 0.1038 |
| 20 | Naive | 0.3096 | 0.0572 | 0.4153 | 0.1999 | 0.7818 | 0.0944 |
| 20 | Fair | 0.2879 (-7.01%) | 0.0540 (-5.57%) | 0.4042 (-2.65%) | 0.1660 (-16.97%) | 0.7933 | 0.1349 |

seed 42에서 10 likes가 가장 명확한 Pareto 개선입니다. NDCG 손실 1.53%로 NDCG gender gap, 장르 노출, residual이 모두 감소합니다. 반대로 3/5 likes에서는 장르 지표는 개선되지만 NDCG gender gap이 악화합니다. 단일 support만 보고 공정성 성공을 선언하면 안 되는 이유입니다.

## 세 seed 반복 결과

아래는 3 seeds × 5 support sizes의 `fair - naive` paired 차이입니다. gap/exposure/residual/probe는 음수가 개선입니다.

| Support | ΔNDCG 평균±표준편차 | ΔNDCG gap | Δ노출 L1 | ΔResidual L1 | ΔProbe distance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.0165 ± 0.0106 | +0.0049 ± 0.0064 | +0.0027 ± 0.0195 | +0.0177 ± 0.0067 | +0.0088 ± 0.0091 |
| 3 | -0.0075 ± 0.0080 | +0.0107 ± 0.0079 | -0.0194 ± 0.0479 | +0.0123 ± 0.0158 | +0.0051 ± 0.0179 |
| 5 | -0.0055 ± 0.0078 | +0.0114 ± 0.0148 | -0.0132 ± 0.0301 | **-0.0317 ± 0.0103** | +0.0139 ± 0.0229 |
| 10 | -0.0052 ± 0.0082 | **-0.0124 ± 0.0093** | -0.0174 ± 0.0235 | -0.0071 ± 0.0172 | +0.0118 ± 0.0117 |
| 20 | -0.0103 ± 0.0190 | +0.0000 ± 0.0040 | -0.0043 ± 0.0275 | -0.0162 ± 0.0238 | +0.0124 ± 0.0017 |

전체 15개 paired case를 macro-average하면 다음과 같습니다.

- ΔNDCG@10: `-0.0090`
- Δ성별 NDCG gap: `+0.0029` — 평균적으로는 악화
- Δ장르 노출 L1: `-0.0103` — 10/15 case 개선
- Δpreference residual L1: `-0.0050` — 8/15 case 개선
- Δprobe AUC distance: `+0.0104` — representation leakage 완화 실패

반복에서 방향이 완전히 안정적인 결과는 support 5의 residual 개선(3/3)과 support 10의 NDCG gap 개선(3/3)입니다. 세 seed는 통계적 유의성을 주장하기에 부족합니다.

## Ablation에서 폐기한 설정

| 설정 | 판단 |
| --- | --- |
| GRL lambda 1/10/50/100 | 큰 lambda는 utility를 크게 낮춤; lambda 1도 probe leakage를 안정적으로 낮추지 못함 |
| support-conditioned soft residual weight 10/50/100/300 | soft surrogate와 hard Top-K residual 방향이 맞지 않아 폐기 |
| soft GBS weight 10/50/100 | raw exposure gap은 더 작아졌으나 preference residual과 utility가 크게 악화해 취향 삭제 위험으로 폐기 |
| soft GBS weight 1 | 세 지표 trade-off가 가장 작아 최종 채택 |

## 해석 경계

- 이 실험은 MovieLens가 제공한 제한적인 M/F label만 평가합니다.
- SoftGBS는 집단 장르 노출을 가까이 만들 수 있지만 개인의 공정성이나 인과 공정성을 보장하지 않습니다.
- 낮은 raw exposure gap은 비개인화 popularity 추천으로도 얻을 수 있습니다. utility gap, preference residual, coverage를 함께 봐야 합니다.
- GRL이 포함돼 있어도 이번 실험의 linear post-hoc attacker는 더 강해졌습니다. 사이트는 성별을 입력받거나 adversary를 실행하지 않지만, likes/latent에 성별 상관 신호가 없다고 주장할 수 없습니다.
- 배포 사이트의 모델 배지는 그래서 “공정성 정규화 후보”를 뜻하며 “공정함이 증명된 모델”을 뜻하지 않습니다.
