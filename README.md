# Fair Unseen-User Movie Recommendation

MovieLens 1M의 **처음 보는 사용자**에게, 좋아한 영화 몇 편만 받아 성별을 입력받지 않고 Top-N 영화를 추천하는 실험 프로젝트입니다. 핵심 비교는 같은 MultVAE 백본을 쓰는 다음 두 모델입니다.

- `MultVAE`: 공정성 정규화가 없는 naive 기준선
- `Adv-MultVAE + soft GBS`: GRL adversary와 성별 간 예상 장르 노출 차이를 줄이는 differentiable regularizer를 추가한 모델

## 결과 상태

전체 ML-1M 학습과 seed 42/43/44 반복 평가를 완료했습니다. 사이트가 쓰는 seed 42 모델은 support 10에서 naive 대비 NDCG@10이 1.53% 낮아지는 대신 성별 NDCG gap 28.97%, 장르 노출 L1 11.49%, support-conditioned residual 13.19%가 감소했습니다. 그러나 15개 seed×support 조합 전체에서는 성별 NDCG gap이 평균적으로 소폭 악화했고 post-hoc gender probe도 악화했습니다.

따라서 최종 모델은 **장르 노출 공정성을 일부 개선한 후보**이지, 성별 공정성이 보장되거나 성별 정보가 제거된 모델이 아닙니다. 상세 표·ablation·해석 경계는 [RESULTS.md](RESULTS.md), 사이트와 연결된 원본 수치는 [`artifacts/metrics/comparison.json`](artifacts/metrics/comparison.json), 3-seed 요약은 [`artifacts/metrics/multiseed_summary.json`](artifacts/metrics/multiseed_summary.json)에 있습니다.

## 왜 DEE가 아니라 Top-N 평가와 결과 공정성 정규화인가

이전 접근은 전체 사용자-아이템 평점 행렬을 완성하고, 임계값으로 이진화한 예측에 DEE 정규화를 적용했습니다. 그 방식은 이미 학습된 사용자 좌표를 전제로 하므로 신규 사용자가 들어올 때마다 별도 적응이 필요하고, DEE가 낮더라도 실제 추천 목록의 품질이나 노출 격차가 함께 좋아졌다고 보장하기 어렵습니다.

이 프로젝트는 문제를 **implicit-feedback Top-N 추천**으로 다시 정의합니다.

1. 평점 4점 이상만 positive interaction으로 사용합니다.
2. 사용자 ID 임베딩 대신 좋아한 영화의 binary multi-hot 벡터를 amortized encoder에 입력합니다.
3. 평가 사용자의 최초 N개 positive를 support, 이후 positive를 query로 삼아 실제 신규 사용자 상황을 모사합니다.
4. NDCG/Recall/HitRate와 성별별 격차, 장르 노출, 잠재표현 gender probe를 함께 측정합니다.
5. Fair arm은 학습 중에만 성별 label을 사용해 GRL adversary와 soft genre-balance score(soft GBS)를 학습합니다.

MultVAE 손실에 adversary 분류 손실을 더하되, GRL이 encoder로 전달되는 분류 gradient의 부호를 뒤집습니다.

~~~text
L = L_multinomial + beta * KL
  + lambda_adv * CE(adversary(GRL(mu)), gender)
  + gamma * soft_GBS(scores, gender, genres)
encoder에 전달되는 adversary gradient = -grl_lambda * gradient
~~~

분류기는 성별을 예측하려 하고 encoder는 그 예측을 어렵게 만드는 방향으로 학습됩니다. soft GBS는 temperature-softmax 점수로 계산한 M/F의 기대 장르 노출 분포 L1을 줄입니다. 평가는 이 surrogate와 분리해 hard Top-10 장르 노출을 측정합니다. 두 항 모두 공정성의 충분조건은 아닙니다.

## 모델 후보

`scripts/benchmark.py`는 동일한 user-disjoint 평가에서 네 후보를 비교합니다.

| 모델 | 역할 | 신규 사용자 적응 | 공정성 개입 |
| --- | --- | --- | --- |
| Popularity | 비개인화 sanity baseline | support 없이도 가능 | 없음 |
| ItemKNN | 간단하고 결정적인 collaborative baseline | 좋아한 아이템 이웃으로 즉시 계산 | 없음 |
| MultVAE | naive neural baseline | binary support를 encoder에 입력 | 없음 |
| Adv-MultVAE + soft GBS | 중심 실험 모델 | MultVAE와 동일 | `mu`의 GRL adversary + 예상 장르 노출 parity |

두 MultVAE arm은 같은 데이터, few-shot 입력 생성 방식, 자연스러운 성별 minibatch 분포를 사용합니다. naive checkpoint는 validation NDCG만으로 선택합니다. fair checkpoint는 최고 validation NDCG의 3% 이내에서 NDCG 성별 gap, hard 장르 노출 L1, support-conditioned residual, probe leakage 순으로 선택합니다. test user는 checkpoint 선택에 사용하지 않습니다.

## 모델과 serving 경계

모델 입력은 shape `[batch, num_items]`의 positive-item 벡터 하나뿐입니다. 학습된 user ID table이 없기 때문에 catalog 안의 좋아한 영화가 있는 신규 사용자를 즉시 encode할 수 있습니다.

성별 label의 사용 범위는 다음으로 제한됩니다.

- Adv-MultVAE adversary와 soft GBS 학습
- validation checkpoint 선택을 위한 결과 공정성 지표와 representation probe
- test 결과의 사후 group audit

웹과 CLI serving에는 성별 field가 없습니다. 성별을 질문하거나 추론해서 추천기에 전달하지도 않습니다. 다만 입력 영화 자체에 성별과 상관된 취향 신호가 남을 수 있으므로, 이것이 demographic 정보의 완전한 제거를 뜻하지는 않습니다.

## User-disjoint cold-start protocol

MovieLens 사용자 6,040명을 성별 비율을 보존해 train/validation/test = 80/10/10으로 한 번 분할합니다. 한 사용자는 정확히 한 split에만 속합니다. 상세 수와 해시는 [DATA_PROVENANCE.md](DATA_PROVENANCE.md)에 기록되어 있습니다.

### 학습

- train 사용자만 사용합니다.
- 평점 4점 이상을 binary positive로 변환합니다.
- 매 epoch·사용자마다 seed로 재현 가능한 support 크기 3, 5, 10, 20 중 하나를 뽑습니다.
- 기본 설정에서는 10% 확률로 전체 positive profile을 입력합니다.
- sparse support를 입력하고 해당 train 사용자의 전체 positive profile을 복원하도록 학습합니다.
- baseline과 adversarial arm에 같은 sampling protocol을 적용합니다.

### Validation과 test

- validation/test 사용자는 학습에서 완전히 제외된 unseen user입니다.
- 각 사용자의 timestamp 순 positive 중 최초 N개를 support, 이후 positive를 query로 사용합니다.
- 기본 N은 1, 3, 5, 10, 20입니다.
- timestamp가 같은 경우 원본 row 순서를 안정적인 tie-break로 사용합니다.
- 학습 사용자에게 평점 4점 이상을 한 번도 받지 않은 아이템은 먼저 제외한 뒤 support prefix를 자릅니다.

마지막 규칙 때문에 오프라인 평가의 “N개 좋아한 영화”와 사이트의 입력 의미가 같습니다. query에 평가 가능한 미래 positive가 없는 사용자는 ranking 평균에서 제외되며, 그 수는 결과 JSON에 별도로 기록됩니다.

## Warm catalog

현재 모델의 입출력 좌표는 고정된 MovieLens catalog입니다. 그중 **train 사용자가 평점 4점 이상을 한 번 이상 준 아이템**만 warm catalog로 정의합니다.

- 사이트 검색과 선택은 warm item만 노출합니다.
- 모델 입력과 추천 후보도 warm item만 허용합니다.
- 사용자가 이미 선택한 영화는 추천 점수에서 `-inf`로 mask합니다.
- train-cold item과 그에 해당하는 query positive는 ranking에서 제외하고 개수를 결과에 남깁니다.

이는 학습 신호가 전혀 없는 좌표를 추천 후보로 섞어 생기는 허위 성능을 막기 위한 조치입니다. 현재 구현이 해결하는 cold-start는 **unseen user**이며 unseen item은 아닙니다.

## 평가 지표

단일 공정성 숫자를 목표로 삼지 않고 다음 지표를 함께 봅니다.

| 구분 | 지표 | 해석 |
| --- | --- | --- |
| 추천 효용 | NDCG@10, Recall@10, HitRate@10 | 뒤의 positive 영화를 Top-10에서 얼마나 잘 찾는가 |
| group utility | 위 지표의 M/F별 평균과 absolute gap | 한 성별에서만 효용이 크게 떨어지는가 |
| catalog | coverage | warm catalog 중 추천에 실제 노출된 비율 |
| 장르 노출 | M/F 장르분포 L1, Jensen-Shannon | hard Top-10의 장르 노출 분포가 얼마나 다른가 |
| 증폭 진단 | preference amplification | 추천 격차에서 관측 support 취향 격차를 뺀 기술적 진단값 |
| 표현 누출 | post-hoc gender probe ROC-AUC, balanced accuracy | 고정된 latent `mu`에서 성별이 얼마나 쉽게 복원되는가 |

Probe는 train latent로 학습하고 validation에서 regularization/threshold를 고른 뒤 test에 한 번 보고합니다. AUC가 0.5에 가까운 것은 이 probe가 구분하지 못했다는 뜻일 뿐, 모든 공격이나 모든 공정성 정의에 대한 보장은 아닙니다. 장르 기반 preference amplification도 causal effect가 아니라 관측된 차이의 기술 통계입니다.

## 설치

Python 3.10 이상이 필요합니다. CUDA는 전체 학습에 권장되지만 테스트와 소규모 실행은 CPU에서도 가능합니다.

~~~bash
cd /NHNHOME/WORKSPACE/26msit001_A/optiml/fair_unseen_rec
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,config]"
python -m pytest -q
~~~

기본 데이터 경로는 `data/processed/movielens_1m.npz`입니다. 파일의 출처, fingerprint, 라이선스 주의사항은 [DATA_PROVENANCE.md](DATA_PROVENANCE.md)를 먼저 확인하십시오.

## 학습

전체 baseline과 adversarial 모델을 별도 디렉터리에 학습합니다. GPU 번호는 환경에 맞게 바꾸십시오.

~~~bash
python scripts/train.py \
  --config configs/baseline.yaml \
  --output-dir artifacts/runs/baseline \
  --device cuda:0

python scripts/train.py \
  --config configs/adversarial.yaml \
  --output-dir artifacts/runs/adversarial \
  --device cuda:0
~~~

각 run은 다음을 만듭니다.

- `best.pt`: validation 규칙으로 고른 checkpoint
- `training_summary.json`: protocol, epoch별 validation 지표, 선택 근거
- `candidates/epoch_XXXX.pt`: `selection.keep_candidate_checkpoints=true`일 때만 보존되는 epoch 후보 checkpoint

코드 경로만 빠르게 확인할 때는 smoke config와 epoch override를 사용할 수 있습니다. 이 결과는 최종 비교에 사용하지 마십시오.

~~~bash
python scripts/train.py \
  --config configs/smoke_baseline.yaml \
  --output-dir artifacts/smoke/baseline \
  --device cpu \
  --epochs 1

python scripts/train.py \
  --config configs/smoke_adversarial.yaml \
  --output-dir artifacts/smoke/adversarial \
  --device cpu \
  --epochs 1
~~~

## 개별 평가와 최종 비교

한 checkpoint를 자세히 평가하려면 다음과 같이 실행합니다.

~~~bash
python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/naive_best.pt \
  --output artifacts/metrics/baseline_test.json \
  --device cuda:0

python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/fair_best.pt \
  --output artifacts/metrics/fair_test.json \
  --device cuda:0
~~~

네 후보를 같은 protocol로 비교하고 canonical 결과를 자동 생성합니다.

~~~bash
python scripts/benchmark.py \
  --baseline-checkpoint artifacts/checkpoints/naive_best.pt \
  --adversarial-checkpoint artifacts/checkpoints/fair_best.pt \
  --output artifacts/metrics/comparison.json \
  --device cuda:0
~~~

`scripts/benchmark.py`는 출력 디렉터리가 없으면 만들고 `artifacts/metrics/comparison.json`을 기록합니다. 이 파일에는 protocol, 데이터 fingerprint, warm/cold catalog 수, support 크기별 네 모델의 지표, neural 모델의 checkpoint 정보가 함께 들어갑니다.

최종 판단 시 최소한 다음을 확인하십시오.

1. Fair arm의 NDCG/Recall이 MultVAE에 비해 허용 가능한 범위인가.
2. M/F ranking gap이 특정 support 크기에서 악화되지 않는가.
3. genre exposure와 preference amplification이 함께 개선되는가.
4. gender-probe AUC의 0.5 거리가 줄었는가.
5. 여러 seed에서도 방향이 유지되는가.

한 지표만 좋아졌거나 한 support 크기에서만 좋아졌다면 “공정한 모델”이라는 결론을 내리지 마십시오.

## 신규 사용자 CLI 추천

MovieLens movie ID를 `--like`로 반복해 전달합니다. 성별 인자는 존재하지 않습니다.

~~~bash
python scripts/recommend.py \
  --checkpoint artifacts/checkpoints/fair_best.pt \
  --like 1 \
  --like 260 \
  --like 1196 \
  --top-k 10 \
  --device cpu
~~~

## 사이트 실행

두 기본 checkpoint와 해시가 결합된 audit JSON이 이미 준비되어 있습니다.

~~~bash
streamlit run app.py
~~~

naive 비교 사이트는 별도 포트로 띄울 수 있습니다.

~~~bash
streamlit run app_baseline.py --server.port 8502
~~~

Fair 사이트는 `fair_best.pt`, naive 사이트는 `naive_best.pt`를 기본으로 읽습니다. 환경 변수로 경로를 바꿀 수 있지만 각 앱은 checkpoint의 model type을 검증하며, audit 수치는 현재 checkpoint SHA-256과 일치할 때만 표시합니다. 사이트에서는 warm catalog에서 좋아한 영화 3~20편을 고르고 Top-10 추천을 받습니다. 성별을 입력받지 않으며 선택 내역을 서버 파일에 저장하지 않습니다.

## Hybrid adaptation 방향

현재 MultVAE는 공유 catalog 좌표 덕분에 unseen user에는 inductive하지만, 학습 중 한 번도 positive가 없던 영화나 catalog 밖 신작에는 점수를 학습할 수 없습니다. 다음 단계는 영화 장르·개봉연도·텍스트/포스터 embedding을 사용해 item representation을 만들고, collaborative decoder와 content-conditioned scorer를 결합하는 hybrid 구조입니다.

가능한 확장 순서는 다음과 같습니다.

1. 기존 item index와 metadata embedding을 함께 학습합니다.
2. warm item에서는 collaborative/content score를 calibration해 결합합니다.
3. train-cold 또는 신작에서는 content branch만으로 후보를 생성합니다.
4. 동일한 user-disjoint protocol에 item-cold split을 별도로 추가합니다.
5. content feature가 성별 proxy를 강화하는지 같은 group audit를 반복합니다.

이 저장소에는 아직 hybrid item model이 구현되어 있지 않습니다. 관련 출발점으로 [A Hybrid Variational Autoencoder for Collaborative Filtering](https://arxiv.org/abs/1808.01006)을 참고할 수 있습니다.

## 프로젝트 구조

~~~text
fair_unseen_rec/
├── app.py, app_baseline.py       # 성별 입력 없는 Streamlit UI
├── configs/                      # full/smoke baseline·adversarial 설정
├── data/processed/               # 로컬 전처리 artifact와 split manifest
├── scripts/                      # train/evaluate/benchmark/recommend/multiseed 요약
├── src/fairrec/                  # data, model, training, evaluation, inference
├── tests/                        # core, protocol, CLI/앱 관련 테스트
├── DATA_PROVENANCE.md            # 데이터 출처·해시·분할·사용 조건
├── RESULTS.md                    # 전체 학습 결과·ablation·해석 경계
└── artifacts/                    # checkpoint와 JSON 결과
~~~

## 제한사항

- MovieLens 1M은 2000년 전후의 데이터이므로 현재의 영화 소비를 대표하지 않습니다.
- 성별은 자발적으로 제공된 `M/F` 값이며 검증되지 않았고, non-binary 사용자를 표현하지 못합니다.
- 데이터의 성별 imbalance 때문에 평균과 gap의 불확실성이 그룹마다 다릅니다.
- GRL은 finite-capacity probe가 읽는 정보를 줄일 수 있을 뿐 개인·인과·교차집단 공정성을 보장하지 않으며, 이번 3-seed 실험에서는 실제 probe leakage도 줄이지 못했습니다.
- soft GBS는 장르 노출의 집단 평균만 제어하므로 실제 선호 차이를 지우거나 다른 속성의 격차를 악화할 수 있습니다.
- 성별과 상관된 장르/인기 신호는 likes에 남을 수 있습니다.
- adversarial 학습은 seed와 weight에 민감하며 utility-fairness trade-off가 생길 수 있습니다.
- 오프라인 Top-N 지표는 실제 클릭, 만족, 장기 노출, feedback loop를 측정하지 않습니다.
- warm catalog 제한 때문에 새로운 영화 cold-start는 해결하지 않습니다.
- 최소 한 번의 전체 run만으로 결론 내리지 말고 여러 seed의 평균과 분산을 보고해야 합니다.

## 1차 자료와 공식 구현

- Liang et al., [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814), WWW 2018
- 저자 공식 구현, [dawenl/vae_cf](https://github.com/dawenl/vae_cf)
- Ganin et al., [Domain-Adversarial Training of Neural Networks](https://www.jmlr.org/papers/v17/15-239.html), JMLR 2016 — Gradient Reversal Layer
- Madras et al., [Learning Adversarially Fair and Transferable Representations](https://proceedings.mlr.press/v80/madras18a.html), ICML 2018
- Beigi et al., [Privacy-Aware Recommendation with Private-Attribute Protection Using Adversarial Learning](https://arxiv.org/abs/1807.03521)
- [Unlearning Protected User Attributes in Recommendations with Adversarial Training](https://arxiv.org/abs/2206.04500), SIGIR 2022; [공식 구현](https://github.com/CPJKU/adv-multvae)
- [Providing Previously Unseen Users Fair Recommendations Using Variational Autoencoders](https://arxiv.org/abs/2308.15230), RecSys 2023; [공식 구현](https://github.com/BjornarVass/fair-vae-rec)
- [Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness](https://arxiv.org/abs/2502.17921), WWW 2025
- GroupLens, [MovieLens 1M 공식 페이지](https://grouplens.org/datasets/movielens/1m/)와 [README/사용 조건](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)
- Harper and Konstan, [The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872), ACM TiiS 2015

