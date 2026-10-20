# Data provenance: MovieLens 1M

이 문서는 `fair_unseen_rec`이 사용하는 MovieLens 1M 처리 artifact의 출처, 로컬 계보, 검증 해시, schema, 분할 규칙과 사용상 주의사항을 기록합니다. 기록 기준일은 **2026-08-13**입니다.

## Upstream dataset

- 데이터셋: MovieLens 1M
- 제공자: GroupLens Research, University of Minnesota
- 공식 페이지: <https://grouplens.org/datasets/movielens/1m/>
- 공식 archive URL: <https://files.grouplens.org/datasets/movielens/ml-1m.zip>
- 공식 README와 사용 조건: <https://files.grouplens.org/datasets/movielens/ml-1m-README.txt>
- 권장 인용: Harper, F. M. and Konstan, J. A. (2015), [The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872), ACM Transactions on Interactive Intelligent Systems

공식 배포본은 익명 사용자 6,040명, 영화 약 3,900편, 1~5점 평점 1,000,209건과 timestamp를 포함합니다. 성별·연령·직업 등 demographic 정보는 사용자가 자발적으로 제공했으며 정확성이 검증된 값이 아닙니다.

## Local artifact lineage

현재 프로젝트의 다음 두 파일은 같은 workspace의 선행 프로젝트 `optiml/aidep/data/processed/`에 있던 처리 결과를 **byte-for-byte 복사해 재사용**한 것입니다.

- `data/processed/movielens_1m.npz`
- `data/processed/split_manifest.json`

이 새 프로젝트가 원본 ZIP을 다시 내려받거나 독립적으로 전처리를 재실행해 만든 파일은 아닙니다. 두 위치의 파일이 동일한지는 SHA-256으로 확인했습니다. 따라서 아래 raw-file 해시는 현 로컬 NPZ에서 역산한 값이 아니라 선행 preprocessing manifest가 기록한 값입니다.

### 현재 파일 검증값

| 파일 | 크기 | SHA-256 |
| --- | ---: | --- |
| `movielens_1m.npz` | 약 3.3 MiB | `51ef38f20b07e20b64058dffc96e3d31fbf54f72ace189a38117190cb791a759` |
| `split_manifest.json` | 약 84 KiB | `1f93f597de7ec80d46f0b669b494fcaf081ddbd6c8f4b080ecb7f53fa78fa178` |

로컬에서 다시 확인하려면 다음을 실행합니다.

~~~bash
sha256sum \
  data/processed/movielens_1m.npz \
  data/processed/split_manifest.json
~~~

### Manifest에 기록된 raw 파일

| 원본 파일 | bytes | SHA-256 |
| --- | ---: | --- |
| `movies.dat` | 171,308 | `0140fc2356357c1a851d0f52e893a1e4d3696df632c4141cea8d5bc3d621f0b9` |
| `ratings.dat` | 24,594,131 | `506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f` |
| `users.dat` | 134,368 | `1dc3a95300cb19f7c10e027daf29b8cdf1d908dab65b5edfb45950aa389a10a3` |

현재 로컬 작업 폴더에는 실행과 계보 확인을 위해 압축을 푼 `data/raw/ml-1m/*.dat`와 upstream README가 있습니다. 다만 `.gitignore`가 raw/processed data를 제외하므로 source repository에 자동 포함되는 파일은 아니며, 재배포 시에는 아래 GroupLens 사용 조건을 별도로 확인해야 합니다.

## Processed dataset summary

| 항목 | 값 |
| --- | ---: |
| 사용자 | 6,040 |
| 영화 catalog 좌표 | 3,883 |
| 평점 event | 1,000,209 |
| 장르 | 18 |
| multi-label 장르 membership | 6,408 |
| 사용자별 event 최소 / 중앙값 / 최대 | 20 / 96 / 2,314 |

NPZ는 모든 1~5점 평점을 보존합니다. `rating >= 4`의 implicit positive 변환은 저장 시점이 아니라 현재 프로젝트의 학습·평가 코드에서 수행합니다.

## User-disjoint split

Manifest schema version은 1이고 분할 seed는 `20260729`입니다. 사용자를 먼저 성별로 층화한 뒤 80/10/10으로 분할했으며, 사용자 overlap은 없습니다.

| split | 사용자 | 평점 | M 사용자 | F 사용자 |
| --- | ---: | ---: | ---: | ---: |
| train | 4,832 | 800,159 | 3,465 | 1,367 |
| validation | 604 | 97,986 | 433 | 171 |
| test | 604 | 102,064 | 433 | 171 |

성별 code는 `M=0`, `F=1`, `U=2`입니다. `U`는 schema 호환을 위한 reserved code이며 이 artifact의 세 split에는 `U` 사용자가 없습니다. 이 이진 값은 MovieLens가 제공한 제한적인 필드일 뿐이며, 실제 성 정체성의 완전하거나 검증된 표현으로 해석하면 안 됩니다.

`split_manifest.json`에는 위 집계와 각 split의 원래 MovieLens user ID가 들어 있습니다. history는 사용자 안에서 timestamp 오름차순으로 정렬하고, timestamp가 같으면 원본 rating row 순서로 정렬했습니다.

## NPZ schema

| array | shape | dtype | 의미 |
| --- | --- | --- | --- |
| `movie_ids` | `(3883,)` | `int32` | 원래 MovieLens movie ID |
| `movie_titles` | `(3883,)` | Unicode | 영화명 |
| `movie_years` | `(3883,)` | `int16` | 제목에서 파싱한 개봉연도 |
| `movie_years_normalized` | `(3883,)` | `float32` | 선행 pipeline의 정규화 연도 feature |
| `movie_genres` | `(3883, 18)` | `uint8` | multi-hot 장르 matrix |
| `dee_item_genre_membership` | `(6408, 2)` | `int32` | 선행 DEE 실험 호환용 item-genre membership |
| `user_ids` | `(6040,)` | `int32` | 원래 익명 MovieLens user ID |
| `user_genders` | `(6040,)` | `uint8` | `M=0`, `F=1`, reserved `U=2` |
| `user_ages` | `(6040,)` | `int16` | MovieLens age category |
| `user_occupations` | `(6040,)` | `int16` | MovieLens occupation code |
| `user_splits` | `(6040,)` | `uint8` | train=0, validation=1, test=2 |
| `user_ptr` | `(6041,)` | `int64` | 사용자 history CSR-style offset |
| `rating_item_indices` | `(1000209,)` | `int32` | contiguous catalog index |
| `rating_values` | `(1000209,)` | `uint8` | 원래 1~5점 평점 |
| `rating_timestamps` | `(1000209,)` | `int64` | Unix timestamp |

현재 `fairrec` loader와 모델은 영화 ID/title/year/genre, user ID/gender/split, history pointer와 rating arrays만 사용합니다. `movie_years_normalized`, `dee_item_genre_membership`, `user_ages`, `user_occupations`는 상속된 schema field이며 현재 추천 모델 입력으로 사용하지 않습니다. 특히 serving에는 gender, age, occupation, user ID가 전달되지 않습니다.

## Runtime derivations

현재 프로젝트는 처리 artifact를 다음과 같이 사용합니다.

1. train 사용자에 대해 평점 4점 이상을 binary positive로 변환합니다.
2. train positive가 한 건 이상 존재하는 영화만 warm catalog로 만듭니다.
3. validation/test에서는 warm positive를 먼저 시간순으로 거른 뒤 최초 N개를 support로 사용합니다.
4. 그 뒤에 발생한 positive를 query 후보로 만들고, train-cold query positive는 ranking에서 제외하되 개수를 JSON에 기록합니다.
5. 모델에는 positive item multi-hot 벡터만 전달합니다. user ID와 demographic field는 모델 feature가 아닙니다.

Checkpoint에는 NPZ의 SHA-256 fingerprint가 저장됩니다. 평가와 inference loader는 checkpoint가 기대하는 fingerprint와 실제 데이터가 다르면 실패하도록 되어 있습니다. 데이터 파일을 교체하거나 재생성했다면 기존 checkpoint를 그대로 호환된다고 가정하지 마십시오.

## Reproduction boundary

이 저장소에는 공식 ZIP downloader나 raw-to-NPZ 전처리 script가 포함되어 있지 않습니다. 따라서 완전한 raw-to-artifact 재현을 주장하지 않습니다. 재현이 필요하면 다음 조건을 모두 고정한 호환 pipeline이 필요합니다.

- 위 공식 MovieLens 1M source와 raw-file hash
- movie ID에서 contiguous index로의 동일한 mapping
- 사용자 history의 timestamp/원본-row 안정 정렬
- seed `20260729`의 gender-stratified user-disjoint split
- 위 NPZ schema와 dtype
- 처리 결과 SHA-256 확인

artifact hash가 달라지면 새 데이터로 모든 model checkpoint를 다시 학습하고 비교 결과도 다시 생성하는 것이 안전합니다.

## Usage and redistribution

MovieLens 1M의 사용과 재배포에는 GroupLens의 조건이 적용됩니다. 공식 README는 연구 결과에서 GroupLens를 명시하고, 별도 허가 없는 데이터 재배포 및 수익 목적 사용을 제한하며, 보증이나 GroupLens의 endorsement를 제공하지 않는다고 안내합니다. 정확한 현재 조건은 반드시 [공식 README](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)를 직접 확인하십시오.

이 프로젝트의 `.gitignore`는 raw 파일과 `data/processed/`를 기본적으로 제외해 실수로 데이터 artifact를 게시하는 일을 줄입니다. 저장소를 다른 사람에게 배포할 때는 NPZ와 manifest를 무단 포함하지 말고, 사용자가 공식 source에서 적절한 권한과 조건 아래 데이터를 준비하도록 하십시오. 이 문서는 법률 자문이 아닙니다.

## Known data limitations

- 수집 시점이 오래되어 현대의 영화 catalog와 이용 행태를 대표하지 않습니다.
- demographic field는 자발적·미검증 값입니다.
- 성별은 `M/F` 이진 값만 있어 non-binary 정체성을 평가할 수 없습니다.
- M/F 사용자 수가 불균형해 작은 그룹 지표의 분산이 더 클 수 있습니다.
- 사용자 익명화와 group audit가 재식별 위험의 부재를 뜻하지 않습니다.
- group-level offline 차이는 개인 공정성, 인과 공정성, 실제 서비스 만족도를 직접 측정하지 않습니다.

