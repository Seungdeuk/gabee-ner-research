# ml/ner 기존 테스트 실패 3건 (2026-11-20 기록)

`--save-every-epoch` 변경 전 HEAD(커밋 `43286c13` 기준 임시 worktree)에서도 같은 3건이 실패함을 확인했다(`pytest tests/test_ner_data.py tests/test_ner_export.py`: 3 failed, 21 passed). 이번 변경(R11 준비)으로 생긴 실패가 아니며 이번 범위에서 고치지 않았다.

| 테스트 | 증상 |
|---|---|
| `tests/test_ner_data.py::test_news_title_augment_adds_person_exposure_without_leak` | `KeyError: "kind"` (뉴스 제목 증강 행 형식) |
| `tests/test_ner_data.py::test_only_leak_guard_and_evaluate_reference_test_file` | 기존 실패 |
| `tests/test_ner_export.py::test_files_exist_and_sha256` | `AssertionError` (내보내기 산출물 해시 기록) |

이번 변경 시험: `tests/test_train_save_every_epoch.py` 4건 통과. 전체 `ml/ner/tests`: 35 passed, 3 failed(위 3건).
