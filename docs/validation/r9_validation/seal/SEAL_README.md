# R9 v1.2 독립 평가셋 봉인 폴더

상태: SEALED (2026-11-19, 사용자 승인). 이 폴더의 파일은 읽기 전용이며 수정·삭제·덮어쓰기 금지.

- 데이터: `r9_representative_v1.2.jsonl`(1,600문장), `r9_stress_v1.2.jsonl`(800문장)
- 해시·검증 결과·승인 기록: `R9_v1.2_SEAL_MANIFEST.json`, `MANIFEST.sha256`
- 확인: `python apps/api/scripts/verify_r9_seal.py`

사용 제한은 매니페스트의 `usage_restrictions` 와 `docs/security/NER_R9_PREREGISTRATION.md` §6, §7, §10 을 따른다. 최종 후보 확정 전에는 열람·예측 실행·튜닝에 쓰지 않는다.
