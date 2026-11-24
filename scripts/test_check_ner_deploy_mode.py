"""배포 NER 모드 게이트 시험(합성 입력만)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_ner_deploy_mode as g  # noqa: E402

OK_STATUS = {"mode": "required", "active": True, "fingerprint": g.EXPECT_FP}


def svc(env):
    return {"spec": {"template": {"spec": {"containers": [{"env": env}]}}}}


@pytest.mark.parametrize("v", ["auto", "off", "", "requird", "Required ", "true"])
def test_non_required_values_stop_deployment(v):
    ok, why = g.check(g.ner_env(svc([{"name": "PRIVACY_NER_MODE", "value": v}])), OK_STATUS)
    assert not ok and any("required 가 아니다" in w for w in why)


def test_required_with_matching_image_passes():
    ok, why = g.check(g.ner_env(svc([{"name": "PRIVACY_NER_MODE", "value": "required"}])), OK_STATUS)
    assert ok and why == []


def test_missing_env_needs_image_status_and_image_default_must_be_required():
    env = g.ner_env(svc([{"name": "OTHER", "value": "x"}]))
    assert not g.check(env, None)[0]
    assert g.check(env, OK_STATUS)[0]
    assert not g.check(env, {**OK_STATUS, "mode": "auto"})[0]


def test_inactive_or_wrong_fingerprint_stops():
    env = {"PRIVACY_NER_MODE": "required"}
    assert not g.check(env, {**OK_STATUS, "active": False})[0]
    assert not g.check(env, {**OK_STATUS, "fingerprint": "46210943c45f"})[0]


def test_overriding_model_dir_or_file_to_old_values_stops():
    base = {"PRIVACY_NER_MODE": "required"}
    assert not g.check({**base, "PRIVACY_NER_MODEL_DIR": "/srv/data/models/ner-koelectra-small-v2b"}, OK_STATUS)[0]
    assert not g.check({**base, "PRIVACY_NER_MODEL_FILE": "model.int8.onnx"}, OK_STATUS)[0]
    assert g.check({**base, "PRIVACY_NER_MODEL_DIR": g.EXPECT_DIR, "PRIVACY_NER_MODEL_FILE": "model.onnx"}, OK_STATUS)[0]


def test_secret_reference_cannot_be_verified():
    env = g.ner_env(svc([{"name": "PRIVACY_NER_MODE", "valueFrom": {"secretKeyRef": {"name": "n", "key": "k"}}}]))
    assert env["PRIVACY_NER_MODE"] == "<secret-ref>" and not g.check(env, OK_STATUS)[0]


def test_only_privacy_ner_names_are_read_and_revision_shape_supported():
    doc = {"spec": {"containers": [{"env": [{"name": "ANTHROPIC_API_KEY", "value": "SECRET-SHOULD-NOT-APPEAR"}, {"name": "PRIVACY_NER_MODE", "value": "required"}]}]}}
    env = g.ner_env(doc)
    assert env == {"PRIVACY_NER_MODE": "required"} and "SECRET" not in json.dumps(env)


def test_cli_exit_codes_and_no_foreign_values_in_output(tmp_path, capsys):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(svc([{"name": "ANTHROPIC_API_KEY", "value": "SECRET-X"}, {"name": "PRIVACY_NER_MODE", "value": "auto"}])), encoding="utf-8")
    assert g.main(["--service-json", str(p)]) == 1
    assert "SECRET-X" not in capsys.readouterr().out
    p.write_text(json.dumps(svc([{"name": "PRIVACY_NER_MODE", "value": "required"}])), encoding="utf-8")
    st = tmp_path / "st.json"
    st.write_text(json.dumps(OK_STATUS), encoding="utf-8")
    assert g.main(["--service-json", str(p), "--image-status-json", str(st)]) == 0
