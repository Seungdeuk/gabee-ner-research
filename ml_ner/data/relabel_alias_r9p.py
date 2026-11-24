"""기존 학습 데이터를 R9 정책 r9-1.1 로 재라벨한다(원본은 보존, 전후 차이 기록).

정책 r9-1.1: ``이름+가족호칭``은 **이름 부분만 PERSON**, 호칭은 보호 라벨 밖(라벨 없음). PERSON_ALIAS 는 실제 별명 자체만.
변환 규칙(결정적, 마지막 글자를 기계적으로 자르지 않는다):
  1. PERSON_ALIAS 구간 값에서 가족 호칭 목록 중 가장 긴 것을 오른쪽 끝에서 찾는다(값이 호칭 하나뿐이면 라벨 오류로 격리).
  2. 남은 앞부분(공백 제거 전 원문)이 이름 풀(PERSON)이나 별명 풀(PERSON_ALIAS)에 있는지 확인한다. 둘 다 아니거나, 앞부분 안에
     공백이 있거나, 문장에서 앞부분 바로 뒤에 호칭이 원문 그대로 이어지지 않으면 **변환하지 않고 격리**한다(추측하지 않는다).
  3. 새 구간 = 앞부분만(시작 위치 동일). 같은 문장의 다른 구간은 그대로 둔다.
호칭이 없는 별명(콩이 등)과 PERSON 은 건드리지 않는다. 입력 폴더는 읽기만 하고 출력 폴더에 같은 파일 이름으로 쓴다(KLUE·R8 신규 파일은 그대로 복사).

사용: ``python relabel_alias_r9p.py --in-dir <data_r8> --out-dir <data_r9> --log <relabel_log.json>``
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth  # noqa: E402

KIN_EXTRA = ["울 엄마", "큰아들", "막내딸", "시어머니", "장모님", "작은이모", "큰누나", "외삼촌", "외할머니", "외할아버지", "누나", "오빠", "삼촌", "형", "동생"]
BASELINE_GIVEN = ["민수", "영희", "철수", "지훈", "수빈", "현우", "서연", "도윤", "재민", "서하", "은서", "태윤", "승현", "유진", "다온", "시우", "주원", "예린"]
BASELINE_GIVEN_OPEN = ["지우", "서아", "하나", "다래", "보라", "유나", "소라", "미래", "채아", "하라", "서린", "리아", "도아", "예나", "시아", "나유"]
BASELINE_NICK = ["콩이", "밤톨이", "똘이", "찡이", "뽀미", "쭈니", "몽이", "솜이", "쭈쭈", "달봉이", "새롬이", "다솜이", "꾸꾸", "몽실이", "쫑이"]
RELABEL_FILES = ["baseline_train", "baseline_val", "synth_train", "synth_val", "synth_val_core", "synth_val_aug"]


def kin_list():
    ks = set(synth.KINS) | set(synth.KIN_TERMS) | set(KIN_EXTRA)
    return sorted(ks, key=len, reverse=True)


def pools():
    names = set(synth.GIVEN2) | set(synth.GIVEN1) | set(BASELINE_GIVEN) | set(BASELINE_GIVEN_OPEN)
    nicks = set(synth.NICKS) | set(BASELINE_NICK)
    return names, nicks


def span_label(s):
    return s.get("label") or s.get("kind")


def relabel_value(text, s, kins, names, nicks):
    """(새 구간 dict 또는 None, 사유). None 이면 변환 안 함(사유가 'no_kin' 이면 정상 유지, 그 외는 격리)."""
    v = text[s["start"]:s["end"]]
    # 공백이 든 호칭("울 엄마")은 앞이 공백이거나 값의 시작일 때만 호칭으로 본다(이름 끝 글자와 겹쳐 이름을 잘못 자르는 것을 막는다)
    k = next((k for k in kins if v.endswith(k) and (" " not in k or len(v) == len(k) or v[len(v) - len(k) - 1] == " ")), None)
    if k is None:
        return None, "no_kin"
    head = v[: len(v) - len(k)]
    name = head.rstrip()
    if not name:
        return None, "kin_only_value"
    if " " in name:
        return None, "inner_space_in_name_part"
    if len(name) < 2:
        return None, "one_char_name_part"
    if name in nicks:
        label = "PERSON_ALIAS"
    elif name in names:
        label = "PERSON"
    else:
        return None, "name_part_not_in_pool"
    end = s["start"] + len(name)
    # 문맥 확인: 이름 바로 뒤(공백 허용)에 호칭이 원문 그대로 이어져야 한다
    tail = text[end:s["end"]].lstrip(" ")
    if tail != k:
        return None, "context_tail_mismatch"
    new = dict(s)
    new["end"] = end
    if "label" in new:
        new["label"] = label
    else:
        new["kind"] = label
    return new, "converted:" + ("space" if text[end:s["end"]].startswith(" ") else "attached")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    kins, (names, nicks) = kin_list(), pools()
    log = {"policy": "r9-1.1", "in_dir": str(a.in_dir), "out_dir": str(a.out_dir), "files": {}, "changes": [], "quarantine": []}
    for f in sorted(a.in_dir.glob("*.jsonl")):
        src_sha = sha(f)
        if f.stem not in RELABEL_FILES:
            shutil.copyfile(f, a.out_dir / f.name)
            log["files"][f.name] = {"action": "copied_unchanged", "sha256_src": src_sha[:16]}
            continue
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        stat = Counter()
        out_rows = []
        for r in rows:
            text = r["text"]
            new_spans, changed = [], []
            for s in r["spans"]:
                if span_label(s) != "PERSON_ALIAS":
                    new_spans.append(s)
                    continue
                stat["alias_before"] += 1
                new, why = relabel_value(text, s, kins, names, nicks)
                if new is not None:
                    new_spans.append(new)
                    stat[why] += 1
                    changed.append({"before": {"label": span_label(s), "start": s["start"], "end": s["end"]}, "after": {"label": span_label(new), "start": new["start"], "end": new["end"]}, "kind": why})
                else:
                    new_spans.append(s)
                    if why == "no_kin":
                        stat["kept_nickname"] += 1
                    else:
                        stat["quarantined:" + why] += 1
                        log["quarantine"].append({"file": f.name, "id": r["id"], "reason": why})
            if changed:
                r = dict(r)
                r["spans"] = new_spans
                r["relabel_r9p"] = "alias_kin_split"
                for c in changed:
                    log["changes"].append({"file": f.name, "id": r["id"], **c})
            out_rows.append(r)
        (a.out_dir / f.name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
        log["files"][f.name] = {"action": "relabeled", "rows": len(rows), "sha256_src": src_sha[:16], "sha256_out": sha(a.out_dir / f.name)[:16], **dict(stat)}
    a.log.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in log["files"].items() if v["action"] == "relabeled"}, ensure_ascii=False))
    print("quarantined", len(log["quarantine"]), "changes", len(log["changes"]))


if __name__ == "__main__":
    main()
