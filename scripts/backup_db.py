#!/usr/bin/env python3
"""PostgreSQL 备份与恢复。

走 ``docker compose exec`` 调用容器内的 pg_dump/psql，宿主机不需要装
客户端，也不会因为客户端版本与服务器不一致而失败。

用法：
    python scripts/backup_db.py backup [--keep 7]
    python scripts/backup_db.py restore backups/fitness_assistant-20260101-120000.sql.gz

恢复是覆盖式操作（``--clean --if-exists``），执行前会要求手动确认。
"""

from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEEP = 7


def _compose(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), *args]


def _service_container(service: str = "postgres") -> str:
    env_override = os.getenv("PG_CONTAINER")
    if env_override:
        return env_override
    try:
        out = subprocess.check_output(
            _compose("ps", "-q", service), text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        sys.exit(f"无法定位 {service} 容器：{exc}")
    if not out:
        sys.exit(f"{service} 容器未运行，请先 docker compose up -d")
    return out.splitlines()[0]


def _pg_env() -> tuple[str, str]:
    return (
        os.getenv("POSTGRES_USER", "ai_fitness"),
        os.getenv("POSTGRES_DB", "fitness_assistant"),
    )


def backup(out_dir: Path, keep: int) -> Path:
    user, db = _pg_env()
    container = _service_container()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = out_dir / f"{db}-{stamp}.sql.gz"

    proc = subprocess.run(
        ["docker", "exec", container, "pg_dump",
         "-U", user, "-d", db, "--clean", "--if-exists", "--no-owner"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        sys.exit(f"pg_dump 失败：{proc.stderr.decode('utf-8', 'replace')}")

    with gzip.open(target, "wb") as fh:
        fh.write(proc.stdout)

    # 只保留最近 N 份，避免备份目录无限增长。
    backups = sorted(out_dir.glob(f"{db}-*.sql.gz"))
    for stale in backups[:-keep] if keep > 0 else []:
        stale.unlink(missing_ok=True)

    return target


def restore(path: Path, assume_yes: bool = False) -> None:
    if not path.exists():
        sys.exit(f"找不到备份文件：{path}")
    user, db = _pg_env()
    container = _service_container()

    if not assume_yes:
        answer = input(
            f"将用 {path.name} 覆盖数据库 {db} 的现有结构，确认继续？输入 yes：")
        if answer.strip().lower() != "yes":
            sys.exit("已取消")

    # 把 gzip 解压后的数据流直接喂给 psql 的 stdin，不落中间文件。
    with gzip.open(path, "rb") as fh:
        proc = subprocess.run(
            ["docker", "exec", "-i", container, "psql", "-U", user, "-d", db,
             "-v", "ON_ERROR_STOP=1"],
            stdin=fh, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        sys.exit(f"psql 恢复失败：{proc.stderr.decode('utf-8', 'replace')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PostgreSQL 备份/恢复")
    sub = parser.add_subparsers(dest="command", required=True)

    backup_cmd = sub.add_parser("backup", help="生成一份 gzip 压缩的逻辑备份")
    backup_cmd.add_argument("--out", default=str(ROOT / "backups"))
    backup_cmd.add_argument("--keep", type=int, default=DEFAULT_KEEP)

    restore_cmd = sub.add_parser("restore", help="用备份覆盖当前数据库")
    restore_cmd.add_argument("path")
    restore_cmd.add_argument("--yes", action="store_true", help="跳过交互确认")

    args = parser.parse_args()
    if args.command == "backup":
        target = backup(Path(args.out), args.keep)
        print(target)
    else:
        restore(Path(args.path), assume_yes=args.yes)
        print("恢复完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
