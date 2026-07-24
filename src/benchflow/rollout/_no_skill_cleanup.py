"""Fail-closed sandbox cleanup for no-skill rollout skill catalogs.

The destructive checks run through ``/bin/sh`` plus standard base-image
utilities.  They deliberately do not depend on Python being installed before
the agent.  Every target is validated before any target is changed.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from textwrap import dedent

NO_SKILL_CLEANUP_SH = dedent(
    r"""
    set -f

    fail() {
        printf '%s\n' \
            "experiment_fidelity/unsafe_skill_cleanup_root: $1: $2" >&2
        exit 86
    }

    prereq() {
        printf '%s\n' \
            "experiment_fidelity/no_skill_cleanup_prerequisite: $1" >&2
        exit 86
    }

    within() {
        [ "$1" = "$2" ] && return 0
        case "$1" in
            "$2"/*) return 0 ;;
        esac
        return 1
    }

    overlaps() {
        within "$1" "$2" || within "$2" "$1"
    }

    resolve_path() {
        cleanup_resolve_cursor=$1
        cleanup_resolve_suffix=
        while [ ! -e "$cleanup_resolve_cursor" ] \
            && [ ! -L "$cleanup_resolve_cursor" ]; do
            cleanup_resolve_leaf=${cleanup_resolve_cursor##*/}
            cleanup_resolve_suffix="/$cleanup_resolve_leaf$cleanup_resolve_suffix"
            cleanup_resolve_cursor=${cleanup_resolve_cursor%/*}
            [ -n "$cleanup_resolve_cursor" ] || cleanup_resolve_cursor=/
        done
        cleanup_resolved=$(readlink -f "$cleanup_resolve_cursor") \
            || fail "$1" "cannot resolve cleanup path"
        if [ "$cleanup_resolved" = / ]; then
            [ -n "$cleanup_resolve_suffix" ] \
                && printf '%s\n' "$cleanup_resolve_suffix" \
                || printf '/\n'
        else
            printf '%s%s\n' "$cleanup_resolved" "$cleanup_resolve_suffix"
        fi
    }

    validate_parent_components() {
        cleanup_parent=${1%/*}
        [ -n "$cleanup_parent" ] || cleanup_parent=/
        while [ "$cleanup_parent" != / ]; do
            [ ! -L "$cleanup_parent" ] \
                || fail "$1" \
                    "intermediate path component is a symlink: $cleanup_parent"
            cleanup_parent=${cleanup_parent%/*}
            [ -n "$cleanup_parent" ] || cleanup_parent=/
        done
    }

    validate_mounts() {
        cleanup_mount_target=$1
        cleanup_mount_resolved=$2
        cleanup_mount_anchor=$3
        cleanup_mountinfo=$4
        [ -r "$cleanup_mountinfo" ] \
            || fail "$cleanup_mount_target" \
                "cannot inspect mounts: $cleanup_mountinfo"
        while IFS= read -r cleanup_mount_line; do
            set -- $cleanup_mount_line
            [ "$#" -ge 5 ] || continue
            cleanup_mounted=$5
            [ "$cleanup_mounted" = / ] && continue
            if overlaps "$cleanup_mount_target" "$cleanup_mounted"; then
                fail "$cleanup_mount_target" \
                    "cleanup overlaps mounted path: $cleanup_mounted"
            fi
            if [ -z "$cleanup_mount_anchor" ] \
                && overlaps "$cleanup_mount_resolved" "$cleanup_mounted"; then
                fail "$cleanup_mount_target" \
                    "resolved cleanup overlaps mounted path: $cleanup_mounted"
            fi
        done < "$cleanup_mountinfo"
    }

    validate_target() {
        cleanup_target=$1
        cleanup_anchor=$2
        cleanup_owner=$3
        cleanup_policy=$4
        cleanup_mountinfo=$5
        shift 5

        case "$cleanup_target" in
            /*) ;;
            *) fail "$cleanup_target" "path must be absolute" ;;
        esac
        [ "$cleanup_target" != / ] \
            || fail "$cleanup_target" "path must be non-root"
        case "$cleanup_target" in
            */../*|*/./*|*/..|*/.) \
                fail "$cleanup_target" "path must be normalized" ;;
        esac
        case "$cleanup_policy" in
            reset|unlink-only) ;;
            *) fail "$cleanup_target" \
                "unknown cleanup policy: $cleanup_policy" ;;
        esac

        validate_parent_components "$cleanup_target"
        cleanup_resolved=$(resolve_path "$cleanup_target")
        validate_mounts \
            "$cleanup_target" "$cleanup_resolved" \
            "$cleanup_anchor" "$cleanup_mountinfo"

        if [ -n "$cleanup_anchor" ]; then
            within "$cleanup_target" "$cleanup_anchor" \
                && [ "$cleanup_target" != "$cleanup_anchor" ] \
                || fail "$cleanup_target" \
                    "agent discovery path escapes its home/workspace anchor"
            cleanup_resolved_anchor=$(resolve_path "$cleanup_anchor")
            cleanup_target_parent=${cleanup_target%/*}
            [ -n "$cleanup_target_parent" ] || cleanup_target_parent=/
            cleanup_resolved_parent=$(resolve_path "$cleanup_target_parent")
            if [ "$cleanup_resolved_parent" = / ]; then
                cleanup_resolved_leaf="/${cleanup_target##*/}"
            else
                cleanup_resolved_leaf="$cleanup_resolved_parent/${cleanup_target##*/}"
            fi
            within "$cleanup_resolved_leaf" "$cleanup_resolved_anchor" \
                || fail "$cleanup_target" \
                    "resolved discovery path escapes its anchor"
        else
            for cleanup_protected in "$@"; do
                cleanup_resolved_protected=$(resolve_path "$cleanup_protected")
                if overlaps "$cleanup_target" "$cleanup_protected" \
                    || overlaps \
                        "$cleanup_resolved" "$cleanup_resolved_protected"; then
                    fail "$cleanup_target" \
                        "cleanup overlaps protected path: $cleanup_protected"
                fi
            done
        fi

        if [ "$cleanup_policy" = unlink-only ] \
            && { [ -e "$cleanup_target" ] || [ -L "$cleanup_target" ]; } \
            && [ ! -L "$cleanup_target" ]; then
            fail "$cleanup_target" \
                "existing workspace discovery path is task-owned data"
        fi
        if [ -n "$cleanup_owner" ]; then
            id -u "$cleanup_owner" >/dev/null 2>&1 \
                || fail "$cleanup_target" \
                    "cleanup owner does not exist: $cleanup_owner"
            id -g "$cleanup_owner" >/dev/null 2>&1 \
                || fail "$cleanup_target" \
                    "cleanup owner has no primary group: $cleanup_owner"
        fi
    }

    validate_distinct_targets() {
        if overlaps "$1" "$2"; then
            fail "$1" "cleanup target overlaps another target: $2"
        fi
    }

    reset_target() {
        cleanup_target=$1
        cleanup_anchor=$2
        cleanup_owner=$3
        cleanup_policy=$4

        if [ -L "$cleanup_target" ]; then
            rm -f "$cleanup_target" \
                || fail "$cleanup_target" "cannot unlink discovery symlink"
        elif [ -e "$cleanup_target" ]; then
            [ "$cleanup_policy" = reset ] \
                || fail "$cleanup_target" \
                    "existing workspace discovery path is task-owned data"
            rm -rf "$cleanup_target" \
                || fail "$cleanup_target" "cannot reset cleanup root"
        fi

        if [ "$cleanup_policy" = reset ]; then
            cleanup_created=
            cleanup_cursor=$cleanup_target
            while [ -n "$cleanup_anchor" ] \
                && [ ! -e "$cleanup_cursor" ] \
                && [ ! -L "$cleanup_cursor" ]; do
                cleanup_created="$cleanup_cursor $cleanup_created"
                [ "$cleanup_cursor" = "$cleanup_anchor" ] && break
                cleanup_cursor=${cleanup_cursor%/*}
                [ -n "$cleanup_cursor" ] || cleanup_cursor=/
            done
            mkdir -p "$cleanup_target" \
                || fail "$cleanup_target" "cannot recreate cleanup root"
            if [ -n "$cleanup_owner" ]; then
                cleanup_uid=$(id -u "$cleanup_owner") \
                    || fail "$cleanup_target" "cannot resolve cleanup owner uid"
                cleanup_gid=$(id -g "$cleanup_owner") \
                    || fail "$cleanup_target" "cannot resolve cleanup owner gid"
                for cleanup_created_path in $cleanup_created; do
                    chown "$cleanup_uid:$cleanup_gid" "$cleanup_created_path" \
                        || fail "$cleanup_target" \
                            "cannot own recreated discovery path"
                done
            fi
        fi

        if [ -e "$cleanup_target" ] || [ -L "$cleanup_target" ]; then
            cleanup_leaked=$(find "$cleanup_target" -name SKILL.md -print 2>/dev/null)
            [ -z "$cleanup_leaked" ] \
                || fail "$cleanup_target" \
                    "SKILL.md remains after no-skill reset"
            cleanup_nonempty=$(find \
                "$cleanup_target" -mindepth 1 -maxdepth 1 -print 2>/dev/null)
            [ -z "$cleanup_nonempty" ] \
                || fail "$cleanup_target" \
                    "catalog remains non-empty after reset"
        fi
        printf '%s\n' "cleared_no_skill_root=$cleanup_target"
    }
    """
).strip()


def build_no_skill_cleanup_cmd(
    targets: Sequence[tuple[str, str | None, str | None, str]],
    *,
    protected_roots: Sequence[str] = (),
    mountinfo_path: str | None = None,
) -> str:
    """Build the dependency-light in-sandbox cleanup command.

    Each target is ``(path, anchor, owner, policy)``.  ``anchor`` is set only
    for an agent's own discovery path, where cleanup is intentionally allowed
    below that exact home/workspace root.  ``owner`` restores sandbox-user
    ownership after the root-run reset.  ``policy="unlink-only"`` protects a
    real workspace directory while still removing a final discovery symlink.
    """

    mountinfo = mountinfo_path or "/proc/self/mountinfo"
    required = ("readlink", "rm", "mkdir", "find", "id", "chown")
    lines = [NO_SKILL_CLEANUP_SH]
    lines.extend(
        f"command -v {command} >/dev/null 2>&1 || "
        f"prereq {shlex.quote(f'required command is unavailable: {command}')}"
        for command in required
    )
    lines.append(
        f"readlink -f / >/dev/null 2>&1 || "
        f"prereq {shlex.quote('readlink -f support is required')}"
    )

    protected_args = " ".join(shlex.quote(root) for root in protected_roots)
    for path, anchor, owner, policy in targets:
        arguments = " ".join(
            shlex.quote(value)
            for value in (
                path,
                anchor or "",
                owner or "",
                policy,
                mountinfo,
            )
        )
        suffix = f" {protected_args}" if protected_args else ""
        lines.append(f"validate_target {arguments}{suffix}")

    for index, (left, *_rest) in enumerate(targets):
        for right, *_ in targets[index + 1 :]:
            lines.append(
                f"validate_distinct_targets {shlex.quote(left)} {shlex.quote(right)}"
            )

    for path, anchor, owner, policy in targets:
        lines.append(
            "reset_target "
            + " ".join(
                shlex.quote(value)
                for value in (path, anchor or "", owner or "", policy)
            )
        )
    return f"/bin/sh -c {shlex.quote(chr(10).join(lines))}"
