from __future__ import annotations

import ast
from collections import Counter
import fnmatch
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_OUTPUT_DIR = "soul"
SOUL_SCHEMA_VERSION = 3
META_ARTIFACTS = (
    "files.jsonl",
    "python_index.jsonl",
    "symbols.jsonl",
    "module_graph.json",
    "entrypoints.jsonl",
    "area_summaries.json",
    "views.json",
    "overview.md",
    "skipped.jsonl",
    "tree.txt",
)
SUMMARY_STOPWORDS = {
    "a",
    "all",
    "and",
    "args",
    "base",
    "build",
    "builder",
    "by",
    "call",
    "class",
    "common",
    "config",
    "current",
    "data",
    "default",
    "def",
    "entry",
    "file",
    "files",
    "from",
    "function",
    "get",
    "helper",
    "id",
    "ids",
    "import",
    "index",
    "item",
    "items",
    "json",
    "key",
    "load",
    "main",
    "manager",
    "model",
    "module",
    "name",
    "new",
    "node",
    "object",
    "path",
    "paths",
    "read",
    "record",
    "response",
    "result",
    "root",
    "run",
    "schema",
    "set",
    "state",
    "summary",
    "system",
    "test",
    "text",
    "tool",
    "type",
    "update",
    "use",
    "value",
    "views",
    "write",
}


@dataclass(frozen=True)
class SoulConfig:
    include: Tuple[str, ...]
    exclude: Tuple[str, ...]
    max_file_bytes: int = 256 * 1024
    output_dir: str = DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class PythonClassInfo:
    name: str
    methods: Tuple[str, ...]


@dataclass(frozen=True)
class PythonImportEntry:
    module: str
    imported_names: Tuple[str, ...]
    level: int
    is_from: bool


@dataclass(frozen=True)
class PythonSymbolInfo:
    name: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    docstring: str = ""
    parent: str = ""


@dataclass(frozen=True)
class PythonAnalysis:
    imports: Tuple[str, ...]
    functions: Tuple[str, ...]
    classes: Tuple[PythonClassInfo, ...]
    constants: Tuple[str, ...]
    module_docstring: str
    parse_error: str = ""
    import_entries: Tuple[PythonImportEntry, ...] = ()
    symbols: Tuple[PythonSymbolInfo, ...] = ()
    has_main_guard: bool = False
    has_main_function: bool = False
    uses_argparse: bool = False


@dataclass(frozen=True)
class SourceFile:
    rel_path: str
    abs_path: Path
    size_bytes: int
    line_count: int
    sha256: str
    language: str
    python_module: str = ""
    python_analysis: Optional[PythonAnalysis] = None


@dataclass(frozen=True)
class SkippedFile:
    rel_path: str
    reason: str
    size_bytes: int = 0


@dataclass(frozen=True)
class SyncResult:
    manifest: Dict[str, Any]
    files: Tuple[SourceFile, ...]
    skipped: Tuple[SkippedFile, ...]
    changed: bool
    copied_files: int
    removed_files: int


def load_config(path: Path) -> SoulConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    include = tuple(_normalize_rel_entry(item) for item in payload.get("include", []))
    exclude = tuple(_normalize_glob(item) for item in payload.get("exclude", []))
    output_dir = _normalize_rel_entry(str(payload.get("output_dir", DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR))
    max_file_bytes = int(payload.get("max_file_bytes", 256 * 1024))
    return SoulConfig(
        include=include,
        exclude=exclude,
        max_file_bytes=max_file_bytes,
        output_dir=output_dir,
    )


def scan_sources(repo_root: Path, config: SoulConfig) -> Tuple[Tuple[SourceFile, ...], Tuple[SkippedFile, ...], Tuple[str, ...]]:
    seen: set[str] = set()
    files: List[SourceFile] = []
    skipped: List[SkippedFile] = []
    missing_includes: List[str] = []

    for entry in config.include:
        if not entry:
            continue
        source = repo_root / entry
        if not source.exists():
            missing_includes.append(entry)
            continue
        if source.is_file():
            candidate_files = [source]
        else:
            candidate_files = sorted(path for path in source.rglob("*") if path.is_file())
        for path in candidate_files:
            rel_path = _to_rel_path(path, repo_root)
            if rel_path in seen:
                continue
            seen.add(rel_path)
            if _matches_any(rel_path, config.exclude):
                continue
            stat = path.stat()
            if stat.st_size > config.max_file_bytes:
                skipped.append(SkippedFile(rel_path=rel_path, reason="max_file_bytes", size_bytes=stat.st_size))
                continue
            payload = path.read_bytes()
            sha256 = hashlib.sha256(payload).hexdigest()
            text = payload.decode("utf-8", errors="replace")
            line_count = len(text.splitlines())
            language = _detect_language(path)
            rel_path = _to_rel_path(path, repo_root)
            python_module = _python_module_name(rel_path) if path.suffix == ".py" else ""
            python_analysis = _analyze_python(text) if path.suffix == ".py" else None
            files.append(
                SourceFile(
                    rel_path=rel_path,
                    abs_path=path,
                    size_bytes=stat.st_size,
                    line_count=line_count,
                    sha256=sha256,
                    language=language,
                    python_module=python_module,
                    python_analysis=python_analysis,
                )
            )

    files.sort(key=lambda item: item.rel_path)
    skipped.sort(key=lambda item: item.rel_path)
    missing_includes.sort()
    return tuple(files), tuple(skipped), tuple(missing_includes)


def sync_soul(repo_root: Path, config: SoulConfig) -> SyncResult:
    files, skipped, missing_includes = scan_sources(repo_root, config)
    output_root = repo_root / config.output_dir
    code_root = output_root / "code"
    meta_root = output_root / "meta"
    output_root.mkdir(parents=True, exist_ok=True)
    code_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    expected_rel_paths = {item.rel_path for item in files}
    current_digest = _digest_sources(files)
    current_config_digest = _digest_config(config)
    symbol_rows = _build_symbol_rows(files)
    module_graph = _build_module_graph(files)
    entrypoints = _build_entrypoints(files)
    area_summaries = _build_area_summaries(files, module_graph, entrypoints, symbol_rows)
    views = _build_views(files, module_graph, entrypoints, area_summaries)
    previous_manifest = _load_json(meta_root / "manifest.json")
    previous_digest = str(previous_manifest.get("source_digest", ""))
    previous_config_digest = str(previous_manifest.get("config_digest", ""))
    previous_schema_version = int(previous_manifest.get("schema_version", 0) or 0)

    manifest = _build_manifest(
        repo_root=repo_root,
        config=config,
        files=files,
        skipped=skipped,
        missing_includes=missing_includes,
        source_digest=current_digest,
        config_digest=current_config_digest,
        symbol_rows=symbol_rows,
        module_graph=module_graph,
        entrypoints=entrypoints,
        area_summaries=area_summaries,
        views=views,
    )

    if (
        previous_schema_version == SOUL_SCHEMA_VERSION
        and previous_digest == current_digest
        and previous_config_digest == current_config_digest
        and _mirror_matches(code_root, expected_rel_paths)
        and _meta_is_complete(meta_root)
    ):
        return SyncResult(
            manifest=manifest,
            files=files,
            skipped=skipped,
            changed=False,
            copied_files=0,
            removed_files=0,
        )

    previous_index = _load_previous_hashes(meta_root / "files.jsonl")
    copied_files = 0
    for item in files:
        target = code_root / item.rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        previous_sha = previous_index.get(item.rel_path)
        if previous_sha == item.sha256 and target.exists():
            continue
        shutil.copy2(item.abs_path, target)
        copied_files += 1

    removed_files = _remove_stale_files(code_root, expected_rel_paths)
    _write_files_index(meta_root / "files.jsonl", files)
    _write_python_index(meta_root / "python_index.jsonl", files)
    _write_symbols_index(meta_root / "symbols.jsonl", symbol_rows)
    _write_json(meta_root / "module_graph.json", module_graph)
    _write_entrypoints_index(meta_root / "entrypoints.jsonl", entrypoints)
    _write_json(meta_root / "area_summaries.json", {"schema_version": SOUL_SCHEMA_VERSION, "areas": area_summaries})
    _write_json(meta_root / "views.json", views)
    _write_skipped_index(meta_root / "skipped.jsonl", skipped)
    (meta_root / "tree.txt").write_text(_render_tree(expected_rel_paths), encoding="utf-8")
    (meta_root / "overview.md").write_text(_render_overview(manifest, views), encoding="utf-8")
    _write_json(meta_root / "manifest.json", manifest)

    return SyncResult(
        manifest=manifest,
        files=files,
        skipped=skipped,
        changed=True,
        copied_files=copied_files,
        removed_files=removed_files,
    )


def watch_soul(repo_root: Path, config: SoulConfig, interval_s: float = 2.0, log=print) -> None:
    while True:
        result = sync_soul(repo_root, config)
        status = "updated" if result.changed else "no changes"
        log(
            "[soul] "
            f"{status}: files={result.manifest['file_count']} "
            f"symbols={result.manifest['symbol_count']} "
            f"edges={result.manifest['internal_dependency_edge_count']} "
            f"copied={result.copied_files} removed={result.removed_files}"
        )
        time.sleep(interval_s)


def _normalize_rel_entry(value: str) -> str:
    raw = value.replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    normalized = str(PurePosixPath(raw))
    if normalized == ".":
        return ""
    return normalized


def _normalize_glob(value: str) -> str:
    normalized = _normalize_rel_entry(value)
    return normalized.rstrip("/")


def _to_rel_path(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _matches_any(rel_path: str, patterns: Sequence[str]) -> bool:
    pure_path = PurePosixPath(rel_path)
    basename = pure_path.name
    for pattern in patterns:
        if not pattern:
            continue
        if pure_path.match(pattern):
            return True
        if "/" not in pattern and fnmatch.fnmatchcase(basename, pattern):
            return True
        if fnmatch.fnmatchcase(rel_path, pattern):
            return True
    return False


def _detect_language(path: Path) -> str:
    suffix = path.suffix.lower()
    language_map = {
        ".json": "json",
        ".md": "markdown",
        ".ps1": "powershell",
        ".py": "python",
        ".sh": "shell",
        ".txt": "text",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    if suffix in language_map:
        return language_map[suffix]
    if suffix:
        return suffix.lstrip(".")
    return "text"


def _python_module_name(rel_path: str) -> str:
    path = PurePosixPath(rel_path)
    if path.suffix != ".py":
        return ""
    parts = list(path.parts)
    if not parts:
        return ""
    if parts[0] == "src":
        parts = parts[1:]
    stem = path.stem
    if stem == "__init__":
        parts = parts[:-1]
    else:
        parts[-1] = stem
    return ".".join(part for part in parts if part)


def _analyze_python(source: str) -> PythonAnalysis:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return PythonAnalysis(imports=(), functions=(), classes=(), constants=(), module_docstring="", parse_error=str(exc))

    imports: List[str] = []
    import_entries: List[PythonImportEntry] = []
    functions: List[str] = []
    classes: List[PythonClassInfo] = []
    constants: List[str] = []
    symbols: List[PythonSymbolInfo] = []
    seen_constant_symbols: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                import_entries.append(PythonImportEntry(module=alias.name, imported_names=(), level=0, is_from=False))
            continue
        if isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            imports.append(module or ".")
            import_entries.append(
                PythonImportEntry(
                    module=node.module or "",
                    imported_names=tuple(alias.name for alias in node.names),
                    level=node.level,
                    is_from=True,
                )
            )
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
            symbols.append(
                PythonSymbolInfo(
                    name=node.name,
                    qualname=node.name,
                    kind="function",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    docstring=ast.get_docstring(node) or "",
                )
            )
            continue
        if isinstance(node, ast.ClassDef):
            methods = tuple(
                item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            classes.append(PythonClassInfo(name=node.name, methods=methods))
            symbols.append(
                PythonSymbolInfo(
                    name=node.name,
                    qualname=node.name,
                    kind="class",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    docstring=ast.get_docstring(node) or "",
                )
            )
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                symbols.append(
                    PythonSymbolInfo(
                        name=item.name,
                        qualname=f"{node.name}.{item.name}",
                        kind="method",
                        start_line=item.lineno,
                        end_line=getattr(item, "end_lineno", item.lineno),
                        docstring=ast.get_docstring(item) or "",
                        parent=node.name,
                    )
                )
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                constants.extend(_constant_names(target))
                for name in _constant_names(target):
                    if name in seen_constant_symbols:
                        continue
                    seen_constant_symbols.add(name)
                    symbols.append(
                        PythonSymbolInfo(
                            name=name,
                            qualname=name,
                            kind="constant",
                            start_line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                        )
                    )
            continue
        if isinstance(node, ast.AnnAssign):
            constants.extend(_constant_names(node.target))
            for name in _constant_names(node.target):
                if name in seen_constant_symbols:
                    continue
                seen_constant_symbols.add(name)
                symbols.append(
                    PythonSymbolInfo(
                        name=name,
                        qualname=name,
                        kind="constant",
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                    )
                )

    return PythonAnalysis(
        imports=tuple(sorted(dict.fromkeys(imports))),
        import_entries=tuple(import_entries),
        functions=tuple(functions),
        classes=tuple(classes),
        constants=tuple(sorted(dict.fromkeys(constants))),
        module_docstring=ast.get_docstring(tree) or "",
        parse_error="",
        symbols=tuple(symbols),
        has_main_guard=_has_main_guard(tree),
        has_main_function="main" in functions,
        uses_argparse=_uses_argparse(tree),
    )


def _constant_names(node: ast.AST) -> List[str]:
    names: List[str] = []
    if isinstance(node, ast.Name) and node.id.isupper():
        names.append(node.id)
    if isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            names.extend(_constant_names(item))
    return names


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        if _is_main_guard_test(node.test):
            return True
    return False


def _is_main_guard_test(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False
    return (
        _is_name_main(node.left) and _is_main_literal(node.comparators[0])
    ) or (
        _is_name_main(node.comparators[0]) and _is_main_literal(node.left)
    )


def _is_name_main(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "__name__"


def _is_main_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "__main__"


def _uses_argparse(tree: ast.Module) -> bool:
    argparse_aliases: set[str] = set()
    parser_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "argparse":
                    argparse_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "argparse":
            for alias in node.names:
                if alias.name == "ArgumentParser":
                    parser_aliases.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
            if isinstance(func.value, ast.Name) and func.value.id in argparse_aliases:
                return True
        if isinstance(func, ast.Name) and func.id in parser_aliases:
            return True
    return False


def _digest_sources(files: Sequence[SourceFile]) -> str:
    payload = [{"path": item.rel_path, "sha256": item.sha256} for item in files]
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _digest_config(config: SoulConfig) -> str:
    payload = {
        "include": list(config.include),
        "exclude": list(config.exclude),
        "max_file_bytes": config.max_file_bytes,
        "output_dir": config.output_dir,
        "schema_version": SOUL_SCHEMA_VERSION,
    }
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return {}


def _load_previous_hashes(path: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    if not path.exists():
        return hashes
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            rel_path = str(payload.get("path", ""))
            sha256 = str(payload.get("sha256", ""))
            if rel_path and sha256:
                hashes[rel_path] = sha256
    return hashes


def _mirror_matches(code_root: Path, expected_rel_paths: Iterable[str]) -> bool:
    expected = {str(PurePosixPath(item)) for item in expected_rel_paths}
    if not code_root.exists():
        return not expected
    current = {
        path.relative_to(code_root).as_posix()
        for path in code_root.rglob("*")
        if path.is_file()
    }
    return current == expected


def _meta_is_complete(meta_root: Path) -> bool:
    return all((meta_root / name).exists() for name in META_ARTIFACTS)


def _remove_stale_files(code_root: Path, expected_rel_paths: Iterable[str]) -> int:
    expected = {str(PurePosixPath(item)) for item in expected_rel_paths}
    removed = 0
    if not code_root.exists():
        return removed
    for path in sorted((item for item in code_root.rglob("*") if item.is_file()), reverse=True):
        rel_path = path.relative_to(code_root).as_posix()
        if rel_path in expected:
            continue
        path.unlink()
        removed += 1
    for path in sorted((item for item in code_root.rglob("*") if item.is_dir()), reverse=True):
        if any(path.iterdir()):
            continue
        path.rmdir()
    return removed


def _write_files_index(path: Path, files: Sequence[SourceFile]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in files:
            payload = {
                "path": item.rel_path,
                "mirror_path": f"code/{item.rel_path}",
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "line_count": item.line_count,
                "language": item.language,
                "has_python_index": item.python_analysis is not None,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_python_index(path: Path, files: Sequence[SourceFile]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in files:
            analysis = item.python_analysis
            if analysis is None:
                continue
            payload = {
                "path": item.rel_path,
                "imports": list(analysis.imports),
                "functions": list(analysis.functions),
                "classes": [{"name": cls.name, "methods": list(cls.methods)} for cls in analysis.classes],
                "constants": list(analysis.constants),
                "module_docstring": analysis.module_docstring,
                "parse_error": analysis.parse_error,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_symbols_index(path: Path, symbols: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in symbols:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def _write_entrypoints_index(path: Path, entrypoints: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in entrypoints:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def _write_skipped_index(path: Path, skipped: Sequence[SkippedFile]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in skipped:
            payload = {
                "path": item.rel_path,
                "reason": item.reason,
                "size_bytes": item.size_bytes,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _render_tree(paths: Iterable[str]) -> str:
    tree: Dict[str, Dict[str, Any]] = {}
    for rel_path in sorted(paths):
        node = tree
        parts = [part for part in rel_path.split("/") if part]
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if parts:
            node.setdefault(parts[-1], {})

    lines: List[str] = []

    def walk(node: Dict[str, Any], depth: int) -> None:
        for name in sorted(node):
            child = node[name]
            indent = "  " * depth
            if child:
                lines.append(f"{indent}{name}/")
                walk(child, depth + 1)
            else:
                lines.append(f"{indent}{name}")

    walk(tree, 0)
    return "\n".join(lines) + ("\n" if lines else "")


def _build_symbol_rows(files: Sequence[SourceFile]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in files:
        analysis = item.python_analysis
        if analysis is None:
            continue
        for symbol in analysis.symbols:
            rows.append(
                {
                    "path": item.rel_path,
                    "module": item.python_module,
                    "name": symbol.name,
                    "qualname": symbol.qualname,
                    "kind": symbol.kind,
                    "parent": symbol.parent,
                    "start_line": symbol.start_line,
                    "end_line": symbol.end_line,
                    "docstring": symbol.docstring,
                }
            )
    rows.sort(key=lambda item: (str(item["path"]), int(item["start_line"]), str(item["qualname"])))
    return rows


def _build_module_graph(files: Sequence[SourceFile]) -> Dict[str, Any]:
    python_files = [item for item in files if item.python_analysis is not None and item.python_module]
    module_map = {item.python_module: item for item in python_files}
    nodes: List[Dict[str, Any]] = []
    for item in python_files:
        analysis = item.python_analysis
        assert analysis is not None
        nodes.append(
            {
                "module": item.python_module,
                "path": item.rel_path,
                "area": _analysis_area(item.rel_path),
                "package": _module_package(item.python_module, _is_package_file(item.rel_path)),
                "symbol_count": len(analysis.symbols),
                "has_main_guard": analysis.has_main_guard,
                "uses_argparse": analysis.uses_argparse,
                "module_docstring": _first_line(analysis.module_docstring),
            }
        )

    edges_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in python_files:
        analysis = item.python_analysis
        assert analysis is not None
        for entry in analysis.import_entries:
            for target_module in _resolve_import_targets(entry, item, module_map):
                if target_module == item.python_module:
                    continue
                target = module_map[target_module]
                key = (item.python_module, target_module)
                edge = edges_by_key.get(key)
                if edge is None:
                    edge = {
                        "source_module": item.python_module,
                        "source_path": item.rel_path,
                        "source_area": _analysis_area(item.rel_path),
                        "target_module": target_module,
                        "target_path": target.rel_path,
                        "target_area": _analysis_area(target.rel_path),
                        "_kinds": set(),
                        "_levels": set(),
                        "_imported_names": set(),
                    }
                    edges_by_key[key] = edge
                edge["_kinds"].add("from" if entry.is_from else "import")
                edge["_levels"].add(entry.level)
                edge["_imported_names"].update(name for name in entry.imported_names if name)

    edges: List[Dict[str, Any]] = []
    for key in sorted(edges_by_key):
        edge = edges_by_key[key]
        edges.append(
            {
                "source_module": edge["source_module"],
                "source_path": edge["source_path"],
                "source_area": edge["source_area"],
                "target_module": edge["target_module"],
                "target_path": edge["target_path"],
                "target_area": edge["target_area"],
                "kind": "mixed" if len(edge["_kinds"]) > 1 else next(iter(edge["_kinds"])),
                "levels": sorted(int(level) for level in edge["_levels"]),
                "imported_names": sorted(str(name) for name in edge["_imported_names"]),
            }
        )

    nodes.sort(key=lambda item: str(item["module"]))
    return {"schema_version": SOUL_SCHEMA_VERSION, "nodes": nodes, "edges": edges}


def _resolve_import_targets(
    entry: PythonImportEntry,
    source_file: SourceFile,
    module_map: Dict[str, SourceFile],
) -> Tuple[str, ...]:
    if not source_file.python_module:
        return ()
    package_parts = source_file.python_module.split(".")
    if not _is_package_file(source_file.rel_path):
        package_parts = package_parts[:-1]
    if entry.level > 0:
        trim = entry.level - 1
        if trim > len(package_parts):
            base_parts: List[str] = []
        else:
            base_parts = package_parts[: len(package_parts) - trim]
    else:
        base_parts = []
    module_parts = [part for part in entry.module.split(".") if part]
    prefix_parts = module_parts if entry.level == 0 else base_parts + module_parts

    candidates: List[str] = []
    if prefix_parts:
        candidates.append(".".join(prefix_parts))
    for imported_name in entry.imported_names:
        if prefix_parts:
            candidates.append(".".join(prefix_parts + [imported_name]))
        elif entry.level > 0:
            candidates.append(".".join(base_parts + [imported_name]))
        elif entry.is_from:
            candidates.append(imported_name)

    resolved = sorted({candidate for candidate in candidates if candidate in module_map})
    return tuple(resolved)


def _build_entrypoints(files: Sequence[SourceFile]) -> List[Dict[str, Any]]:
    entrypoints: List[Dict[str, Any]] = []
    for item in files:
        if item.language in {"shell", "powershell"} and item.rel_path.startswith("scripts/"):
            entrypoints.append(
                {
                    "path": item.rel_path,
                    "module": "",
                    "role": "launcher_script",
                    "language": item.language,
                    "has_main_guard": False,
                    "has_main_function": False,
                    "uses_argparse": False,
                }
            )
            continue
        analysis = item.python_analysis
        if analysis is None:
            continue
        role = _entrypoint_role(item, analysis)
        if not role:
            continue
        entrypoints.append(
            {
                "path": item.rel_path,
                "module": item.python_module,
                "role": role,
                "language": item.language,
                "has_main_guard": analysis.has_main_guard,
                "has_main_function": analysis.has_main_function,
                "uses_argparse": analysis.uses_argparse,
                "module_docstring": _first_line(analysis.module_docstring),
            }
        )
    entrypoints.sort(key=lambda item: (str(item["role"]), str(item["path"])))
    return entrypoints


def _entrypoint_role(item: SourceFile, analysis: PythonAnalysis) -> str:
    if item.rel_path.startswith("src/") and item.rel_path.endswith("/main.py"):
        return "service_main"
    if item.rel_path.startswith("scripts/") and analysis.has_main_guard:
        return "cli_script"
    if analysis.has_main_guard:
        return "python_main"
    return ""


def _build_views(
    files: Sequence[SourceFile],
    module_graph: Dict[str, Any],
    entrypoints: Sequence[Dict[str, Any]],
    area_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    edges = module_graph.get("edges", [])
    inbound_counts: Dict[str, int] = {}
    outbound_counts: Dict[str, int] = {}
    for edge in edges:
        source_module = str(edge.get("source_module", ""))
        target_module = str(edge.get("target_module", ""))
        outbound_counts[source_module] = outbound_counts.get(source_module, 0) + 1
        inbound_counts[target_module] = inbound_counts.get(target_module, 0) + 1

    largest_files = [
        {"path": item.rel_path, "size_bytes": item.size_bytes, "language": item.language}
        for item in sorted(files, key=lambda item: (-item.size_bytes, item.rel_path))[:10]
    ]

    widest_symbol_surfaces = [
        {
            "path": item.rel_path,
            "module": item.python_module,
            "symbol_count": len(item.python_analysis.symbols),
        }
        for item in sorted(
            (item for item in files if item.python_analysis is not None),
            key=lambda item: (-len(item.python_analysis.symbols), item.rel_path),
        )[:10]
    ]

    most_connected_modules = [
        {
            "module": str(node.get("module", "")),
            "path": str(node.get("path", "")),
            "outbound_dependencies": outbound_counts.get(str(node.get("module", "")), 0),
            "inbound_dependencies": inbound_counts.get(str(node.get("module", "")), 0),
            "total_dependencies": outbound_counts.get(str(node.get("module", "")), 0)
            + inbound_counts.get(str(node.get("module", "")), 0),
        }
        for node in sorted(
            module_graph.get("nodes", []),
            key=lambda node: (
                -(
                    outbound_counts.get(str(node.get("module", "")), 0)
                    + inbound_counts.get(str(node.get("module", "")), 0)
                ),
                str(node.get("module", "")),
            ),
        )[:10]
    ]

    runtime = {
        "service_entrypoints": [entry for entry in entrypoints if entry.get("role") == "service_main"],
        "cli_entrypoints": [entry for entry in entrypoints if entry.get("role") == "cli_script"],
        "launcher_scripts": [entry for entry in entrypoints if entry.get("role") == "launcher_script"],
        "config_files": sorted(item.rel_path for item in files if item.rel_path.startswith("config/")),
    }

    return {
        "schema_version": SOUL_SCHEMA_VERSION,
        "runtime": runtime,
        "areas": list(area_summaries),
        "hotspots": {
            "largest_files": largest_files,
            "most_connected_modules": most_connected_modules,
            "widest_symbol_surfaces": widest_symbol_surfaces,
        },
    }


def _analysis_area(rel_path: str) -> str:
    parts = PurePosixPath(rel_path).parts
    if not parts:
        return rel_path
    if parts[0] == "src" and len(parts) >= 3:
        return f"src/{parts[1]}"
    if parts[0] == "src":
        return "src"
    if parts[0] in {"scripts", "config", "docs"}:
        return parts[0]
    return parts[0]


def _module_package(module_name: str, is_package: bool) -> str:
    if not module_name:
        return ""
    if is_package or "." not in module_name:
        return module_name
    return module_name.rsplit(".", 1)[0]


def _is_package_file(rel_path: str) -> bool:
    return PurePosixPath(rel_path).name == "__init__.py"


def _key_file_sort_key(item: SourceFile) -> Tuple[int, int, str]:
    name = PurePosixPath(item.rel_path).name
    if name == "main.py":
        rank = 0
    elif name == "__init__.py":
        rank = 1
    elif item.rel_path.startswith("scripts/"):
        rank = 2
    else:
        rank = 3
    return (rank, -item.size_bytes, item.rel_path)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _build_area_summaries(
    files: Sequence[SourceFile],
    module_graph: Dict[str, Any],
    entrypoints: Sequence[Dict[str, Any]],
    symbol_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    files_by_area: Dict[str, List[SourceFile]] = {}
    nodes_by_area: Dict[str, List[Dict[str, Any]]] = {}
    symbols_by_area: Dict[str, List[Dict[str, Any]]] = {}
    entrypoints_by_area: Dict[str, List[Dict[str, Any]]] = {}
    depends_on: Dict[str, set[str]] = {}
    depended_on_by: Dict[str, set[str]] = {}
    outbound_edge_counts: Dict[str, int] = {}
    inbound_edge_counts: Dict[str, int] = {}

    for item in files:
        area = _analysis_area(item.rel_path)
        files_by_area.setdefault(area, []).append(item)

    for node in module_graph.get("nodes", []):
        area = str(node.get("area", ""))
        nodes_by_area.setdefault(area, []).append(node)

    for symbol in symbol_rows:
        area = _analysis_area(str(symbol.get("path", "")))
        symbols_by_area.setdefault(area, []).append(symbol)

    for entry in entrypoints:
        area = _analysis_area(str(entry.get("path", "")))
        entrypoints_by_area.setdefault(area, []).append(entry)

    for edge in module_graph.get("edges", []):
        source_area = str(edge.get("source_area", ""))
        target_area = str(edge.get("target_area", ""))
        if not source_area or not target_area or source_area == target_area:
            continue
        depends_on.setdefault(source_area, set()).add(target_area)
        depended_on_by.setdefault(target_area, set()).add(source_area)
        outbound_edge_counts[source_area] = outbound_edge_counts.get(source_area, 0) + 1
        inbound_edge_counts[target_area] = inbound_edge_counts.get(target_area, 0) + 1

    area_names = sorted(
        set(files_by_area)
        | set(nodes_by_area)
        | set(symbols_by_area)
        | set(entrypoints_by_area)
        | set(depends_on)
        | set(depended_on_by)
    )

    module_degree: Dict[str, int] = {}
    for edge in module_graph.get("edges", []):
        source_module = str(edge.get("source_module", ""))
        target_module = str(edge.get("target_module", ""))
        module_degree[source_module] = module_degree.get(source_module, 0) + 1
        module_degree[target_module] = module_degree.get(target_module, 0) + 1

    areas: List[Dict[str, Any]] = []
    for area in area_names:
        area_files = sorted(files_by_area.get(area, []), key=_key_file_sort_key)
        area_nodes = list(nodes_by_area.get(area, []))
        area_symbols = list(symbols_by_area.get(area, []))
        area_entrypoints = sorted(
            entrypoints_by_area.get(area, []),
            key=lambda item: (str(item.get("role", "")), str(item.get("path", ""))),
        )
        responsibilities = _area_responsibilities(area_files)
        dominant_terms = _dominant_area_terms(area_files, area_nodes, area_symbols, responsibilities)
        kind = _classify_area(area, area_entrypoints)
        area_depends_on = sorted(depends_on.get(area, set()))
        area_depended_on_by = sorted(depended_on_by.get(area, set()))
        summary = _compose_area_summary(
            area=area,
            kind=kind,
            file_count=len(area_files),
            python_file_count=sum(1 for item in area_files if item.python_analysis is not None),
            responsibilities=responsibilities,
            entrypoints=area_entrypoints,
            depends_on=area_depends_on,
            depended_on_by=area_depended_on_by,
            dominant_terms=dominant_terms,
        )
        key_modules = [
            str(node.get("module", ""))
            for node in sorted(
                area_nodes,
                key=lambda node: (
                    -module_degree.get(str(node.get("module", "")), 0),
                    -int(node.get("symbol_count", 0) or 0),
                    str(node.get("module", "")),
                ),
            )[:5]
            if str(node.get("module", ""))
        ]
        areas.append(
            {
                "area": area,
                "kind": kind,
                "summary": summary,
                "file_count": len(area_files),
                "python_file_count": sum(1 for item in area_files if item.python_analysis is not None),
                "total_bytes": sum(item.size_bytes for item in area_files),
                "key_files": [item.rel_path for item in area_files[:5]],
                "key_modules": key_modules,
                "responsibilities": responsibilities,
                "dominant_terms": dominant_terms,
                "entrypoints": [
                    {
                        "path": str(entry.get("path", "")),
                        "module": str(entry.get("module", "")),
                        "role": str(entry.get("role", "")),
                    }
                    for entry in area_entrypoints
                ],
                "depends_on": area_depends_on,
                "depended_on_by": area_depended_on_by,
                "outbound_dependency_count": outbound_edge_counts.get(area, 0),
                "inbound_dependency_count": inbound_edge_counts.get(area, 0),
            }
        )

    return areas


def _area_responsibilities(files: Sequence[SourceFile]) -> List[str]:
    phrases: List[str] = []
    for item in files:
        phrase = _file_component_phrase(item.rel_path)
        if not phrase or phrase in phrases:
            continue
        phrases.append(phrase)
        if len(phrases) >= 6:
            break
    return phrases


def _file_component_phrase(rel_path: str) -> str:
    path = PurePosixPath(rel_path)
    stem = path.stem
    if stem == "__init__":
        return ""
    if stem == "main":
        if len(path.parts) >= 3 and path.parts[0] == "src":
            return f"{_humanize_identifier(path.parts[1])} main"
        return ""
    return _humanize_identifier(stem)


def _dominant_area_terms(
    files: Sequence[SourceFile],
    nodes: Sequence[Dict[str, Any]],
    symbols: Sequence[Dict[str, Any]],
    responsibilities: Sequence[str],
) -> List[str]:
    counts: Counter[str] = Counter()
    for phrase in responsibilities:
        for token in _summary_tokens(phrase):
            counts[token] += 5
    for item in files:
        for token in _summary_tokens(PurePosixPath(item.rel_path).stem):
            counts[token] += 3
    for node in nodes:
        for token in _summary_tokens(str(node.get("module", "")).rsplit(".", 1)[-1]):
            counts[token] += 2
        for token in _summary_tokens(str(node.get("module_docstring", ""))):
            counts[token] += 1
    for symbol in symbols:
        for token in _summary_tokens(str(symbol.get("name", ""))):
            counts[token] += 1
    return [term for term, _ in counts.most_common(6)]


def _summary_tokens(text: str) -> List[str]:
    if not text:
        return []
    normalized = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text.replace("/", " ").replace("_", " ").replace(".", " ").replace("-", " "),
    )
    tokens = [token.lower() for token in re.split(r"[^A-Za-z0-9]+", normalized) if token]
    return [token for token in tokens if token not in SUMMARY_STOPWORDS and len(token) > 1 and not token.isdigit()]


def _humanize_identifier(text: str) -> str:
    tokens = _summary_tokens(text)
    if not tokens:
        return ""
    return " ".join(tokens)


def _classify_area(area: str, entrypoints: Sequence[Dict[str, Any]]) -> str:
    roles = {str(entry.get("role", "")) for entry in entrypoints}
    if area.startswith("src/"):
        return "runtime subsystem" if "service_main" in roles else "runtime code area"
    if area == "src":
        return "runtime code area"
    if area == "scripts":
        return "operational tooling area"
    if area == "config":
        return "configuration surface"
    if area == "docs":
        return "documentation area"
    return "project area"


def _compose_area_summary(
    area: str,
    kind: str,
    file_count: int,
    python_file_count: int,
    responsibilities: Sequence[str],
    entrypoints: Sequence[Dict[str, Any]],
    depends_on: Sequence[str],
    depended_on_by: Sequence[str],
    dominant_terms: Sequence[str],
) -> str:
    article = _indefinite_article(kind)
    if python_file_count and python_file_count != file_count:
        opening = (
            f"{area} is {article} {kind} with {_counted_phrase(file_count, 'file')}, "
            f"including {_counted_phrase(python_file_count, 'Python module')}."
        )
    elif python_file_count:
        opening = f"{area} is {article} {kind} with {_counted_phrase(python_file_count, 'Python module')}."
    else:
        opening = f"{area} is {article} {kind} with {_counted_phrase(file_count, 'file')}."

    sentences = [opening]
    if responsibilities:
        sentences.append(f"It centers on {_join_phrases(responsibilities[:4])}.")
    elif dominant_terms:
        sentences.append(f"Its dominant terms are {_join_phrases(dominant_terms[:4])}.")

    if entrypoints:
        role_labels: List[str] = []
        role_names = {
            "service_main": "service entrypoint",
            "cli_script": "CLI script",
            "launcher_script": "launcher script",
            "python_main": "Python main module",
        }
        for role in ("service_main", "cli_script", "launcher_script", "python_main"):
            count = sum(1 for entry in entrypoints if str(entry.get("role", "")) == role)
            if not count:
                continue
            role_labels.append(_counted_phrase(count, role_names[role]))
        if role_labels:
            sentences.append(f"It exposes {_join_phrases(role_labels)}.")

    if depends_on:
        sentences.append(f"It depends on {_join_phrases(depends_on[:3])}.")
    if depended_on_by:
        sentences.append(f"It is used by {_join_phrases(depended_on_by[:3])}.")
    return " ".join(sentences)


def _join_phrases(items: Sequence[str]) -> str:
    filtered = [str(item) for item in items if str(item)]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 2:
        return f"{filtered[0]} and {filtered[1]}"
    return ", ".join(filtered[:-1]) + f", and {filtered[-1]}"


def _counted_phrase(count: int, noun: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {noun}{suffix}"


def _indefinite_article(text: str) -> str:
    stripped = text.strip().lower()
    if not stripped:
        return "a"
    return "an" if stripped[0] in {"a", "e", "i", "o", "u"} else "a"


def _build_manifest(
    repo_root: Path,
    config: SoulConfig,
    files: Sequence[SourceFile],
    skipped: Sequence[SkippedFile],
    missing_includes: Sequence[str],
    source_digest: str,
    config_digest: str,
    symbol_rows: Sequence[Dict[str, Any]],
    module_graph: Dict[str, Any],
    entrypoints: Sequence[Dict[str, Any]],
    area_summaries: Sequence[Dict[str, Any]],
    views: Dict[str, Any],
) -> Dict[str, Any]:
    python_file_count = sum(1 for item in files if item.python_analysis is not None)
    total_bytes = sum(item.size_bytes for item in files)
    return {
        "schema_version": SOUL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": str(repo_root),
        "output_dir": config.output_dir,
        "source_digest": source_digest,
        "config_digest": config_digest,
        "file_count": len(files),
        "python_file_count": python_file_count,
        "symbol_count": len(symbol_rows),
        "internal_module_count": len(module_graph.get("nodes", [])),
        "internal_dependency_edge_count": len(module_graph.get("edges", [])),
        "entrypoint_count": len(entrypoints),
        "area_count": len(area_summaries),
        "skipped_count": len(skipped),
        "total_bytes": total_bytes,
        "include": list(config.include),
        "exclude": list(config.exclude),
        "max_file_bytes": config.max_file_bytes,
        "missing_includes": list(missing_includes),
        "artifacts": {
            "files": "meta/files.jsonl",
            "python_index": "meta/python_index.jsonl",
            "symbols": "meta/symbols.jsonl",
            "module_graph": "meta/module_graph.json",
            "entrypoints": "meta/entrypoints.jsonl",
            "area_summaries": "meta/area_summaries.json",
            "views": "meta/views.json",
            "overview": "meta/overview.md",
            "skipped": "meta/skipped.jsonl",
            "tree": "meta/tree.txt",
            "mirror_root": "code",
        },
    }


def _render_overview(manifest: Dict[str, Any], views: Dict[str, Any]) -> str:
    runtime = views.get("runtime", {})
    hotspots = views.get("hotspots", {})
    areas = views.get("areas", [])

    lines = [
        "# Soul Overview",
        "",
        "## Summary",
        f"- Files: {manifest.get('file_count', 0)}",
        f"- Python files: {manifest.get('python_file_count', 0)}",
        f"- Symbols: {manifest.get('symbol_count', 0)}",
        f"- Internal modules: {manifest.get('internal_module_count', 0)}",
        f"- Internal dependency edges: {manifest.get('internal_dependency_edge_count', 0)}",
        f"- Entrypoints: {manifest.get('entrypoint_count', 0)}",
        "",
        "## Runtime",
    ]

    service_entrypoints = runtime.get("service_entrypoints", [])
    cli_entrypoints = runtime.get("cli_entrypoints", [])
    launcher_scripts = runtime.get("launcher_scripts", [])

    if service_entrypoints:
        lines.append("- Service entrypoints:")
        for entry in service_entrypoints:
            lines.append(f"  - {entry.get('path')} ({entry.get('module')})")
    if cli_entrypoints:
        lines.append("- CLI entrypoints:")
        for entry in cli_entrypoints:
            lines.append(f"  - {entry.get('path')} ({entry.get('module')})")
    if launcher_scripts:
        lines.append("- Launcher scripts:")
        for entry in launcher_scripts:
            lines.append(f"  - {entry.get('path')}")
    if not service_entrypoints and not cli_entrypoints and not launcher_scripts:
        lines.append("- No entrypoints detected.")

    lines.extend(["", "## Area Summaries"])
    for area in areas[:10]:
        lines.append(f"- {area.get('area')}: {area.get('summary')}")

    lines.extend(["", "## Hotspots", "- Largest files:"])
    for item in hotspots.get("largest_files", [])[:5]:
        lines.append(f"  - {item.get('path')} ({item.get('size_bytes')} bytes)")

    lines.append("- Most connected modules:")
    for item in hotspots.get("most_connected_modules", [])[:5]:
        lines.append(
            f"  - {item.get('module')}: outbound={item.get('outbound_dependencies')} "
            f"inbound={item.get('inbound_dependencies')}"
        )

    lines.append("- Widest symbol surfaces:")
    for item in hotspots.get("widest_symbol_surfaces", [])[:5]:
        lines.append(f"  - {item.get('path')} ({item.get('symbol_count')} symbols)")

    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
