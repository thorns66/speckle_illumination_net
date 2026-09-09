"""Verify the transitive local import graph, not unrelated experiments."""
from __future__ import annotations

import ast
from pathlib import Path


def local_dependencies(root, entrypoints):
    root = Path(root).resolve()
    pending = [Path(p).resolve() for p in entrypoints]
    found = set()
    while pending:
        path = pending.pop()
        if path in found or not path.is_file():
            continue
        path.relative_to(root)
        found.add(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = path.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    base = '.'.join(parent.relative_to(root).parts)
                    module = '.'.join(filter(None, [base, node.module]))
                else:
                    module = node.module or ''
                names = [module] + ['.'.join(filter(None, [module, item.name])) for item in node.names]
            for name in names:
                parts = name.split('.')
                for base in (root, root / 'tools'):
                    target = base.joinpath(*parts)
                    candidates = [target.with_suffix('.py'), target / '__init__.py']
                    candidates.extend(base.joinpath(*parts[:i]) / '__init__.py' for i in range(1, len(parts)))
                    pending.extend(p for p in candidates if p.is_file())
    return found


def scoped_run(run, root, entrypoints):
    dependencies = local_dependencies(root, entrypoints)
    expected = run['source_snapshot']
    missing = [str(p) for p in dependencies if str(p) not in expected]
    if missing:
        raise RuntimeError(f'Unrecorded dependency: {missing}')
    scoped = dict(run)
    scoped['source_snapshot'] = {str(p): expected[str(p)] for p in sorted(dependencies)}
    return scoped
