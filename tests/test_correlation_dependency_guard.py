import tempfile
import unittest
from pathlib import Path
from tools.correlation_dependency_guard import local_dependencies, scoped_run


class DependencyGuardTests(unittest.TestCase):
    def test_transitive_lazy_and_relative_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'pkg').mkdir()
            (root/'pkg/__init__.py').write_text('')
            (root/'main.py').write_text('from pkg import helper\n')
            (root/'pkg/helper.py').write_text('def f():\n from . import leaf\n')
            (root/'pkg/leaf.py').write_text('')
            (root/'unrelated.py').write_text('')
            found = local_dependencies(root,[root/'main.py'])
            self.assertEqual({str(p.relative_to(root)) for p in found},
                             {'main.py','pkg/__init__.py','pkg/helper.py','pkg/leaf.py'})
            snapshot = {str(p):'hash' for p in root.rglob('*.py')}
            run = {'source_snapshot':snapshot,'checkpoints':{'keep':'unchanged'}}
            scoped = scoped_run(run,root,[root/'main.py'])
            self.assertNotIn(str(root/'unrelated.py'),scoped['source_snapshot'])
            self.assertEqual(scoped['checkpoints'],run['checkpoints'])
            self.assertIn(str(root/'unrelated.py'),run['source_snapshot'])
            del run['source_snapshot'][str(root/'pkg/leaf.py')]
            with self.assertRaises(RuntimeError):
                scoped_run(run,root,[root/'main.py'])
