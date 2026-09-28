"""Regression for license copying from a444/555 immutable source export."""
import ast
from pathlib import Path
import shutil
import stat
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/package_release.py'
# The copied function uses only pathlib/shutil. Compile that exact function so
# this CPU/filesystem regression does not import unrelated GPU/torch packages.
tree = ast.parse(SCRIPT.read_text())
function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'copy_license_assets')
namespace = {'Path':Path, 'shutil':shutil}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SCRIPT), 'exec'), namespace)
copy_license_assets = namespace['copy_license_assets']


def source_state(root):
    return {str(path.relative_to(root)):{'mode':stat.S_IMODE(path.stat().st_mode),
            'mtime_ns':path.stat().st_mtime_ns,
            'bytes':path.read_bytes() if path.is_file() else None}
            for path in (root, *sorted(root.rglob('*')))}


class ReadonlyLicenseRegression(unittest.TestCase):
    def test_readonly_export_preserved_and_destination_project_license_written(self):
        for existing_placeholder in (False, True):
            with self.subTest(existing_project_license=existing_placeholder):
                self.check_export_case(existing_placeholder)

    def check_export_case(self, existing_placeholder):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, destination = base/'immutable-export', base/'release-licenses'
            (source/'licenses/nested').mkdir(parents=True)
            payloads = {'LICENSE':b'actual project license\n',
                        'licenses/NOTICE':b'upstream notice\x00\xff',
                        'licenses/nested/LICENSE.txt':b'nested attribution\n'}
            if existing_placeholder:
                payloads['licenses/PROJECT-LICENSE'] = b'old placeholder must be replaced\n'
            for name, data in payloads.items():
                (source/name).write_bytes(data)
                (source/name).chmod(0o444)
            directories = [source, source/'licenses', source/'licenses/nested']
            for path in directories:path.chmod(0o555)
            before = source_state(source)
            try:
                copy_license_assets(source, destination)
                self.assertEqual(source_state(source), before)
                self.assertEqual((destination/'PROJECT-LICENSE').read_bytes(), payloads['LICENSE'])
                self.assertEqual((destination/'NOTICE').read_bytes(), payloads['licenses/NOTICE'])
                self.assertEqual((destination/'nested/LICENSE.txt').read_bytes(), payloads['licenses/nested/LICENSE.txt'])
                self.assertEqual({str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()},
                                 {'PROJECT-LICENSE','NOTICE','nested/LICENSE.txt'})
                for path in (destination, *destination.rglob('*')):
                    self.assertTrue(path.stat().st_mode & stat.S_IWUSR, str(path))
                with self.assertRaises(FileExistsError):copy_license_assets(source, destination)
                self.assertEqual(source_state(source), before)
            finally:
                # Only the disposable test fixture is made removable, after the
                # source immutability assertions; production code never chmods.
                for path in directories:path.chmod(0o755)


if __name__ == '__main__':unittest.main()
