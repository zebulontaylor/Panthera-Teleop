"""Publication guards prevent accidental overwrite and unverified remote releases."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('publish_retimed_release',Path(__file__).parents[1]/'scripts/publish_retimed_release.py')
pub=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pub)


class PublicationTests(unittest.TestCase):
    def test_inventory_is_bounded_to_portable_release_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'meta').mkdir();(root/'meta/info.json').write_text('{}')
            (root/'credentials.json').write_text('private')
            (root/'meta/.cache').mkdir();(root/'meta/.cache/private').write_text('private')
            self.assertEqual(set(pub.inventory(root)),{'meta/info.json'})
            (root/'meta/link').symlink_to(root/'credentials.json')
            with self.assertRaises(ValueError):pub.inventory(root)

    def test_existing_destination_is_never_overwritten(self):
        class API:
            def whoami(self):return {'name':'owner'}
            def repo_exists(self,*args,**kwargs):return True
            def create_repo(self,*args,**kwargs):raise AssertionError('Existing repository must not be changed')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pub.write(root/'verification.json',{'passed':True})
            pub.write(root/'processing/manifest.json',{'repo_id':'owner/new','totals':{'episodes':1}})
            (root/'README.md').write_text('card')
            args=argparse.Namespace(root=root,repo_id='owner/new',state=root/'state.json')
            with patch.object(pub,'credential',return_value='fake'),patch.object(pub,'HfApi',return_value=API()):
                with self.assertRaises(FileExistsError):pub.publish(args)

    def test_remote_lfs_mismatch_fails_before_publication(self):
        class Sibling:
            rfilename='data/example.parquet';lfs={'sha256':'bad'};size=3
        class Info:siblings=[Sibling()]
        class API:
            def dataset_info(self,*args,**kwargs):return Info()
        with self.assertRaises(AssertionError):
            pub.verify_remote(API(),'owner/new','revision',{'data/example.parquet':{'sha256':'good','bytes':3}},'fake')


if __name__=='__main__':unittest.main()
