"""Dashboard/server contract compatibility checks without robot hardware."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import inference_dashboard as dashboard


class DashboardContractTests(unittest.TestCase):
    def test_identical_contract_is_accepted_without_mutation(self):
        contract = copy.deepcopy(dashboard.CONTRACT)
        before = copy.deepcopy(contract)
        dashboard.validate_server_contract(contract)
        self.assertEqual(contract, before)

    def test_training_provenance_fields_may_differ_or_be_absent(self):
        for key in ('dataset_manifest_sha256', 'dataset_mix'):
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    contract = copy.deepcopy(dashboard.CONTRACT)
                    if missing:
                        contract.pop(key, None)
                    else:
                        contract[key] = 'different training provenance'
                    dashboard.validate_server_contract(contract)
        contract = copy.deepcopy(dashboard.CONTRACT)
        contract.pop('dataset_manifest_sha256', None)
        contract.pop('dataset_mix', None)
        dashboard.validate_server_contract(contract)

    def test_every_other_field_must_match_and_be_present(self):
        ignored = {'dataset_manifest_sha256', 'dataset_mix'}
        for key in dashboard.CONTRACT.keys() - ignored:
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    contract = copy.deepcopy(dashboard.CONTRACT)
                    if missing:
                        del contract[key]
                    else:
                        contract[key] = {'unexpected': 'value'}
                    with self.assertRaisesRegex(RuntimeError, key):
                        dashboard.validate_server_contract(contract)

    def test_nested_image_and_safety_changes_are_rejected(self):
        for key in ('image_crops_xyxy', 'torque_limits_nm'):
            with self.subTest(key=key):
                contract = copy.deepcopy(dashboard.CONTRACT)
                if key == 'image_crops_xyxy':
                    camera = next(iter(contract[key]))
                    contract[key][camera][0] += 1
                else:
                    contract[key][0] += 1
                with self.assertRaisesRegex(RuntimeError, key):
                    dashboard.validate_server_contract(contract)

    def test_unknown_field_is_rejected(self):
        contract = copy.deepcopy(dashboard.CONTRACT)
        contract['unknown_contract_field'] = True
        with self.assertRaisesRegex(RuntimeError, 'unknown_contract_field'):
            dashboard.validate_server_contract(contract)

    def test_non_dictionary_contract_is_rejected(self):
        for contract in (None, [], '', 20, True):
            with self.subTest(contract=contract):
                with self.assertRaises(RuntimeError):
                    dashboard.validate_server_contract(contract)


if __name__ == '__main__':
    unittest.main()
