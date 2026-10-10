"""Pre-observation boundary and fake-transport tests; no network calls."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import probe_v2 as probe


def tick_record(kind="CALL", **updates):
    result = {"underlying_symbol": "BOOM500", "contract_type": kind,
              "barriers": 0, "expiry_type": "tick", "min_contract_duration": "5t",
              "max_contract_duration": "10t"}
    result.update(updates)
    return result


def mult_record(kind="MULTUP", **updates):
    result = {"underlying_symbol": "BOOM500", "contract_type": kind,
              "multiplier_range": [50, 10, 20]}
    result.update(updates)
    return result


class Boundaries(unittest.TestCase):
    def test_http_redirect_is_not_followed(self):
        original = RuntimeError("synthetic redirect rejection")
        self.assertIs(probe.PublicConnect.process_redirect(None, original), original)

    def test_exact_catalogue_matrix(self):
        for symbol in probe.SYMBOLS:
            probe.validate_request(probe.catalog_request(symbol))

    def test_no_account_order_or_legacy_fields(self):
        base = probe.catalog_request("BOOM500")
        for field in ("authorize", "token", "buy", "sell", "currency", "loginid", "product_type"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.validate_request({**base, field: 1})

    def test_boolean_catalogue_id_rejected(self):
        with self.assertRaises(ValueError):
            probe.validate_request({"contracts_for": "BOOM500", "req_id": True})

    def test_tick_interval_is_explicit(self):
        request, _ = probe.eligible_quote("BOOM500", "CALL", [tick_record()])
        self.assertEqual(request["duration_unit"], "t")
        self.assertEqual(request["duration"], 5)
        probe.validate_request(request)

    def test_bare_or_other_unit_duration_unknown(self):
        for value in ("5", "5s", "1m", 5, None, "5T", " 5t", "5.0t"):
            with self.subTest(value=value):
                request, _ = probe.eligible_quote("BOOM500", "CALL", [tick_record(min_contract_duration=value)])
                self.assertIsNone(request)

    def test_duration_not_inside_interval(self):
        for updates in ({"min_contract_duration": "6t"}, {"max_contract_duration": "4t"}):
            self.assertIsNone(probe.eligible_quote("BOOM500", "CALL", [tick_record(**updates)])[0])

    def test_barrier_or_non_tick_record_rejected(self):
        for updates in ({"barriers": 1}, {"barriers": False}, {"expiry_type": "intraday"}):
            self.assertIsNone(probe.eligible_quote("BOOM500", "CALL", [tick_record(**updates)])[0])

    def test_wrong_symbol_or_product_rejected(self):
        for updates in ({"underlying_symbol": "CRASH500"}, {"contract_type": "PUT"}):
            self.assertIsNone(probe.eligible_quote("BOOM500", "CALL", [tick_record(**updates)])[0])

    def test_copy_smallest_advertised_multiplier(self):
        request, _ = probe.eligible_quote("BOOM500", "MULTUP", [mult_record()])
        self.assertEqual(request["multiplier"], 10)
        self.assertNotIn("duration", request)
        self.assertNotIn("subscribe", request)
        probe.validate_request(request)

    def test_invalid_multiplier_choices_unknown(self):
        for choices in ([], None, "10", [True], ["10"], [0], [-1], [float("inf")], [float("nan")], [{}], [10**400]):
            with self.subTest(choices=choices):
                self.assertIsNone(probe.eligible_quote("BOOM500", "MULTUP", [mult_record(multiplier_range=choices)])[0])

    def test_conflicting_multiplier_records_unknown(self):
        self.assertIsNone(probe.eligible_quote("BOOM500", "MULTUP",
            [mult_record(), mult_record(multiplier_range=[20, 50])])[0])

    def test_equivalent_multiplier_records_valid(self):
        self.assertIsNotNone(probe.eligible_quote("BOOM500", "MULTUP",
            [mult_record(), mult_record(multiplier_range=[20, 10, 50, 10])])[0])

    def test_malformed_second_multiplier_record_unknown(self):
        self.assertIsNone(probe.eligible_quote("BOOM500", "MULTUP",
            [mult_record(), mult_record(multiplier_range=[])])[0])

    def test_extra_proposal_fields_rejected(self):
        request, _ = probe.eligible_quote("BOOM500", "CALL", [tick_record()])
        for key, value in (("buy", "id"), ("subscribe", 0), ("subscribe", 1),
                           ("symbol", "BOOM500"), ("limit_order", {}), ("proposal", True)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                probe.validate_request({**request, key: value})

    def test_minimal_proposal_and_string_prices_preserved(self):
        request, _ = probe.eligible_quote("BOOM500", "CALL", [tick_record()])
        for obj in ({"id": "public-id"}, {"id": "public-id", "ask_price": "10.0", "payout": None}):
            row = probe.summarize(request, {"echo_req": request, "req_id": request["req_id"],
                                  "msg_type": "proposal", "proposal": obj})
            self.assertEqual(row["status"], "SUCCESS")
            self.assertEqual(row["object_keys"], sorted(obj))

    def test_missing_or_malformed_proposal_id_unusable(self):
        request, _ = probe.eligible_quote("BOOM500", "CALL", [tick_record()])
        for obj in ({}, {"id": None}, {"id": ""}, {"id": 12}, {"id": True}):
            with self.subTest(obj=obj):
                row = probe.summarize(request, {"echo_req": request, "req_id": request["req_id"],
                                     "msg_type": "proposal", "proposal": obj})
                self.assertEqual(row["status"], "UNUSABLE_PROPOSAL")

    def test_documented_multiplier_echo_default(self):
        request, _ = probe.eligible_quote("BOOM500", "MULTUP", [mult_record()])
        base = {"req_id": request["req_id"], "msg_type": "proposal", "proposal": {"id": "fixture"}}
        for echo in (request, {**request, "duration_unit": "s"}):
            self.assertEqual(probe.summarize(request, {**base, "echo_req": echo})["status"], "SUCCESS")
        for updates in ({"duration_unit": "t"}, {"duration_unit": None}, {"subscribe": 1},
                        {"duration_unit": "s", "amount": True}, {"duration_unit": "s", "buy": "fixture"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                probe.summarize(request, {**base, "echo_req": {**request, **updates}})

    def test_response_errors_remain_observations(self):
        request = probe.catalog_request("BOOM500")
        row = probe.summarize(request, {"echo_req": request, "req_id": 1,
                    "msg_type": "contracts_for", "error": {"code": "NoMarket", "message": "x"}})
        self.assertEqual(row["status"], "API_ERROR")
        self.assertNotIn("available_records", row)

    def test_response_mismatch_rejected(self):
        request = probe.catalog_request("BOOM500")
        base = {"echo_req": request, "req_id": 1, "msg_type": "contracts_for", "contracts_for": {"available": []}}
        for updates in ({"req_id": True}, {"req_id": 2}, {"msg_type": "proposal"}, {"echo_req": {}},
                        {"echo_req": {"contracts_for": "BOOM500", "req_id": True}}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                probe.summarize(request, {**base, **updates})

    def test_duplicate_keys_nonfinite_binary_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '[]', b"{}"): 
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                probe.decode(raw)

    def test_safety_guard_all_flags(self):
        for key in probe.SAFETY:
            with patch.dict(os.environ, {key: "true"}), self.assertRaises(ValueError):
                probe.safety_guard()

    def test_fake_transport_all_catalogue_errors_no_proposals(self):
        sent = []
        class Fake:
            async def send(self, text):
                sent.append(json.loads(text))
            async def recv(self):
                request = sent[-1]
                return json.dumps({"echo_req": request, "req_id": request["req_id"],
                    "msg_type": "contracts_for", "error": {"code": "Unavailable", "message": "fixture"}})
            async def close(self):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture"
            connections = []
            async def connect(endpoint, **kwargs):
                self.assertTrue((output / "declaration.json").is_file())
                self.assertEqual(endpoint, probe.ENDPOINT)
                connections.append(endpoint)
                return Fake()
            result = asyncio.run(probe.probe(output, connector=connect))
            self.assertEqual(len(connections), 1)
            self.assertEqual(len(sent), 5)
            self.assertEqual(result["transport_status"], "COMPLETED")
            self.assertEqual(result["response_observations"], 5)
            self.assertTrue(result["connection_closed"])
            self.assertEqual(result["orders_sent"], 0)
            with self.assertRaises(FileExistsError):
                asyncio.run(probe.probe(output, connector=connect))


if __name__ == "__main__":
    unittest.main()
