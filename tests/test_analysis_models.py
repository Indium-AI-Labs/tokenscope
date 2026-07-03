from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

from analysis_models import (
    ChatMessage,
    PricingProfile,
    ProjectState,
    RegressionCase,
    add_recent_tokenizer,
    analyze_rag_chunks,
    analyze_batch_prompts,
    analyze_chat_budget,
    analyze_corpus_path,
    build_export_payload,
    calculate_prompt_budget,
    compare_corpus_path,
    decode_round_trip,
    diff_tokenizers,
    distribution_from_counts,
    estimate_token_cost,
    extract_special_tokens,
    first_difference_offset,
    format_export,
    inspect_token,
    inspect_unicode,
    load_project_state,
    load_pricing_profiles,
    load_recent_tokenizers,
    pipeline_debug,
    run_regression_suite,
    save_project_state,
    save_pricing_profiles,
    search_tokens,
    simulate_packing,
    suggest_tokenizer_repairs,
    tokenizer_metadata,
)
from tokenizer_engine import TokenizationResult, TokenizerEngine


def make_word_engine(root: Path, *, special: bool = False, chat_template: bool = False) -> TokenizerEngine:
    root.mkdir(parents=True, exist_ok=True)
    vocab = {
        "hello": 0,
        "world": 1,
        "alpha": 2,
        "beta": 3,
        "one": 4,
        "two": 5,
        "three": 6,
        "[UNK]": 7,
        "[PAD]": 8,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if special:
        tokenizer.add_special_tokens(
            [
                AddedToken(
                    "[PAD]",
                    special=True,
                    normalized=False,
                    single_word=True,
                    lstrip=True,
                    rstrip=False,
                )
            ]
        )
    tokenizer.save(str(root / "tokenizer.json"))
    if chat_template:
        (root / "tokenizer_config.json").write_text(
            (
                "{"
                "\"bos_token\":\"<s>\","
                "\"chat_template\":\"{{ bos_token }}{% for message in messages %}"
                "{{ message['role'] }}: {{ message['content'] }}\\n"
                "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}\""
                "}"
            ),
            encoding="utf-8",
        )
    return TokenizerEngine.load(root)


class AnalysisModelTests(unittest.TestCase):
    def test_token_inspector_selection_and_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok")
            result = engine.encode("hello hello world")

            inspection = inspect_token("primary", engine, result, 1)

            self.assertIsNotNone(inspection)
            assert inspection is not None
            self.assertEqual(inspection.token_string, "hello")
            self.assertEqual(inspection.frequency_in_input, 2)
            self.assertTrue(inspection.in_vocab)
            self.assertIn("Merge tree", inspection.merge_tree)

    def test_decode_round_trip_exact_and_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok")
            result = engine.encode("hello world")

            exact = decode_round_trip("primary", engine, result)
            mismatch = decode_round_trip(
                "primary",
                engine,
                TokenizationResult(
                    input_text="helloworld",
                    tokens=result.tokens,
                    token_ids=result.token_ids,
                    spans=result.spans,
                    stats=result.stats,
                ),
            )

            self.assertTrue(exact.exact_match if exact else False)
            self.assertFalse(mismatch.exact_match if mismatch else True)
            self.assertEqual(mismatch.first_difference if mismatch else None, 5)
            self.assertIsNone(first_difference_offset("same", "same"))

    def test_special_token_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok", special=True)

            specials = extract_special_tokens(engine)

            pad = next(item for item in specials if item.token == "[PAD]")
            self.assertEqual(pad.token_id, 8)
            self.assertTrue(pad.special)
            self.assertFalse(pad.normalized)
            self.assertTrue(pad.single_word)
            self.assertTrue(pad.lstrip)
            self.assertFalse(pad.rstrip)

    def test_budget_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok")
            result = engine.encode("hello world")

            budget = calculate_prompt_budget("primary", result, 4)

            self.assertEqual(budget.used_tokens if budget else None, 2)
            self.assertEqual(budget.remaining_tokens if budget else None, 2)
            self.assertEqual(budget.percent_used if budget else None, 50.0)

    def test_corpus_file_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            engine = make_word_engine(workspace / "tok")
            corpus = workspace / "corpus"
            corpus.mkdir()
            (corpus / "a.txt").write_text("hello world\nalpha beta", encoding="utf-8")
            (corpus / "b.md").write_text("# hello", encoding="utf-8")
            (corpus / "c.jsonl").write_text('{"text": "one two"}\n{"text": "three"}', encoding="utf-8")
            (corpus / "d.json").write_text('{"items": ["hello", "world"]}', encoding="utf-8")
            (corpus / "e.csv").write_text("one,two\nthree,hello", encoding="utf-8")
            (corpus / "skip.bin").write_bytes(b"\x00\x01")

            result = analyze_corpus_path(corpus, engine)

            self.assertEqual(result.total_files, 5)
            self.assertEqual(result.skipped_files, 1)
            self.assertGreater(result.total_chars, 0)
            self.assertGreater(result.total_tokens, 0)
            self.assertTrue(result.top_token_ids)
            self.assertTrue(result.longest_lines)

    def test_token_search_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok")
            result = engine.encode("hello world hello")

            by_text = search_tokens("primary", result, "hello", "text")
            by_token = search_tokens("primary", result, "world", "token")
            by_id = search_tokens("primary", result, "0", "id")

            self.assertEqual([match.index for match in by_text], [0, 2])
            self.assertEqual([match.index for match in by_token], [1])
            self.assertEqual([match.index for match in by_id], [0, 2])

    def test_csv_and_markdown_export_formatting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok")
            result = engine.encode("hello world")
            payload = build_export_payload(engine, result, budget_limit=8)

            csv_export = format_export(payload, "csv")
            markdown_export = format_export(payload, "md")

            self.assertIn("primary_tokens", csv_export)
            self.assertIn("budget", csv_export)
            self.assertIn("## Token Table", markdown_export)
            self.assertIn("## Budget", markdown_export)

    def test_recent_tokenizer_history_ordering_and_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            paths = [workspace / f"tok{i}" for i in range(12)]

            for path in paths:
                recent = add_recent_tokenizer(path, workspace)
            recent = add_recent_tokenizer(paths[3], workspace)

            self.assertEqual(len(recent), 10)
            self.assertEqual(recent[0], str(paths[3].resolve()))
            self.assertEqual(load_recent_tokenizers(workspace), recent)

    def test_byte_level_symbol_conversion(self) -> None:
        symbols = TokenizerEngine.byte_level_symbols(" hello")

        self.assertEqual(symbols[0], chr(288))
        self.assertEqual(symbols[1:], list("hello"))

    def test_metadata_reports_loaded_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok")

            metadata = tokenizer_metadata(engine)

            self.assertIsNotNone(metadata)
            assert metadata is not None
            self.assertEqual(metadata.model_type, "WordLevel")
            self.assertIn("tokenizer.json", metadata.loaded_tokenizer_file)
            self.assertTrue(metadata.config_files_found)

    def test_chat_template_budget_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok", chat_template=True)

            result = analyze_chat_budget(
                [ChatMessage("user", "hello world")],
                engine,
                add_generation_prompt=True,
                budget_limit=16,
            )

            self.assertIsNotNone(result.primary)
            assert result.primary is not None
            self.assertIn("user: hello world", result.primary.rendered_text)
            self.assertIn("assistant:", result.primary.rendered_text)
            self.assertGreater(result.primary.token_count, 0)
            self.assertIsNone(result.primary.error)

    def test_chat_template_missing_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok")

            result = analyze_chat_budget([ChatMessage("user", "hello")], engine)

            self.assertIsNotNone(result.primary)
            assert result.primary is not None
            self.assertFalse(result.primary.has_template)
            self.assertIn("chat_template", result.primary.error or "")

    def test_pipeline_debug_reports_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok")

            result = pipeline_debug("primary", engine, "hello world")

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.token_count, 2)
            self.assertEqual([piece.text for piece in result.pre_tokens], ["hello", "world"])
            self.assertEqual(result.token_ids, (0, 1))

    def test_batch_prompt_analysis_with_compare_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            primary = make_word_engine(workspace / "primary")
            compare = make_word_engine(workspace / "compare")
            batch = workspace / "batch.jsonl"
            batch.write_text('{"prompt": "hello world"}\n{"prompt": "alpha beta one"}', encoding="utf-8")

            result = analyze_batch_prompts(batch, primary, compare, budget_limit=2)

            self.assertEqual(result.total_prompts, 2)
            self.assertEqual(result.total_files, 1)
            self.assertEqual(result.budget_failures, 1)
            self.assertIsNotNone(result.compare_total_tokens)
            self.assertTrue(result.longest_prompts)

    def test_corpus_compare_reports_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            primary = make_word_engine(workspace / "primary")
            compare = make_word_engine(workspace / "compare")
            corpus = workspace / "corpus"
            corpus.mkdir()
            (corpus / "sample.txt").write_text("hello world\nalpha beta one", encoding="utf-8")

            result = compare_corpus_path(corpus, primary, compare)

            self.assertEqual(result.total_files, 1)
            self.assertGreater(result.primary_total_tokens, 0)
            self.assertEqual(result.compare_total_tokens, result.primary_total_tokens)
            self.assertTrue(result.biggest_savings)
            self.assertTrue(result.biggest_regressions)

    def test_html_export_formatting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok", chat_template=True)
            tokenized = engine.encode("hello <world>")
            chat = analyze_chat_budget([ChatMessage("user", "hello <world>")], engine)
            pipeline = pipeline_debug("primary", engine, "hello <world>")
            payload = build_export_payload(
                engine,
                tokenized,
                chat_budget=chat,
                pipeline_result=pipeline,
            )

            exported = format_export(payload, "html")

            self.assertIn("<!doctype html>", exported)
            self.assertIn("Chat Budget", exported)
            self.assertIn("&lt;world&gt;", exported)

    def test_project_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = ProjectState(
                version=2,
                tokenizer_path="tok",
                compare_tokenizer_path="compare",
                input_text="hello",
                encode_special_tokens=True,
                export_format="html",
                budget_limit=32,
                chat_messages=(ChatMessage("user", "hello"),),
                add_generation_prompt=False,
                corpus_path="corpus",
                batch_path="batch",
                active_tab="unicode-tab",
                selected_source="compare",
                pricing_profile=PricingProfile("team", 1.25, 2.5, 128),
            )
            path = Path(tmp) / "project.json"

            save_project_state(project, path)
            loaded = load_project_state(path)

            self.assertEqual(loaded.input_text, "hello")
            self.assertTrue(loaded.encode_special_tokens)
            self.assertEqual(loaded.chat_messages[0].content, "hello")
            self.assertEqual(loaded.active_tab, "unicode-tab")
            self.assertEqual(loaded.pricing_profile.name, "team")
            self.assertEqual(loaded.pricing_profile.input_per_million, 1.25)

    def test_project_state_v1_defaults_cost_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_project.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "tokenizer_path": "tok",
                        "input_text": "hello",
                        "chat_messages": [{"role": "user", "content": "hello"}],
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_project_state(path)

            self.assertEqual(loaded.version, 1)
            self.assertEqual(loaded.pricing_profile.name, "custom")
            self.assertEqual(loaded.pricing_profile.input_per_million, 0.0)
            self.assertEqual(loaded.pricing_profile.output_per_million, 0.0)
            self.assertEqual(loaded.pricing_profile.estimated_output_tokens, 0)

    def test_pricing_profiles_round_trip_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            profiles = (
                PricingProfile("cheap", 1.5, 2.25, 300),
                PricingProfile("expensive", 10.0, 20.0, 1000),
            )

            save_pricing_profiles(profiles, workspace)
            loaded = load_pricing_profiles(workspace)

            self.assertEqual(loaded, profiles)

            (workspace / ".tokenscope_pricing.json").write_text("{bad json", encoding="utf-8")
            fallback = load_pricing_profiles(workspace)
            self.assertEqual(fallback, (PricingProfile("custom", 0.0, 0.0, 0),))

            (workspace / ".tokenscope_pricing.json").write_text(
                json.dumps(
                    {
                        "profiles": [
                            {"name": "valid", "input_per_million": 1, "output_per_million": 2, "estimated_output_tokens": 3},
                            {"name": "invalid", "input_per_million": "nope"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_pricing_profiles(workspace)
            self.assertEqual(loaded, (PricingProfile("valid", 1.0, 2.0, 3),))

    def test_next_feature_analyzers_and_export_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            primary = make_word_engine(workspace / "primary")
            compare = make_word_engine(workspace / "compare")
            tokenized = primary.encode("hello world alpha beta")

            diff = diff_tokenizers(primary, compare)
            packing = simulate_packing(primary, tokenized.input_text, budget_limit=2)
            regression = run_regression_suite(
                "suite",
                (
                    RegressionCase("pass", "hello", False, primary.encode("hello").token_ids, None),
                    RegressionCase("fail", "hello", False, (999,), None),
                ),
                primary,
            )
            unicode_result = inspect_unicode("a\u200be\u0301")
            rag = analyze_rag_chunks(primary, ["hello world alpha beta"], max_tokens=2, overlap_tokens=1)
            distribution = distribution_from_counts("sample", [1, 2, 3, 10], budget_limit=2)
            cost = estimate_token_cost(
                PricingProfile("test", input_per_million=1.0, output_per_million=2.0, estimated_output_tokens=500),
                input_tokens=1_000,
            )
            repair = suggest_tokenizer_repairs(primary)
            payload = build_export_payload(
                primary,
                tokenized,
                compare,
                tokenizer_diff=diff,
                packing_result=packing,
                regression_result=regression,
                unicode_result=unicode_result,
                rag_result=rag,
                distribution_result=distribution,
                cost_estimate=cost,
                repair_result=repair,
            )

            self.assertIsNotNone(diff)
            self.assertGreater(packing.dropped_tokens if packing else 0, 0)
            self.assertEqual(regression.failed_cases, 1)
            self.assertEqual(unicode_result.zero_width_count, 1)
            self.assertGreater(rag.chunk_count if rag else 0, 1)
            self.assertEqual(distribution.budget_failures, 2)
            self.assertGreater(cost.total_cost, 0)
            self.assertIsNotNone(repair)
            exported = format_export(payload, "html")
            self.assertIn("Tokenizer Diff", exported)
            self.assertIn("Packing", exported)
            self.assertIn("Regression", exported)
            self.assertIn("Unicode", exported)
            self.assertIn("RAG Chunking", exported)
            self.assertIn("Distribution", exported)
            self.assertIn("Cost", exported)
            self.assertIn("Repair", exported)


if __name__ == "__main__":
    unittest.main()
