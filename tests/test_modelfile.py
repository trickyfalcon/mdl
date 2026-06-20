"""Tests for Modelfile generation + Ollama model naming."""

from mdl.registry.ollama import build_modelfile, model_name_for


def test_build_modelfile_quotes_absolute_path():
    content = build_modelfile(r"D:\models\gguf\bartowski\Qwen3-32B-GGUF\Qwen3-32B-Q4_K_M.gguf")
    assert content == 'FROM "D:\\models\\gguf\\bartowski\\Qwen3-32B-GGUF\\Qwen3-32B-Q4_K_M.gguf"\n'


def test_build_modelfile_handles_spaces_in_path():
    content = build_modelfile(r"D:\my models\file.gguf")
    assert content.startswith('FROM "')
    assert 'FROM "D:\\my models\\file.gguf"' in content


def test_build_modelfile_extra_lines():
    content = build_modelfile(r"D:\x\f.gguf", extra_lines=["PARAMETER temperature 0.7"])
    assert content.splitlines()[0] == 'FROM "D:\\x\\f.gguf"'
    assert "PARAMETER temperature 0.7" in content


def test_model_name_for_with_quant():
    assert model_name_for("bartowski/Qwen3-32B-GGUF", "Q4_K_M") == "qwen3-32b:q4_k_m"
    assert model_name_for("Qwen/Qwen3-32B", "Q5_K_M") == "qwen3-32b:q5_k_m"


def test_model_name_for_strips_gguf_suffix():
    assert model_name_for("TheBloke/Llama-2-7B-GGUF", "Q4_0") == "llama-2-7b:q4_0"


def test_model_name_for_without_quant():
    assert model_name_for("org/My_Model", None) == "my_model"
