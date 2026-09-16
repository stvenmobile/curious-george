import torch

from curious_george.curiosity_tools.loss_measurement import compute_probe_loss, evaluate_recall_probes, measure_content_loss

VOCAB_SIZE = 20
PROMPT_LEN = 3
TARGET_LEN = 2

PROMPT_TEXT = "test prompt"
TARGET_TEXT = " test target"
FULL_TEXT = PROMPT_TEXT + TARGET_TEXT

# Aligned so the fake model's position-based logits are "correct":
# logits[prompt_len-1+k] has argmax (prompt_len-1+k) % VOCAB_SIZE, and we
# want that to equal target_ids[k].
ALIGNED_TARGET_IDS = [PROMPT_LEN - 1 + k for k in range(TARGET_LEN)]     # [2, 3]
SHIFTED_TARGET_IDS = [PROMPT_LEN + k for k in range(TARGET_LEN)]        # [3, 4] - every value +1


class FakeBatchEncoding(dict):
    def to(self, device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


class FakeTokenizer:
    """Returns a fixed, controllable token sequence per known text -
    content doesn't matter to the fake model (its logits are purely
    position-based), only length and, for the full text, the specific
    target-position values under test."""
    def __init__(self, target_ids):
        self.target_ids = target_ids

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        assert add_special_tokens is False
        if text == PROMPT_TEXT:
            ids = [0] * PROMPT_LEN
        elif text == FULL_TEXT:
            ids = [0] * PROMPT_LEN + self.target_ids
        else:
            raise ValueError(f"unexpected text: {text!r}")
        return FakeBatchEncoding(input_ids=torch.tensor([ids], dtype=torch.long))


class FakeOutputs:
    def __init__(self, logits):
        self.logits = logits


class PositionIdentityFakeModel:
    """logits[0, i, :] is a strong one-hot at class (i % VOCAB_SIZE) -
    fully deterministic and content-independent, so the exact
    position/target alignment compute_probe_loss computes can be checked
    by hand rather than trusting a real model's behavior."""
    def __call__(self, input_ids, attention_mask):
        seq_len = input_ids.shape[1]
        logits = torch.full((1, seq_len, VOCAB_SIZE), -10.0)
        for i in range(seq_len):
            logits[0, i, i % VOCAB_SIZE] = 10.0
        return FakeOutputs(logits=logits)


def test_correctly_aligned_target_scores_perfect_accuracy_and_near_zero_loss():
    model = PositionIdentityFakeModel()
    aligned_tokenizer = FakeTokenizer(ALIGNED_TARGET_IDS)
    loss, accuracy = compute_probe_loss(model, aligned_tokenizer, "cpu", PROMPT_TEXT, TARGET_TEXT)
    assert accuracy == 1.0, f"expected perfect accuracy with correctly-aligned targets, got {accuracy}"
    assert loss.item() < 0.01, f"expected near-zero loss, got {loss.item()}"


def test_shifted_by_one_target_scores_zero_accuracy():
    model = PositionIdentityFakeModel()
    aligned_tokenizer = FakeTokenizer(ALIGNED_TARGET_IDS)
    loss, _ = compute_probe_loss(model, aligned_tokenizer, "cpu", PROMPT_TEXT, TARGET_TEXT)

    shifted_tokenizer = FakeTokenizer(SHIFTED_TARGET_IDS)
    loss2, accuracy2 = compute_probe_loss(model, shifted_tokenizer, "cpu", PROMPT_TEXT, TARGET_TEXT)
    assert accuracy2 == 0.0, f"expected zero accuracy with every target shifted by one, got {accuracy2}"
    assert loss2.item() > loss.item(), "shifted (wrong) targets should score a strictly higher loss than aligned ones"


def test_evaluate_recall_probes_averages_correctly_across_probes():
    model = PositionIdentityFakeModel()
    aligned_tokenizer = FakeTokenizer(ALIGNED_TARGET_IDS)
    loss, _ = compute_probe_loss(model, aligned_tokenizer, "cpu", PROMPT_TEXT, TARGET_TEXT)
    shifted_tokenizer = FakeTokenizer(SHIFTED_TARGET_IDS)
    loss2, _ = compute_probe_loss(model, shifted_tokenizer, "cpu", PROMPT_TEXT, TARGET_TEXT)

    class DualProbeTokenizer:
        """Serves two different probes: probe A is perfectly aligned, probe B
        is deliberately shifted - lets evaluate_recall_probes's averaging be
        checked against a known expected result (0.5 accuracy, not 1.0 or 0.0)."""
        def __call__(self, text, return_tensors="pt", add_special_tokens=False):
            assert add_special_tokens is False
            if text == "prompt a":
                ids = [0] * PROMPT_LEN
            elif text == "prompt a" + " target a":
                ids = [0] * PROMPT_LEN + ALIGNED_TARGET_IDS
            elif text == "prompt b":
                ids = [0] * PROMPT_LEN
            elif text == "prompt b" + " target b":
                ids = [0] * PROMPT_LEN + SHIFTED_TARGET_IDS
            else:
                raise ValueError(f"unexpected text: {text!r}")
            return FakeBatchEncoding(input_ids=torch.tensor([ids], dtype=torch.long))

    probes = [
        {"prompt": "prompt a", "target": " target a"},
        {"prompt": "prompt b", "target": " target b"},
    ]
    avg_loss, avg_accuracy = evaluate_recall_probes(model, DualProbeTokenizer(), "cpu", probes)
    assert avg_accuracy == 0.5, f"expected averaged accuracy of exactly 0.5 (one perfect, one zero), got {avg_accuracy}"
    expected_avg_loss = (loss.item() + loss2.item()) / 2
    assert abs(avg_loss - expected_avg_loss) < 1e-6, f"expected averaged loss {expected_avg_loss}, got {avg_loss}"


def test_measure_content_loss_tokenizes_without_special_tokens_and_uses_labels_equal_input_ids():
    class FakeTokenizerForContent:
        def __call__(self, text, return_tensors="pt", add_special_tokens=False):
            assert add_special_tokens is False
            ids = [ord(c) % 50 for c in text]
            return FakeBatchEncoding(input_ids=torch.tensor([ids], dtype=torch.long))

    class FakeModelWithLoss:
        def __init__(self):
            self.last_call = None

        def __call__(self, input_ids, attention_mask, labels):
            assert torch.equal(labels, input_ids), "plain causal-LM loss needs labels == input_ids"
            assert torch.equal(attention_mask, torch.ones_like(input_ids)), "no padding here - every token is real"
            self.last_call = {"input_ids": input_ids}

            class FakeOutputs:
                loss = input_ids.float().mean()

            return FakeOutputs()

    model = FakeModelWithLoss()
    loss = measure_content_loss(model, FakeTokenizerForContent(), "cpu", "abc")

    expected_ids = torch.tensor([[ord(c) % 50 for c in "abc"]], dtype=torch.long)
    assert model.last_call["input_ids"].tolist() == expected_ids.tolist()
    assert abs(loss - expected_ids.float().mean().item()) < 1e-6
    assert isinstance(loss, float), "should return a plain float, not a tensor"
