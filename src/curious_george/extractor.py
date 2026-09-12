"""
Residual stream activation extractor.

Ported from piper_assistant's src/piper_geometry/extractor.py (the
cross-model reasoning-geometry track) - genuinely general-purpose
infrastructure for instrumenting a transformer's intermediate layers,
not specific to that track's research question. curious-george's
QwenHiddenStateEmbedder reuses it as-is to repurpose Qwen's own hidden
states as a retrieval embedding.
"""

from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ResidualExtractor:
    """Instruments a transformer model's residual stream with non-invasive forward hooks."""

    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"[Extractor] Loading probe model '{model_name_or_path}' onto {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Load weights and explicitly cast to device without device_map dependency
        model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=model_dtype
        ).to(self.device)
        self.model.eval()

        self.layers = self._resolve_layers()
        self.num_layers = len(self.layers)
        self.captured_activations: Dict[int, torch.Tensor] = {}
        self._hooks = []
        print(f"[Extractor] Model ready. Detected {self.num_layers} transformer blocks.")

    def _resolve_layers(self) -> torch.nn.ModuleList:
        """Finds the layer ModuleList across different standard architectures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers  # LLaMA, Qwen, Mistral
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h  # GPT-Neo, GPT-2
        raise AttributeError("Could not resolve transformer layer stack for this model architecture.")

    def _register_hook(self, layer_idx: int):
        """Creates a forward hook callback for a specific layer."""
        def hook_fn(module, input_tensor, output_tensor):
            hidden = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            self.captured_activations[layer_idx] = hidden.detach().clone()
        return hook_fn

    def attach_hooks(self, layer_indices: Optional[List[int]] = None):
        """Attaches hooks to the requested layer indices."""
        self.clear_hooks()
        targets = layer_indices if layer_indices is not None else list(range(self.num_layers))

        for idx in targets:
            if 0 <= idx < self.num_layers:
                hook = self.layers[idx].register_forward_hook(self._register_hook(idx))
                self._hooks.append(hook)

    def extract_activations(self, prompt: str, target_layers: List[int]) -> Dict[int, torch.Tensor]:
        """Runs a forward pass and returns captured hidden states at the last token position."""
        self.attach_hooks(target_layers)
        self.captured_activations.clear()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            self.model(**inputs)

        results: Dict[int, torch.Tensor] = {}
        for layer_idx in target_layers:
            if layer_idx in self.captured_activations:
                # Isolate the last token vector [hidden_dim]
                last_token_vec = self.captured_activations[layer_idx][0, -1, :].cpu()
                # Normalize vector to unit length
                norm_vec = last_token_vec / torch.norm(last_token_vec, p=2)
                results[layer_idx] = norm_vec

        self.clear_hooks()
        return results

    def clear_hooks(self):
        """Removes active forward hooks to avoid memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.captured_activations.clear()
