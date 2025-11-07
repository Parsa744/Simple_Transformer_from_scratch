SimpleTransformer: Minimal Transformer Components in PyTorch

This module provides small, educational implementations of the core building blocks of the Transformer architecture—written from scratch using PyTorch.
The goal is to offer readable code that demonstrates how attention, feed-forward layers, and residual connections work without depending on PyTorch’s high-level nn.Transformer module.

Install dependencies with:
```
pip install torch numpy
```

Simple Transformer Block:
```
transformer = SimpleTransformer(input_size=4, contex_size=6, hidden_dim=16)

inputs = torch.randn(2, 10, 4)
context = torch.randn(2, 10, 6)

output = transformer(inputs, context)
print(output.shape)

```
