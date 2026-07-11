"""
=====================================================
Multi-Head Attention Demo
=====================================================

Author : Vasanth

This is a simplified demonstration showing how
multiple attention heads can produce different
representations of the same input.
"""

import numpy as np

np.random.seed(42)

# Sample embeddings (3 tokens × 4 features)
X = np.random.rand(3, 4)

print("=" * 60)
print("Input Embeddings")
print("=" * 60)
print(X)

# Simulate outputs from two attention heads
head1 = X @ np.random.rand(4, 4)
head2 = X @ np.random.rand(4, 4)

print("\nHead 1 Output")
print(head1)

print("\nHead 2 Output")
print(head2)

# Concatenate outputs
multi_head_output = np.concatenate([head1, head2], axis=1)

print("\nConcatenated Multi-Head Output")
print(multi_head_output)