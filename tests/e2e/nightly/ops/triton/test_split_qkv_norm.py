#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import pytest
import torch

from tests.conftest import init_device_properties_triton
from vllm_ascend.ops.triton.split_qkv_rmsnorm import split_qkv_rmsnorm_impl

DEVICES = ["npu"]
NUM_TOKENS = [1, 4, 16, 32]
NUM_QKV_HEADS = [(8, 2), (16, 4), (32, 8)]
HEAD_SIZES = [128]
EPS = [1e-6]
DTYPES = [torch.bfloat16]
SEEDS = [0]


def custom_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float,
                   bias: torch.Tensor = None) -> torch.Tensor:
    """Custom implementation of RMSNorm."""
    original_shape = x.shape
    head_dim = weight.shape[0]
    x_reshaped = x.view(-1, head_dim).float()
    variance = x_reshaped.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_reshaped * torch.rsqrt(variance + eps)
    if bias is not None:
        output = (x_normed * weight.float() + bias.float()).to(x.dtype)
    else:
        output = (x_normed * weight.float()).to(x.dtype)
    return output.view(original_shape)


def custom_split_qkv_rmsnorm(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor = None,
    k_bias: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom implementation combining split and RMSNorm."""
    q = input[:, :q_hidden_size]
    k = input[:, q_hidden_size:q_hidden_size + kv_hidden_size]
    v = input[:, q_hidden_size + kv_hidden_size:]

    q_normed = custom_rmsnorm(q, q_weight, eps, q_bias)
    k_normed = custom_rmsnorm(k, k_weight, eps, k_bias)

    return q_normed, k_normed, v.clone()


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
def test_split_qkv_rmsnorm_no_bias(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    dtype: torch.dtype,
    seed: int,
    device: str,
):
    """Test split_qkv_rmsnorm without bias."""
    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()

    q_hidden_size = num_q_heads * head_size
    kv_hidden_size = num_kv_heads * head_size
    total_hidden_size = q_hidden_size + kv_hidden_size * 2

    input_tensor = torch.randn(num_tokens, total_hidden_size, dtype=dtype)
    q_weight = torch.randn(head_size, dtype=dtype)
    k_weight = torch.randn(head_size, dtype=dtype)

    q_ref, k_ref, v_ref = custom_split_qkv_rmsnorm(
        input_tensor,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hidden_size,
        head_size,
        eps,
    )

    q_out, k_out, v_out = split_qkv_rmsnorm_impl(
        input_tensor,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hidden_size,
        head_size,
        eps,
    )

    torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out, k_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_out, v_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
def test_split_qkv_rmsnorm_with_bias(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    dtype: torch.dtype,
    seed: int,
    device: str,
):
    """Test split_qkv_rmsnorm with bias."""
    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()

    q_hidden_size = num_q_heads * head_size
    kv_hidden_size = num_kv_heads * head_size
    total_hidden_size = q_hidden_size + kv_hidden_size * 2

    input_tensor = torch.randn(num_tokens, total_hidden_size, dtype=dtype)
    q_weight = torch.randn(head_size, dtype=dtype)
    k_weight = torch.randn(head_size, dtype=dtype)
    q_bias = torch.randn(head_size, dtype=dtype)
    k_bias = torch.randn(head_size, dtype=dtype)

    q_ref, k_ref, v_ref = custom_split_qkv_rmsnorm(
        input_tensor,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hidden_size,
        head_size,
        eps,
        q_bias,
        k_bias,
    )

    q_out, k_out, v_out = split_qkv_rmsnorm_impl(
        input_tensor,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hidden_size,
        head_size,
        eps,
        q_bias,
        k_bias,
    )

    torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out, k_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_out, v_ref, atol=1e-2, rtol=1e-2)
