# Owner(s): ["oncall: quantization"]

import unittest

import torch
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize
from torch.ao.quantization.experimental.fake_quantize_function import (
    fake_quantize_function,
)
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import dequantize_APoT, quantize_APoT
from torch.testing._internal.common_utils import HardwareClassification


forward_helper = fake_quantize_function.forward
backward = fake_quantize_function.backward


class TestFakeQuantize(unittest.TestCase):
    r""" Tests fake quantize calculate_qparams() method
         by comparing with result from observer calculate_qparams.
         Uses hard-coded values: alpha=1.0, b=4, k=2.
    """
    hw_classification = HardwareClassification.GENERIC

    def test_fake_calc_qparams(self):
        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.activation_post_process.min_val = torch.tensor([0.0])
        apot_fake.activation_post_process.max_val = torch.tensor([1.0])

        alpha, gamma, quantization_levels, level_indices = apot_fake.calculate_qparams(signed=False)

        observer = APoTObserver(b=4, k=2)
        observer.min_val = torch.tensor([0.0])
        observer.max_val = torch.tensor([1.0])

        qparams_expected = observer.calculate_qparams(signed=False)

        self.assertEqual(alpha, qparams_expected[0])
        self.assertTrue(torch.equal(gamma, qparams_expected[1]))
        self.assertTrue(torch.equal(quantization_levels, qparams_expected[2]))
        self.assertTrue(torch.equal(level_indices, qparams_expected[3]))

    r""" Tests fake quantize forward() method
         by comparing result with expected
         quant_dequant_APoT mapping of input tensor.
         Uses input tensor with random values from 0 -> 1000
         and APoT observer with hard-coded values b=4, k=2
    """
    def test_forward(self):
        # generate a tensor of size 20 with random values
        # between 0 -> 1000 to quantize -> dequantize
        X = 1000 * torch.rand(20)

        observer = APoTObserver(b=4, k=2)
        observer.forward(X)
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.enable_observer()
        apot_fake.enable_fake_quant()

        X_reduced_precision_fp = apot_fake.forward(torch.clone(X))

        # get X_expected by converting fp -> apot -> fp to simulate quantize -> dequantize
        X_to_apot = quantize_APoT(X, alpha, gamma, quantization_levels, level_indices)
        X_expected = dequantize_APoT(X_to_apot)

        self.assertTrue(torch.equal(X_reduced_precision_fp, X_expected))

    r""" Tests fake quantize forward() method
         throws error when qparams are None
    """
    def test_forward_exception(self):
        # generate a tensor of size 20 with random values
        # between 0 -> 1000 to quantize -> dequantize
        X = 1000 * torch.rand(20)

        apot_fake = APoTFakeQuantize(b=4, k=2)
        # disable observer so qparams not set, qparams are all None
        apot_fake.disable_observer()
        apot_fake.enable_fake_quant()

        with self.assertRaises(Exception):
            apot_fake.forward(torch.clone(X))

    r""" Tests fake quantize helper backward() method
         using its straight-through clipping mask.
    """
    def test_backward(self):
        input = torch.tensor(
            [-0.5, -0.25, 0.0, 0.25, 0.5], dtype=torch.double, requires_grad=True
        )
        input_before = input.detach().clone()

        observer = APoTObserver(b=4, k=2)
        observer(input)
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        output = fake_quantize_function.apply(
            input, alpha, gamma, quantization_levels, level_indices
        )
        torch.testing.assert_close(input.detach(), input_before)

        output.sum().backward()
        expected_grad = ((input_before >= -alpha) & (input_before <= alpha)).to(
            input.dtype
        )
        torch.testing.assert_close(input.grad, expected_grad)

if __name__ == '__main__':
    unittest.main()
