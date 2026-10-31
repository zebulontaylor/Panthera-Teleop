import pytest
import torch

from infra.training.trajectory_loss import timing_tolerant_loss


def loss(x, y, **kwargs):
    return timing_tolerant_loss(x, y, scale=torch.ones(x.shape[-1]), **kwargs)


def test_speed_change_costs_less_than_spatial_error():
    # Same line, same endpoints; only the sampling clock changes.
    t = torch.linspace(0, 1, 31)
    reference = torch.stack((t, torch.zeros_like(t)), -1)[None]
    retimed = torch.stack((t.square(), torch.zeros_like(t)), -1)[None]
    spatial = reference + torch.tensor([0.0, 0.2])
    speed = loss(retimed, reference, gamma=0.001)
    precision = loss(spatial, reference, gamma=0.001)
    assert speed['aligned'] < speed['synchronous'] * 0.2
    assert speed['loss'] < precision['loss'] * 0.25
    assert loss(torch.zeros_like(reference), reference)['loss'] > speed['loss']


def test_identity_and_finite_gradients():
    torch.manual_seed(7)
    target = torch.randn(2, 7, 3, dtype=torch.double)
    same = target.clone().requires_grad_()
    result = loss(same, target)
    assert result['loss'].abs() < 1e-12
    result['loss'].backward()
    assert torch.isfinite(same.grad).all()
    prediction = (target + 0.1).requires_grad_()
    loss(prediction, target)['loss'].backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_zero_band_equals_mse():
    x, y = torch.randn(2, 8, 3), torch.randn(2, 8, 3)
    result = loss(x, y, max_shift=0)
    torch.testing.assert_close(result['loss'], (x-y).square().mean())


def test_padding_excluded_even_when_nan_and_has_zero_gradient():
    x = torch.tensor([[[0.], [0.6], [1.], [float('nan')]]], requires_grad=True)
    y = torch.tensor([[[0.], [0.5], [1.], [float('nan')]]])
    valid = torch.tensor([[True, True, True, False]])
    result = loss(x, y, valid=valid)
    expected = loss(x[:, :3], y[:, :3])
    torch.testing.assert_close(result['loss'], expected['loss'])
    result['loss'].backward()
    assert x.grad[0, 3, 0] == 0
    assert torch.isfinite(x.grad).all()


def test_reversing_trajectory_is_penalized():
    x = torch.linspace(0, 1, 15)[None, :, None]
    assert loss(x.flip(1), x)['loss'] > 0.1


def test_autograd_matches_finite_difference():
    x = torch.tensor([[[0.1], [0.4], [0.9]]], dtype=torch.double, requires_grad=True)
    y = torch.tensor([[[0.], [0.5], [1.]]], dtype=torch.double)
    assert torch.autograd.gradcheck(lambda z: loss(z, y, gamma=0.1)['loss'], (x,))


@pytest.mark.parametrize('valid', [
    [[False, False, False]], [[True, False, True]],
])
def test_invalid_masks_rejected(valid):
    x = torch.zeros(1, 3, 2)
    with pytest.raises(ValueError, match='prefix'):
        loss(x, x, valid=torch.tensor(valid))


def test_single_frame_and_scale():
    x, y = torch.ones(1, 1, 2), torch.zeros(1, 1, 2)
    result = timing_tolerant_loss(x, y, scale=torch.tensor([2., 2.]))
    torch.testing.assert_close(result['loss'], torch.tensor(0.25))
