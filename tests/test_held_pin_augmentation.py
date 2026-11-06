import torch
from app.held_pin_augmentation import augment_batch, apply_shadow_lighting
from app.carrier_face_augmentation import augment_batch as carrier_augment


def test_deterministic_finite_and_does_not_mutate_inputs():
    x = torch.full((4,3,24,32),.8)
    p = torch.full((4,10,2),.5)
    v = torch.ones((4,10),dtype=torch.bool)
    first = augment_batch(x,p,v,torch.Generator().manual_seed(19))
    second = augment_batch(x,p,v,torch.Generator().manual_seed(19))
    for a,b in zip(first,second):
        assert torch.equal(a,b)
        assert torch.isfinite(a).all()
    assert first[0].min() >= 0 and first[0].max() <= 1
    assert (x == .8).all() and (p == .5).all() and v.all()


def test_shadow_wrapper_preserves_carrier_geometry_and_visibility():
    x = torch.ones((3,3,24,32))
    p = torch.rand((3,10,2))
    v = torch.ones((3,10),dtype=torch.bool)
    v[0,0] = False
    base = carrier_augment(x,p,v,torch.Generator().manual_seed(81))
    result = augment_batch(x,p,v,torch.Generator().manual_seed(81))
    assert torch.equal(base[1],result[1])
    assert torch.equal(base[2],result[2])
    assert (result[0] <= base[0]).all()


def test_severe_darkness_and_spatial_variation_are_actually_generated():
    x = torch.ones((16,3,32,40))
    result = apply_shadow_lighting(x,torch.Generator().manual_seed(42))
    assert result.min() >= .03
    assert result.min() <= .031
    assert (result < .1).float().mean() > .1
    assert result.std(dim=(-2,-1)).max() > .08
    assert not torch.equal(result[0],result[1])


def test_extreme_stress_darkens_cap_and_retains_peripheral_context():
    from app.held_pin_augmentation import apply_shadows
    x = torch.ones((4,3,96,128))
    a = apply_shadows(x,torch.Generator().manual_seed(5),extreme=True)
    b = apply_shadows(x,torch.Generator().manual_seed(5),extreme=True)
    assert torch.equal(a,b)
    assert a[:,:,62,67].max() < .05
    assert a.max() > .1
