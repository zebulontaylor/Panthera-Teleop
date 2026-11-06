"""Training augmentation preserves geometry and visibility contracts."""
import pytest
torch=pytest.importorskip('torch')
from app.carrier_face_augmentation import augment_batch,transform_points

def test_forward_map_and_image_inverse_have_same_coordinates():
    # Exact known pixel-space affine, independent of random photometric effects.
    p=torch.tensor([[[.2,.3],[.8,.9]]])
    matrix=torch.tensor([[[1.,0.,.04],[0.,1.,-.03]]])
    torch.testing.assert_close(transform_points(p,matrix),p+torch.tensor([.04,-.03]))

def test_deterministic_finite_and_original_hidden_stays_hidden():
    torch.set_num_threads(2)
    x=torch.linspace(0,1,3*32*48).reshape(1,3,32,48).expand(8,-1,-1,-1)
    p=torch.full((8,10,2),.5);visible=torch.ones(8,10,dtype=torch.bool);visible[:,0]=False
    a=augment_batch(x,p,visible,torch.Generator().manual_seed(22))
    b=augment_batch(x,p,visible,torch.Generator().manual_seed(22))
    for first,second in zip(a,b):torch.testing.assert_close(first,second)
    assert torch.isfinite(a[0]).all() and a[0].min()>=0 and a[0].max()<=1
    assert not a[2][:,0].any()
    assert not torch.equal(a[0],x)

def test_cropped_and_occluded_points_masked_without_clipping_coordinates():
    x=torch.ones(128,3,32,32);p=torch.rand(128,10,2);v=torch.ones(128,10,dtype=torch.bool)
    p[:,0]=torch.tensor([-1.,2.]);v[:,1]=False
    _,points,visible=augment_batch(x,p,v,torch.Generator().manual_seed(7))
    inside=(points>=0).all(-1)&(points<=1).all(-1)
    assert not visible[:,0].any() and not visible[:,1].any()
    assert not visible[~inside].any()
    # Some interior landmarks are removed by actual image occluders.
    assert ((~visible[:,2:])&inside[:,2:]).any()
    assert (points[:,0]<0).any()

def test_grid_sampling_inverse_matches_forward_rotation_and_translation():
    from app.carrier_face_augmentation import sampling_transform
    matrix=torch.tensor([[[.97,-.1,.03],[.1,.97,-.02]]])
    original=torch.tensor([[[.2,.3],[.6,.7],[.4,.8]]])
    transformed=transform_points(original,matrix)
    inverse=sampling_transform(matrix)
    reconstructed=torch.bmm(transformed*2-1,inverse[:,:,:2].transpose(1,2))+inverse[:,:,2][:,None,:]
    torch.testing.assert_close(reconstructed,original*2-1,atol=1e-6,rtol=1e-6)
