import numpy as np
from scripts.pin_reference_augmentation import augment_pair


def test_reproducible_paired_photometric_labels_unchanged():
    image=np.full((120,160,3),130,np.uint8)
    points=np.array([[.44,.55],[.51,.60]],np.float32);valid=np.array([True,False])
    a=augment_pair(image,image,points,valid,np.random.default_rng(41))
    b=augment_pair(image,image,points,valid,np.random.default_rng(41))
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(a[2],points)
    assert not a[3][1]
    assert a[0].dtype==np.uint8 and a[1].shape==image.shape
    assert not np.array_equal(a[0],image)
    np.testing.assert_array_equal(image,np.full_like(image,130))


def test_seed_sweep_keeps_coordinates_and_never_unmasks():
    image=np.full((80,100,3),180,np.uint8)
    points=np.array([[.4,.5],[.6,.6]])
    for seed in range(100):
        current,ref,k,v=augment_pair(image,image,points,[False,True],np.random.default_rng(seed))
        np.testing.assert_array_equal(k,points)
        assert not v[0] and current.shape==ref.shape==image.shape
