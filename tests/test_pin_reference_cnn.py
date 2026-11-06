"""Offline paired model contracts, no hardware imports or external data."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from app.pin_insertion_cnn import HeatmapPinInsertionCNN
from app.pin_reference_cnn import PinReferenceCNN,paired_tensor,PinReferencePredictor,SCHEMA
from scripts.train_pin_reference_cnn import geometric_pair

def test_zero_reference_weights_preserve_rgb_ancestor():
    torch.set_num_threads(2);torch.manual_seed(4)
    source=HeatmapPinInsertionCNN().eval();paired=PinReferenceCNN.from_rgb_model(source).eval()
    current=torch.rand(2,3,48,48);reference=torch.rand_like(current);joints=torch.rand(2,12)
    with torch.inference_mode():
        original=source(current,joints,return_logits=True)
        actual=paired(torch.cat((current,reference,current-reference),1),joints,return_logits=True)
    for key in original:torch.testing.assert_close(original[key],actual[key],rtol=1e-5,atol=1e-6)
    assert torch.count_nonzero(paired.vision[0].weight[:,3:])==0

def test_signed_difference_and_invalid_pairs():
    white=np.full((20,24,3),255,np.uint8);black=np.zeros_like(white)
    actual=paired_tensor(black,white,32)
    assert actual.shape==(1,9,32,32)
    assert torch.all(actual[:,6:]==-1)
    with pytest.raises(ValueError):paired_tensor(white,black[:19],32)
    with pytest.raises(ValueError):paired_tensor(white.astype(float),black,32)

def test_paired_checkpoint_predictor_and_schema(tmp_path):
    model=PinReferenceCNN();path=tmp_path/'paired.pt'
    torch.save({'model':model.state_dict(),'metadata':{'schema':SCHEMA,'image_size':32}},path)
    predictor=PinReferencePredictor(path)
    out=predictor.predict(np.zeros((20,24,3),np.uint8),np.zeros((20,24,3),np.uint8),np.zeros(14))
    assert len(out['keypoints'])==4 and np.isfinite(out['keypoints']).all()
    torch.save({'model':model.state_dict(),'metadata':{'schema':'wrong'}},path)
    with pytest.raises(ValueError):PinReferencePredictor(path)

def test_shared_affine_labels_follow_image_and_reference_jitter_is_small():
    pytest.importorskip('cv2')
    image=np.zeros((100,100,3),np.uint8);image[45:56,45:56]=255
    points=np.array([[.5,.5],[.7,.7]],np.float32);valid=np.array([True,True])
    current,reference,transformed,new_valid=geometric_pair(image,image,points,valid,np.random.default_rng(42))
    y,x=np.unravel_index(current[:,:,0].argmax(),current.shape[:2])
    # The original square's bright interior remains within its transformed label neighborhood.
    assert np.linalg.norm(np.array([x,y])/100-transformed[0])<.1
    assert current.shape==reference.shape==image.shape and new_valid.all()

def test_weak_box_optimizes_mass_without_exact_point():
    from scripts.train_pin_reference_cnn import weak_box_loss
    logits=torch.zeros(1,2,8,8,requires_grad=True)
    loss=weak_box_loss(logits,torch.tensor([[.25,.25,.5,.5]]))
    loss.backward()
    assert torch.all(logits.grad[0,0,2:4,2:4]<0)
    assert torch.all(logits.grad[0,1]==0)
    assert weak_box_loss(logits,torch.full((1,4),float('nan'))).item()==0

def test_continuation_exempts_only_verified_same_partition_ids(tmp_path):
    import json,hashlib
    from scripts.train_pin_reference_cnn import continuation_metadata
    original={'frames':[{'id':'a','episode_id':'dev-a','split':'train','image_path':'a.png','image_sha256':'hash-a','state14':[0]*14},{'id':'b','episode_id':'dev-b','split':'validation','image_path':'b.png','image_sha256':'hash-b','state14':[1]*14}]}
    source=tmp_path/'source.json';source.write_text(json.dumps(original));digest=hashlib.sha256(source.read_bytes()).hexdigest()
    paired=dict(original,source_manifest_path=str(source),source_manifest_sha256=digest)
    path=tmp_path/'paired.json';path.write_text(json.dumps(paired))
    metadata={'adaptation':{'manifest_sha256':digest,'train_episode_ids':['dev-a','other-source'],'validation_episode_ids':['dev-b']}}
    adjusted=continuation_metadata(path,metadata)
    assert adjusted['adaptation']['train_episode_ids']==['other-source']
    assert metadata['adaptation']['validation_episode_ids']==['dev-b']
    paired['frames'][1]['split']='train';path.write_text(json.dumps(paired))
    with pytest.raises(ValueError):continuation_metadata(path,metadata)
