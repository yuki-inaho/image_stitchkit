from __future__ import annotations
import dataclasses
import gc
import json
from pathlib import Path
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import stitchkit as sk

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tests/assets"

@pytest.fixture(scope="module")
def pair():
    return [sk.load(ASSETS / "synthetic/view0.png"), sk.load(ASSETS / "synthetic/view1.png")]

@pytest.fixture(scope="module")
def prepared(pair):
    return sk.prepare(pair)


def test_capabilities():
    c = sk.capabilities()
    assert c["version"] == "0.2.0"
    assert c["nanobind"] == "2.15.0"
    assert isinstance(c["cuda_available"], bool)


def test_immutable_options():
    o = sk.Options()
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.grid_size = 40
    assert o.with_updates(grid_size=40).grid_size == 40
    assert o.grid_size == 32
    assert json.loads(json.dumps(o.as_dict()))["device"] == "cpu"

@pytest.mark.parametrize("changes", [
    {"method":"invalid"}, {"device":"gpu"}, {"structures":"not-a-detector"},
    {"grid_size":0}, {"ratio_threshold":float("nan")}, {"tps_lambda":-1},
    {"max_features":False}, {"seed":1.2}, {"use_apap":1}, {"feather_width":float("inf")},
    {"lineae_device":"invalid"}, {"lineae_variant":10}, {"lineae_model":4},
])
def test_invalid_options(changes):
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        sk.Options(**changes)

@pytest.mark.parametrize("bad", [
    np.zeros((8,8,3),dtype=float), np.zeros((8,8,4),dtype=np.uint8),
    np.zeros((8,8),dtype=np.uint8), np.zeros((0,8,3),dtype=np.uint8),
    np.zeros((1,8,3),dtype=np.uint8), "not-array",
])
def test_input_contract(bad, pair):
    with pytest.raises((ValueError, TypeError)):
        sk.stitch([bad,pair[1]])


def test_single_array_rejected(pair):
    with pytest.raises(TypeError): sk.stitch(pair[0])
    with pytest.raises(ValueError): sk.stitch([pair[0]])
    with pytest.raises(ValueError): sk.stitch([pair[0]]*17)


def test_output_lifetime_and_readonly(prepared):
    result=sk.compose(prepared,sk.Options(method="rew"))
    rgb,mask=result.image,result.mask
    expected_rgb,expected_mask=rgb.copy(),mask.copy()
    assert not rgb.flags.writeable and not mask.flags.writeable
    with pytest.raises(ValueError): rgb[0,0,0]=2
    with pytest.raises(ValueError): rgb.setflags(write=True)
    del result
    gc.collect()
    np.testing.assert_array_equal(rgb,expected_rgb)
    np.testing.assert_array_equal(mask,expected_mask)
    # Memory views must also retain the owner rather than borrowing a dead vector.
    view=memoryview(rgb);del rgb;gc.collect()
    assert view.shape==expected_rgb.shape


def test_prepare_snapshot_independent(pair):
    originals=[a.copy() for a in pair]
    options=sk.Options(method="rew")
    scene=sk.prepare(originals,options)
    reference=sk.compose(scene,options).image.copy()
    for a in originals:a.fill(0)
    del originals;gc.collect()
    np.testing.assert_array_equal(sk.compose(scene,options).image,reference)
    assert scene.image_count==2 and scene.feature_extractions==2


def test_noncontiguous_inputs(pair):
    # Same pixels, a larger row stride; wrapper must explicitly copy, not reinterpret.
    padded=[]
    for a in pair:
        target=np.zeros((a.shape[0],a.shape[1]*2,3),dtype=np.uint8)
        target[:,::2]=a;padded.append(target[:,::2])
    o=sk.Options(method="rew")
    np.testing.assert_array_equal(sk.stitch(padded,o).image,sk.stitch(pair,o).image)


def test_prepared_configuration_key(prepared):
    with pytest.raises(sk.StitchError,match="Registration options"):
        sk.compose(prepared,sk.Options(seed=10))
    with pytest.raises(TypeError):sk.compose(object())

@pytest.mark.parametrize("method",["rew","niswgsp","gesgsp"])
def test_direct_prepared_equal(method,pair,prepared):
    o=sk.Options(method=method)
    direct,reused=sk.stitch(pair,o),sk.compose(prepared,o)
    np.testing.assert_array_equal(direct.image,reused.image)
    np.testing.assert_array_equal(direct.mask,reused.mask)
    d,r=sk.report(direct),sk.report(reused)
    assert d["schema_version"]==2
    assert not d["registration_reused"] and d["feature_extractions"]==2
    assert r["registration_reused"] and r["feature_extractions"]==0
    assert r["timing_ms"]["registration"]==0 and r["prepared_registration_ms"]>0
    assert d["timing_ms"]["total"] >= sum(d["timing_ms"][k] for k in ("registration","structure","geometry","render"))
    assert (r["structure_equations"]>0) == (method=="gesgsp")


def test_chain_feature_reuse(pair):
    scene=sk.prepare(pair+[sk.load(ASSETS/"synthetic/view2.png")])
    assert scene.feature_extractions==3 and scene.image_count==3


def test_gil_released(prepared):
    pulses=[]; stop=threading.Event(); ready=threading.Event()
    def heartbeat():
        ready.set()
        while not stop.is_set():
            pulses.append(time.perf_counter());time.sleep(.002)
    worker=threading.Thread(target=heartbeat);worker.start();ready.wait()
    begin=time.perf_counter()
    try:sk.compose(prepared,sk.Options(method="gesgsp"))
    finally:
        end=time.perf_counter();stop.set();worker.join()
    assert end-begin>.05, "Test work was too small to discriminate GIL release"
    assert any(begin+.015<t<end-.015 for t in pulses), "No Python heartbeat during native computation"


def test_concurrent_prepared_reads(prepared):
    o=sk.Options(method="rew")
    expected=sk.compose(prepared,o).image.copy()
    with ThreadPoolExecutor(max_workers=2) as executor:
        images=list(executor.map(lambda _:sk.compose(prepared,o).image,[0,1]))
    for image in images:np.testing.assert_array_equal(image,expected)


def test_explicit_device_behavior(prepared):
    auto=sk.report(sk.compose(prepared,sk.Options(method="rew",device="auto")))
    assert auto["actual_device"] in ("cpu","cuda")
    if not sk.capabilities()["cuda_available"]:
        with pytest.raises(sk.StitchError,match="CUDA"):
            sk.compose(prepared,sk.Options(device="cuda"))


def test_load_errors(tmp_path):
    with pytest.raises(sk.StitchError):sk.load(tmp_path/"missing.png")
    corrupt=tmp_path/"broken.png";corrupt.write_bytes(b"invalid PNG")
    with pytest.raises(sk.StitchError):sk.load(corrupt)
    with pytest.raises(TypeError):sk.load(corrupt,max_side=False)


def test_save_png_and_unicode_path(prepared,tmp_path):
    from PIL import Image
    r=sk.compose(prepared,sk.Options(method="rew"))
    path=tmp_path/"画像_試験.png";r.save(path)
    rgba=np.asarray(Image.open(path).convert("RGBA"))
    np.testing.assert_array_equal(rgba[...,:3],r.image)
    np.testing.assert_array_equal(rgba[...,3],r.mask)


def test_cli_native_parity(pair,tmp_path):
    from PIL import Image
    cli=ROOT/"build/release/stitch"
    if not cli.exists():pytest.skip("Native CLI not built in this checkout")
    target=tmp_path/"cli.png"
    subprocess.run([str(cli),"--method","rew","--device","cpu","--max-side","400","--output",str(target),
                    str(ASSETS/"synthetic/view0.png"),str(ASSETS/"synthetic/view1.png")],check=True,capture_output=True)
    r=sk.stitch(pair,sk.Options(method="rew"))
    actual=np.asarray(Image.open(target))
    np.testing.assert_array_equal(actual[...,:3],r.image)
    np.testing.assert_array_equal(actual[...,3],r.mask)


def test_benchmark_contract(pair):
    from stitchkit.benchmark import benchmark_images, reference_metrics
    for bad in (0,True,1.2):
        with pytest.raises(ValueError):benchmark_images(pair,case="x",repeats=bad)
    for bad in (-1,True):
        with pytest.raises(ValueError):benchmark_images(pair,case="x",warmups=bad)
    result=sk.stitch(pair,sk.Options(method="rew"))
    truth=sk.load(ASSETS/"synthetic/pair_truth.png")
    metrics=reference_metrics(result,truth)
    assert metrics["coverage_fraction"] > .985 and metrics["rgb_rmse_255"] < 6
    with pytest.raises(ValueError):reference_metrics(result,truth,border=200)


def test_falsy_options_are_not_silently_defaulted(pair,prepared):
    for value in (False,0,"",{},object()):
        for function,argument in ((sk.stitch,pair),(sk.prepare,pair),(sk.compose,prepared)):
            with pytest.raises(TypeError,match="options"):
                function(argument,value)
