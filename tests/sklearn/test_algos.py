import numpy as np
import os
import pytest
import scipy.special

from rail.utils.testing_utils import one_algo
from rail.core.stage import RailStage
from rail.utils.path_utils import RAILDIR
from rail.core.data import TableHandle
from rail.estimation.algos import k_nearneigh, sklearn_neurnet, random_forest

sci_ver_str = scipy.__version__.split(".")


DS = RailStage.data_store
DS.__class__.allow_overwrite = True

def test_simple_nn():
    train_config_dict = {
        "width": 0.025,
        "zmin": 0.0,
        "zmax": 3.0,
        "nzbins": 301,
        "max_iter": 250,
        "hdf5_groupname": "photometry",
        "model": "model.tmp",
    }
    estim_config_dict = {"hdf5_groupname": "photometry", "model": "model.tmp"}
    # zb_expected = np.array([0.152, 0.135, 0.109, 0.158, 0.113, 0.176, 0.13 , 0.15 , 0.119, 0.133])
    train_algo = sklearn_neurnet.SklNeurNetInformer
    pz_algo = sklearn_neurnet.SklNeurNetEstimator
    results, rerun_results, rerun3_results = one_algo(
        "SimpleNN", train_algo, pz_algo, train_config_dict, estim_config_dict
    )
    # assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"]).all()



@pytest.mark.skipif(
    int(sci_ver_str[0]) < 2 and int(sci_ver_str[1]) < 8,
    reason="mixmod parameterization known to break for scipy<1.8 due to array broadcast change",
)
def test_KNearNeigh():
    def_bands = ["u", "g", "r", "i", "z", "y"]
    refcols = [f"mag_{band}_lsst" for band in def_bands]
    def_maglims = dict(
        mag_u_lsst=27.79,
        mag_g_lsst=29.04,
        mag_r_lsst=29.06,
        mag_i_lsst=28.62,
        mag_z_lsst=27.98,
        mag_y_lsst=27.05,
    )
    train_config_dict = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        trainfrac=0.75,
        random_seed=87,
        ref_column_name="mag_i_lsst",
        column_names=refcols,
        mag_limits=def_maglims,
        sigma_grid_min=0.02,
        sigma_grid_max=0.03,
        ngrid_sigma=2,
        leaf_size=2,
        nneigh_min=2,
        nneigh_max=3,
        redshift_column_name="redshift",
        hdf5_groupname="photometry",
        model="KNearNeighEstimator.pkl",
    )
    estim_config_dict = dict(hdf5_groupname="photometry", model="KNearNeighEstimator.pkl")

    # zb_expected = np.array([0.13, 0.14, 0.13, 0.13, 0.11, 0.15, 0.13, 0.14,
    #                         0.11, 0.12])
    train_algo = k_nearneigh.KNearNeighInformer
    pz_algo = k_nearneigh.KNearNeighEstimator
    results, rerun_results, rerun3_results = one_algo(
        "KNN", train_algo, pz_algo, train_config_dict, estim_config_dict
    )
    # assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"]).all()

# test for k=1 when data point has same value, used to cause errors because of
# a divide by zero, should be fixed now but add a test
def test_same_data_knn():
    train_config_dict = dict(hdf5_groupname="photometry",
                             model="KNearNeighEstimator.pkl")
    estim_config_dict = dict(hdf5_groupname="photometry",
                             model="KNearNeighEstimator.pkl")

    traindata = os.path.join(RAILDIR, 'rail/examples_data/testdata/training_100gal.hdf5')
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    training_data = DS.read_file('training_data', TableHandle, traindata)
    trainer = k_nearneigh.KNearNeighInformer.make_stage(name='same_train', **train_config_dict)
    trainer.inform(training_data)
    pz = k_nearneigh.KNearNeighEstimator.make_stage(name='same_estim', **estim_config_dict)
    estim = pz.estimate(training_data)  # run estimate on same input file
    modes = estim().ancil['zmode']
    assert ~(np.isnan(modes).all())
    os.remove(pz.get_output(pz.get_aliased_tag('output'), final_name=True))

    
def test_catch_bad_bands():
    params = dict(bands="u,g,r,i,z,y")
    with pytest.raises(ValueError):
        sklearn_neurnet.SklNeurNetInformer.make_stage(hdf5_groupname="", **params)
    with pytest.raises(ValueError):
        sklearn_neurnet.SklNeurNetEstimator.make_stage(hdf5_groupname="", **params)
        

def test_randomForestClassifier():
    class_bands = [ "r","i","z"]
    bands = {"r": "mag_r_lsst", "i": "mag_i_lsst", "z": "mag_z_lsst"}
    bin_edges=[0,0.2,0.5]
    
    train_config_dict=dict(
        class_bands=class_bands,
        bands=bands,
        redshift_col="redshift",
        bin_edges=bin_edges,
        random_seed=10,
        hdf5_groupname="photometry",
        model="model.tmp",
    )
    
    estim_config_dict=dict(hdf5_groupname="photometry", model="model.tmp", id_name="")
    
    train_algo = random_forest.RandomForestInformer
    tomo_algo = random_forest.RandomForestClassifier
    results, rerun_results, rerun3_results = one_algo(
        "randomForestClassifier", train_algo, tomo_algo, train_config_dict, estim_config_dict,
        is_classifier=True,
    )
    assert np.isclose(results["data"]["class_id"], rerun_results["data"]["class_id"]).all()
    assert len(results["data"]["class_id"])==len(results["data"]["row_index"])


def test_randomForestClassifier_id():
    class_bands = [ "r","i","z"]
    bands = {"r": "mag_r_lsst", "i": "mag_i_lsst", "z": "mag_z_lsst"}
    bin_edges=[0,0.2,0.5]
    
    train_config_dict=dict(
        class_bands=class_bands,
        bands=bands,
        redshift_col="redshift",
        bin_edges=bin_edges,
        random_seed=10,
        hdf5_groupname="photometry",
        model="model.tmp",
    )
    estim_config_dict=dict(hdf5_groupname="photometry", model="model.tmp", id_name="id")
    
    train_algo = random_forest.RandomForestInformer
    tomo_algo = random_forest.RandomForestClassifier
    
    traindata = os.path.join(RAILDIR, 'rail/examples_data/testdata/training_100gal.hdf5')
    validdata = os.path.join(RAILDIR, 'rail/examples_data/testdata/validation_10gal.hdf5')
    
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()
    training_data = DS.read_file('training_data', TableHandle, traindata)
    validation_data = DS.read_file('validation_data', TableHandle, validdata)
    
    train_pz = train_algo.make_stage(**train_config_dict)
    train_pz.inform(training_data)
    pz = tomo_algo.make_stage(name="randomForestClassifier", **estim_config_dict)
    estim = pz.classify(training_data)
    results=estim.data
    
    os.remove(pz.get_output(pz.get_aliased_tag('output'), final_name=True))
    model_file = estim_config_dict.get('model', 'None')
    if model_file != 'None':
        try:
            os.remove(model_file)
        except FileNotFoundError:  #pragma: no cover
            pass
    
    assert len(results["data"]["class_id"])==len(results["data"]["id"])