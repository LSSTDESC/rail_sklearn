"""
An example classifier that uses catalogue information to
classify objects into tomoragphic bins using random forest.
This is the base method in TXPipe, adapted from TXpipe/binning/random_forest.py
Note: extra dependence on sklearn and input training file.
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.tomographer import CatTomographer, CatInformer
from rail.core.data import TanbleHandle, ModelHandle
from sklearn.ensemble import RandomForestClassifier


class randomForestmodel:
    """
    Temporary class to store the trained model.
    """
    def __init__(self, classifier, features):
        self.classifier = classifier
        self.features = features


class Inform_randomForestClassifier(CatInformer):
    """Train the random forest classifier"""
    
    name = 'Inform_randomForestClassifier'
    config_options = CatInformer.config_options.copy()
    config_options.update(
        bands=Param(tuple, ["r","i","z"], msg="Which bands to use for classification"),
        band_names=param(dict, {"r": "mag_r", "i": "mag_i", "z":"mag_z"}, msg="Band column names"),
        redshift_col=param(str, "sz", msg="Redshift column names"),
        bin_edges=Param(tuple, [0,0.5,1.0], msg="Binning for training data"),
        random_seed=Param(int, msg="random seed"),)
    outputs = [('model', ModelHandle)]
    
    def __init__(self, args, comm=None):
        CatInformer.__init__(self, args, comm=comm)
        
    def run(self):
        # Load the training data
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data('input')

        # Pull out the appropriate columns and combinations of the data
        print(f"Using these bands to train the tomography selector: {self.config.bands}")
        
        # Generate the training data that we will use
        # We record both the name of the column and the data itself
        features = []
        training_data = []
        for b1 in self.config.bands[:]:
            b1_cat=self.config.band_names[b1]
            # First we use the magnitudes themselves
            features.append(b1)
            training_data.append(training_data_table[b1_cat])
            # We also use the colours as training data, even the redundant ones
            for b2 in self.config.bands[:]:
                b2_cat=self.config.band_names[b2]
                if b1 < b2:
                    features.append(f"{b1}-{b2}")
                    training_data.append(training_data_table[b1_cat] - training_data_table[b2_cat])
        training_data = np.array(training_data).T

        print("Training data for bin classifier has shape ", training_data.shape)

        # Now put the training data into redshift bins
        # We use -99 to indicate that we are outside the desired ranges
        z = training_data_table[self.config.redshift_col]
        training_bin = np.repeat(-99, len(z))
        print("Using these bin edges:", self.config.bin_edges)
        for i, zmin in enumerate(self.config.bin_edges[:-1]):
            zmax = self.config.bin_edges[i + 1]
            training_bin[(z > zmin) & (z < zmax)] = i
            ntrain_bin = ((z > zmin) & (z < zmax)).sum()
            print(f"Training set: {ntrain_bin} objects in tomographic bin {i}")

        # Can be replaced with any classifier
        classifier = RandomForestClassifier(
            max_depth=10,
            max_features=None,
            n_estimators=20,
            random_state=self.config.random_seed,
        )
        classifier.fit(training_data, training_bin)

        #return classifier, features
        self.model = randomForestmodel(classifier, features)
        self.add_data('model', self.model)
        

class randomForestClassifier(CatTomographer):
    """Classifier that assigns tomographic 
    bins based on random forest method"""
    
    name = 'randomForestClassifier'
    config_options = CatTomographer.config_options.copy()
    config_options.update(
        bands=Param(tuple, ["r","i","z"], msg="Which bands to use for classification"),
        band_names=param(dict, {"r": "mag_r", "i": "mag_i", "z":"mag_z"}, msg="Band column names"),)
    outputs = [('output', TableHandle)]
    
    def __init__(self, args, comm=None):
        CatTomographer.__init__(self, args, comm=comm)
            
            
    def open_model(self, **kwargs):
        CatTomographer.open_model(self, **kwargs)
        if self.model is None:  # pragma: no cover
            return
        self.classifier = self.model.classifier
        self.features = self.model.features

    
    def run(self):
        """Apply the classifier to the measured magnitudes"""
        
        test_data = self.get_data('input')
        
        data = []
        for f in self.features:
            # may be a single band
            if len(f) == 1:
                f_cat=self.config.band_names[f]
                col = test_data[f_cat]
            # or a colour
            else:
                b1, b2 = f.split("-")
                b1_cat=self.config.band_names[b1]
                b2_cat=self.config.band_names[b2]
                col = (test_data[b1_cat] - test_data[b2_cat])
            if np.all(~np.isfinite(col)):
                # entire column is NaN.  Hopefully this will get deselected elsewhere
                col[:] = 30.0
            else:
                ok = np.isfinite(col)
                col[~ok] = col[ok].max()
            data.append(col)
            data = np.array(data).T

        # Run the random forest on this data chunk
        bin_index = self.classifier.predict(data)

        tomo = {"tomo": bin_index}
        self.add_data('output', tomo)