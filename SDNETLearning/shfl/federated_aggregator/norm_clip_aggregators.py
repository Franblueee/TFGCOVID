import numpy as np
from numpy import linalg as LA

from shfl.federated_aggregator import FedAvgAggregator
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic


class NormClipAggregator(FedAvgAggregator):
    """
    Implementation of Average Federated Aggregator with Clipped Norm.
    It clips the norm of the client's updates and averages them.

    # Arguments:
        clip: value used to clip each client's update
    """

    def __init__(self, clip):
        self._clip = clip

    def _serialize(self, data):
        """
        It turns a list of multidimensional arrays into a list of one-dimensional arrays

        # Arguments:
            data: list of multidimensional arrays
        """
        data = [np.array(j) for j in data]
        self._data_shape_list = [j.shape for j in data]
        serialized_data = [j.ravel() for j in data]
        serialized_data = np.hstack(serialized_data)
        return serialized_data
        
    def _deserialize(self, data):
        """
        It turns a list of one-dimensional arrays into a list of multidimensional arrays.
        The multidimensional shape is stored when it is serialized

        # Arguments:
            data: list of one-dimensional arrays
        """

        firstInd = 0
        deserialized_data = []
        for shp in self._data_shape_list:
            if len(shp) > 1:
                shift = np.prod(shp)
            elif len(shp) == 0:
                shift = 1
            else:
                shift = shp[0]
            tmp_array = data[firstInd:firstInd+shift]
            tmp_array = tmp_array.reshape(shp)
            deserialized_data.append(tmp_array)
            firstInd += shift
        return deserialized_data

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Aggregation of arrays"""
        clients_params = np.array(params)
        for i, v in enumerate(clients_params):
            norm = LA.norm(v)
            clients_params[i] = np.multiply(v, min(1, self._clip/norm))
        
        return np.mean(clients_params, axis=0)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Aggregation of (nested) lists of arrays"""        
        serialized_params = np.array([self._serialize(client) for client in params])
        serialized_aggregation = self._aggregate(*serialized_params)
        aggregated_weights = self._deserialize(serialized_aggregation)
        
        return aggregated_weights


class CDPAggregator(NormClipAggregator):
    """
    Implementation of Average Federated Aggregator with Differential Privacy also known \
    as Central Differential Privacy.
    It clips the norm of the client's updates, averages them and adds gaussian noise \
    calibrated to noise_mult*clip/number_of_clients.

    # Arguments:
        clip: value used to clip each client's update.
        noise_mult: quantity of noise to add. To ensure proper Differential Privacy, \
        it must be calibrated according to some composition theorem.
    """

    def __init__(self, clip, noise_mult):
        super().__init__(clip=clip)
        self._noise_mult = noise_mult
    
    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Aggregation of arrays with gaussian noise calibrated to noise_mult*clip/number_of_clients"""
        clients_params = np.array(params)
        mean = super()._aggregate(*params)
        noise = np.random.normal(loc=0.0, scale=self._noise_mult*self._clip/len(clients_params), size=mean.shape) 
        return mean + noise

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        return super()._aggregate(*params)


class WeakDPAggregator(CDPAggregator):
    """
    Implementation of Average Federated Aggregator with Weak Differential Privacy.
    It clips the norm of the client's updates, averages them and adds gaussian noise \
    calibrated to 0.025*clip/number_of_clients.
    The noise multiplier 0.025 is not big enough to ensure proper Differential Privacy.

    # Arguments:
        clip: value used to clip each client's update
    """
    def __init__(self, clip, noise_mult=0.025):
        super().__init__(clip=clip, noise_mult = noise_mult)

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        return super()._aggregate(*params)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        return super()._aggregate(*params)
