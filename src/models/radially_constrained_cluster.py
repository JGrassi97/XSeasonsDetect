from random import randint
import numpy as np

from scipy.spatial.distance import cosine
from scipy.spatial.distance import hamming
from scipy.spatial.distance import jaccard, braycurtis


class Radially_Constrained_Cluster(object):

    def __init__(self, data_to_cluster, n_seas, n_iter = 1000, learning_rate = 1, scheduling_factor = 1, min_len = 1, mode = 'single', n_ensemble = 1000, s_factor = 0.1, starting_bp=None, metric='euclidean'):

        '''
            Mandatory parameters:
                -> data to cluster: time series with timesteps on first dimension and features on second
                -> n_seas: number of clusters

            Optional parameters:
                -> n_iter: number of iterations
                -> learning_rate: maximun number of day for stochastic breakpoints upgrade 
                -> scheduling_factor: factor for reducing learning_rate
                -> min_len: minimum length for bounded seasonal length
                -> mode: 'single' for single fit, 'ensemble_stochastic' for ensemble fit with stochastic parameters

            Experimental parameters:
                -> n_ensemble: number of ensemble fit
                -> s_factor: factor for stochastic parameters

        '''

        self.len_serie = np.size(data_to_cluster,axis=0)
        self.data_to_cluster = data_to_cluster
        self.starting_bp = starting_bp

        # Check len_serie-n_seas-min_len consistancy
        if self.len_serie/n_seas < min_len:
            raise ValueError(f'Cannot create {n_seas} season of {min_len} days. Please check your input parameters')
        else:
            self.n_seas = n_seas
            self.min_len = min_len

        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.scheduling_factor = scheduling_factor
        self.mode = mode
        self.s_factor = s_factor
        self.n_ensemble = n_ensemble
        self.metric = metric


    def fit(self):

        '''
            Function for fitting the model and saving the results in the class.
            This functions manages the fitting mode (single vs ensemble) and calls the single fit function
            which contains the core of the algorithm.
        '''

        # Single mode fit: just one fit
        if self.mode == 'single':
            self.breakpoints, self.centroid_history, self.error_history, self.breakpoint_history, self.learningrate_history, self.prediction_history =  self.single_fit()

        ### THE ENSEMBLE MODE IS STILL TO BE DEVELOPED ###



    def single_fit(self):

        # Defining list for metrics saving
        prediction_history = []
        breakpoint_list = []
        centroid_list = []
        error_list = []
        learningrate_list = []
        
        # Main loop
        for j in range(self.n_iter):

            # Generating random starting breakpoints - equally distributed over time (firt iteration)
            if j == 0:

                if self.starting_bp is None:
                    upgrade, b = self.generate_starting_bpoints()
                
                else:
                    upgrade = 0
                    b = self.starting_bp

            # Randomly upgrading breakpoints in the range breakpoint +- learning rate (other iteration)
            else:
                upgrade, b = self.upgrade_breakpoints(b)

            # Generating index for each season
            idx = self.generate_season_idx(b)

            # Control on min season length - if false is skipped
            len_ok = self.check_season_len(idx)

            # Case all season lengths are ok -> computing metrics
            if len_ok == True:

                # Storing current bp
                self.breakpoints = b
                breakpoint_list.append(self.breakpoints)

                self.prediction = self.get_prediction()
                prediction_history.append(self.prediction)

                centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx, self.metric)
                centroid_list.append(centroids)
                error_list.append(np.nanmean(error))
                learningrate_list.append(self.learning_rate)

                # Skipping first iteration
                if j > 0:
                    # Checking if the breakpoints upgrade has improved the metrics
                    if error_list[-1]>error_list[-2]:
                        # If not downgrade breakpoints on last iteration
                        b = downgrade_breakpoints(self.n_seas, b, upgrade, self.len_serie)

                    # Scheduling learning rate for best minimun localization
                    elif (error_list[-1] - error_list[-2]) < 0 and self.scheduling_factor > 1 and self.learning_rate > 1:
                        self.learning_rate = schedule_learning_rate(self.learning_rate, self.scheduling_factor)

            # If there are too short seasons just pretend like nothing happend
            # Downgrading breakpoints to previous iteration
            else:
                b = downgrade_breakpoints(self.n_seas, b, upgrade, self.len_serie)
                idx = self.generate_season_idx(b)
                centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx, self.metric)

        return np.sort(np.int32(b)), np.float64(centroid_list), np.float64(error_list), np.int32(breakpoint_list), np.int32(learningrate_list), np.int32(prediction_history)
    



    def upgrade_breakpoints(self, old_b):

        upgrade = []
        new_b = []

        for k in range(self.n_seas):

            upgrade.append(randint(-self.learning_rate,self.learning_rate))
                
            new_b.append(old_b[k]+upgrade[k])

            if new_b[k]>self.len_serie-1:

                new_b[k]=new_b[k]-self.len_serie-1

            if new_b[k]<0:

                new_b[k]=self.len_serie-1+new_b[k]

        return upgrade, np.array(new_b)

    


    def generate_starting_bpoints(self):

        '''
            Function for generating starting breakpoints.  
        '''

        b_start = []
        upgrade = []

        # Core of breakpoints generation
        for i in range(self.n_seas):

            # If it's the first season 
            if i == 0:

                b_start.append(int((self.len_serie-1)/self.n_seas))
                upgrade.append(0)

            else:
            
                b_start.append(b_start[i-1]+int((self.len_serie-1)/self.n_seas))
                upgrade.append(0)

            if b_start[i] > self.len_serie-1:
                b_start[i] = b_start[i]-self.len_serie-1

        b_start = np.sort(b_start)

        return upgrade, b_start




    def get_prediction(self):

        # Converting breakpoints in a time series 
        prediction = np.zeros((self.len_serie,1))

        idx = self.generate_season_idx(self.breakpoints)

        for i in range(self.n_seas):
            prediction[idx[i]] = i

        return prediction


    def get_final_error(self):

        idx = self.generate_season_idx(self.breakpoints)

        centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx, self.metric)

        return np.sum(error)
    
    
     
    def get_centroids(self):

        idx = self.generate_season_idx(self.breakpoints)

        centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx, self.metric)

        return centroids
        

    def get_index(self):

        idx = self.generate_season_idx(self.breakpoints)

        return idx


    def generate_season_idx(self, b):

        idx = []

        if self.n_seas == 1:
            idx.append(np.arange(0, self.len_serie, 1))

        else:
            for i in np.arange(-1, self.n_seas-1,1):
                if b[i]>b[i+1]:
                    idx_0 = np.arange(b[i], self.len_serie, 1)
                    idx_1 = np.arange(0, b[i+1], 1)
                    idx.append(np.concatenate((idx_0, idx_1), axis=None))

                else:
                    idx.append(np.arange(b[i], b[i+1],1))

        return idx
    


    def check_season_len(self, idx):

        len_ok = True

        for k in range(self.n_seas):

            if len(idx[k])<self.min_len:

                len_ok = False

        return len_ok


def compute_metrics(n_season, data_to_cluster, idx, metric='euclidean', p=2):
    """
    Calcola i centroidi e l'errore per ciascun cluster utilizzando diverse metriche.

    Parameters:
    n_season (int): Numero di cluster.
    data_to_cluster (array): Array di dati da clusterizzare.
    idx (list of arrays): Lista di indici per ciascun cluster.
    metric (str): La metrica da utilizzare per il calcolo dell'errore.
                  Opzioni: 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 
                           'cosine', 'hamming', 'jaccard', 'mahalanobis'
    p (int): Parametro per la distanza di Minkowski (solo se metric='minkowski').
    covariance_matrix (array): Matrice di covarianza per la distanza di Mahalanobis.

    Returns:
    centroids (list): Lista dei centroidi per ciascun cluster.
    error (list): Lista degli errori per ciascun cluster secondo la metrica scelta.
    """
    centroids = []
    error = []

    for i in range(n_season):
        # Otteniamo i dati del cluster corrente
        data_cluster = data_to_cluster[idx[i]]

        # Calcolo del centroide (media ignorando i NaN)
        centroid = np.nanmean(data_cluster, axis=0)
        centroids.append(centroid)
        
        # Calcoliamo l'errore in base alla metrica scelta
        if metric == 'euclidean':
            # Distanza Euclidea
            error.append(np.nansum(np.power(data_cluster - centroid, 2)))

        elif metric == 'manhattan':
            # Distanza di Manhattan (L1)
            error.append(np.nansum(np.abs(data_cluster - centroid)))

        elif metric == 'chebyshev':
            # Distanza di Chebyshev (massima distanza su un asse)
            error.append(np.nanmax(np.abs(data_cluster - centroid)))

        elif metric == 'minkowski':
            # Distanza di Minkowski (generale)
            error.append(np.nansum(np.power(np.abs(data_cluster - centroid), p))**(1/p))

        elif metric == 'cosine':
            # Distanza Coseno (usando il vettore medio come centroide)
            error.append(np.nansum([cosine(row, centroid) for row in data_cluster]))
        
        elif metric == 'braycurtis':
            # Distanza di Bray-Curtis
            error.append(np.nansum([braycurtis(row, centroid) for row in data_cluster]))

        else:
            raise ValueError(f"Metrica non riconosciuta: {metric}")

    return centroids, error





def downgrade_breakpoints(n_season, new_b, upgrade, len_serie):

    old_b = []

    for k in range(n_season):

        old_b.append(new_b[k]-upgrade[k])

        if old_b[k]>len_serie-1:

            old_b[k]=old_b[k]-len_serie-1

        if old_b[k]<0:

            old_b[k]=len_serie-1+old_b[k]

    return np.array(old_b)



def schedule_learning_rate(learning_rate, scheduling_factor):

    return np.int32(learning_rate/scheduling_factor)





