from random import randint
import numpy as np

from scipy.spatial.distance import cosine, braycurtis
from scipy.stats import entropy

from itertools import combinations
from tqdm import tqdm
import numpy as np



def generate_season_idx(b, len_serie, n_seas):
    """
    Genera gli indici per ciascuna stagione (cluster) basati sui breakpoint.
    
    Parameters:
    b (list): Lista dei breakpoint.
    len_serie (int): Lunghezza della serie temporale.
    n_seas (int): Numero di stagioni (cluster).
    
    Returns:
    list: Lista di array di indici per ogni stagione.
    """
    idx = []
    if n_seas == 1:
        idx.append(np.arange(0, len_serie, 1))
    else:
        for i in np.arange(-1, n_seas-1, 1):
            if b[i] > b[i+1]:
                idx_0 = np.arange(b[i], len_serie, 1)
                idx_1 = np.arange(0, b[i+1], 1)
                idx.append(np.concatenate((idx_0, idx_1), axis=None))
            else:
                idx.append(np.arange(b[i], b[i+1], 1))
    return idx

def check_season_len(idx, n_seas, min_len=1):
    """
    Controlla se la lunghezza di ogni stagione rispetta la lunghezza minima.

    Parameters:
    idx (list): Lista di indici delle stagioni.
    n_seas (int): Numero di stagioni.
    min_len (int): Lunghezza minima per ogni stagione.
    
    Returns:
    bool: True se tutte le stagioni rispettano la lunghezza minima, altrimenti False.
    """
    return all(len(idx[k]) >= min_len for k in range(n_seas))


def crps_vectorized(x):
    N = len(x)
    
    # Calcoliamo la differenza assoluta tra ogni coppia di elementi di x (broadcasting)
    diffs = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
    
    # Calcoliamo la media delle differenze per ogni elemento (axis=1)
    crps_values = np.mean(diffs, axis=1)
    
    return crps_values

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

        elif metric == 'crps':
            # Calcolo del CRPS (Continuous Ranked Probability Score)
            error.append(np.nansum(crps_vectorized(data_cluster)))

        elif metric == 'braycurtis':
            # Distanza di Bray-Curtis
            error.append(np.nansum([braycurtis(row, centroid) for row in data_cluster]))

        elif metric == 'jensenshannon':
            # Distanza di Jensen-Shannon (simmetrica, basata su distribuzioni di probabilità)
            m = 0.5 * (data_cluster + centroid)
            error.append(np.nansum([entropy(row, m) + entropy(centroid, m) for row in data_cluster]) / 2)

        else:
            raise ValueError(f"Metrica non riconosciuta: {metric}")

    return centroids, error


def single_fit_optimized(data_to_cluster, n_seas, n_days, metric='euclidean'):
    """
    Ottimizza la selezione dei breakpoint cercando tra le combinazioni possibili
    quella che minimizza l'errore totale.

    Parameters:
    data_to_cluster (ndarray): Time series data (timesteps x features).
    n_seas (int): Numero di stagioni (cluster).
    n_days (list): Lista dei giorni considerati per i breakpoints.

    Returns:
    tuple: Migliori breakpoints, lista degli errori, lista di breakpoints, storici delle predizioni.
    """
    best_error = float('inf')
    error_list = []
    breakpoint_list = []
    prediction_history = []
    best_combination = None
    
    # Per ogni combinazione dei breakpoint calcola l'errore totale
    for combination in tqdm(combinations(n_days, n_seas)):
        b = list(combination)
        breakpoint_list.append(b)
        
        # Genera gli indici delle stagioni basati sui breakpoint
        idx = generate_season_idx(b, len(data_to_cluster), n_seas)
        prediction_history.append(idx)
        
        if check_season_len(idx, n_seas):
            centroids, error = compute_metrics(n_seas, data_to_cluster, idx, metric)
            total_error = np.nanmean(error)
            error_list.append(total_error)
            
            # Aggiorna se si trova un errore migliore
            if total_error < best_error:
                best_error = total_error
                best_combination = b

    return np.sort(np.int32(best_combination)), np.float64(error_list), np.int32(breakpoint_list)


