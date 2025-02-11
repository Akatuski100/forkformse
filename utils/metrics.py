import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe

import numpy as np

def channel_MAE(pred, true):
    """
    Compute Mean Absolute Error (MAE) for each channel.
    pred, true: numpy arrays of shape (N, T, C)
    Returns:
      mae_per_channel: numpy array of shape (C,)
    """
    return np.mean(np.abs(pred - true), axis=(0, 1))

def channel_MSE(pred, true):
    """
    Compute Mean Squared Error (MSE) for each channel.
    pred, true: numpy arrays of shape (N, T, C)
    Returns:
      mse_per_channel: numpy array of shape (C,)
    """
    return np.mean((pred - true) ** 2, axis=(0, 1))

def channel_RMSE(pred, true):
    """
    Compute Root Mean Squared Error (RMSE) for each channel.
    """
    return np.sqrt(channel_MSE(pred, true))

def channel_MAPE(pred, true):
    """
    Compute Mean Absolute Percentage Error (MAPE) for each channel.
    """
    return np.mean(np.abs((pred - true) / true), axis=(0, 1))

def channel_MSPE(pred, true):
    """
    Compute Mean Squared Percentage Error (MSPE) for each channel.
    """
    return np.mean(np.square((pred - true) / true), axis=(0, 1))

def metric_per_channel(pred, true):
    """
    Given prediction and true arrays of shape (N, T, C),
    returns per-channel metrics: MAE, MSE, RMSE, MAPE, MSPE.
    """
    cmae = channel_MAE(pred, true)
    cmse = channel_MSE(pred, true)
    crmse = channel_RMSE(pred, true)
    cmape = channel_MAPE(pred, true)
    cmspe = channel_MSPE(pred, true)
    return cmae, cmse, crmse, cmape, cmspe
