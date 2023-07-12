from common import *

error = rmse_mae


# This means the results will be saved in res/evaluation
res_dir = os.path.join('old-experiment','res')

# Data
dataset = 'ctscan'

# training/test split
test_ratio = 0.2

dat, y_min, y_max = get_data(dataset)

# Hyperparameters
best_params = {}
best_params['lr'] = 1e-3 # learning rate
best_params['h_l'] = 4 # number of hidden layers
best_params['dropout_rate'] = 0.05 # dropout rate
best_params['n_bins'] = 100 # number of bins
best_params['ker_par'] = 1.0 # standard deviation of the Gaussian target distribution
best_params['batch_size'] = 256 # batch size

num_runs = 1
epochs = 100 # Change this to a small number like 10 to see interim performance.
verbose = 0 # Change this to 1 to print all training details
seeds = range(num_runs) # random seeds

meta_data = {}
meta_data['best_params'] = best_params
meta_data['num_runs'] = num_runs
meta_data['test_ratio'] = test_ratio

# Linear regression is a linear model trained with squared error
# Regression is a neural network trained with squared error
# Histogram-Gaussian is a neural network trained with the Histogram Loss and Gaussian target distribution
keys = ['Linear Regression', 'Regression', 'Histogram-Gaussian']

# Dictionary for storing learning curves
history = {}

# Dictionary for storing results
results = {}


for key in keys:
    if key not in history:
        history[key] = []
    if key not in results:
        results[key] = []



for run in range(num_runs):
    print('****run:', run)

    X_tv, X_test, y_tv, y_test = get_split(dat, dataset, seed=seeds[run])

    print('Data shape:', dat.shape)
    print('Using these splits:', X_tv.shape, y_tv.shape, X_test.shape, y_test.shape, 'mean of first train point:', X_tv[0,:].mean())
    print('---')

    np.random.seed(seeds[run])
    tf.random.set_seed(seeds[run])

    key = 'Linear Regression'

    if key in keys:
        print(key)
        lreg = LinearRegression(n_jobs=-1)
        lreg.fit(X_tv, y_tv)
        y_pred_tr = lreg.predict(X_tv)
        train_error = error(y_tv, y_pred_tr)
        print('train error (RMSE and MAE):', train_error)
        y_pred_test = lreg.predict(X_test)
        test_error = error(y_test, y_pred_test)
        print('test error (RMSE and MAE):', test_error)
        print('---')
        results[key].append((train_error, test_error))


    key = 'Regression'

    if key in keys:
        y_tv_norm = (1.0*y_tv - y_min)/(y_max - y_min)
        y_test_norm = (1.0*y_test - y_min)/(y_max - y_min)

        np.random.seed(seeds[run])
        tf.random.set_seed(seeds[run])

        # Training the baseline NN
        print(key)
        reg_model = create_baseline_nn_model(best_params['h_l'], best_params['lr'], input_dim=X_tv.shape[1], dropout_rate=best_params['dropout_rate'])
        history[key].append(reg_model.fit(X_tv, y_tv_norm, validation_data=(X_test, y_test_norm), epochs=epochs, batch_size=best_params['batch_size'], verbose=verbose).history)
        y_pred_tr = reg_model.predict(X_tv).flatten() * (y_max - y_min) + y_min
        train_error = error(y_tv, y_pred_tr)
        print('train error (RMSE and MAE):', train_error)
        y_pred_test = reg_model.predict(X_test).flatten() * (y_max - y_min) + y_min
        test_error = error(y_test, y_pred_test)
        print('test error (RMSE and MAE):', test_error)
        print('---')
        results[key].append((train_error, test_error))

    key = 'Histogram-Gaussian'

    if key in keys:
        y_tv_dist, y_test_dist, centers = transform_normal(y_tv, y_test, y_min, y_max, n_bins=best_params['n_bins'], ker_par_ratio=best_params['ker_par'])

        np.random.seed(seeds[run])
        tf.random.set_seed(seeds[run])

        # Training the new model
        print(key)
        cat_model = create_main_model(best_params['h_l'], best_params['lr'], best_params['n_bins'], input_dim=X_tv.shape[1], dropout_rate=best_params['dropout_rate'])
        history[key].append(cat_model.fit(X_tv, y_tv_dist, validation_data=(X_test, y_test_dist), epochs=epochs, batch_size=best_params['batch_size'],verbose=verbose).history)

        y_pred_tr = (centers[np.newaxis,:] * cat_model.predict(X_tv)).sum(axis=1)
        train_error = error(y_tv, y_pred_tr)
        print('train error (RMSE and MAE):', train_error)
        y_pred_test = (centers[np.newaxis,:] * cat_model.predict(X_test)).sum(axis=1)
        test_error = error(y_test, y_pred_test)
        print('test error (RMSE and MAE):', test_error)
        print('---')
        results[key].append((train_error, test_error))

print(results)

with open(os.path.join(res_dir, 'history_'+dataset+'.json'), 'w') as f:
    json.dump(history, f)

with open(os.path.join(res_dir, 'res_'+dataset+'.json'), 'w') as f:
    json.dump(results, f)

with open(os.path.join(res_dir, 'meta_data_'+dataset+'_.json'), 'w') as f:
    json.dump(meta_data, f)

# with open(os.path.join(res_dir, 'history_'+dataset+'.pkl'), 'wb') as f:
#     pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

# with open(os.path.join(res_dir, 'res_'+dataset+'.pkl'), 'wb') as f:
#     pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

# with open(os.path.join(res_dir, 'meta_data_'+dataset+'_.pkl'), 'wb') as f:
#     pickle.dump(meta_data, f, pickle.HIGHEST_PROTOCOL)