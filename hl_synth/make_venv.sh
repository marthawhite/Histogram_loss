virtualenv venv --system-site-packages

source venv/bin/activate

pip install fire

export HDF5_USE_FILE_LOCKING='FALSE'