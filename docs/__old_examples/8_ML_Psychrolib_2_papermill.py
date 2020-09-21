# IMPORTANT: Install papermill from conda-forge, not pip!
# conda install -c conda-forge papermill jupyter_client
import os
from concurrent.futures import ProcessPoolExecutor
import papermill as pm

os.makedirs('outputs', exist_ok=True)

test_size = 9000
train_sizes = [100, 500, 1000]
factors_synthetic = [5, 10, 20, 50]

def run(params):
    out_path = f'outputs/test={params["test_size"]},train={params["train_size"]},syn={params["factor_synthetic"]}.ipynb'
    if os.path.exists(out_path):
        return
    pm.execute_notebook(
        '8_ML_Psychrolib_2.ipynb',
        out_path,
        parameters=params
        )

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as executor:
        for train_size in train_sizes:
            for factor_synthetic in factors_synthetic:
                executor.submit(run, dict(
                    test_size=test_size,
                    train_size=train_size,
                    factor_synthetic=factor_synthetic
                ))
                