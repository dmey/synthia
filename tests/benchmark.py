# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

import time
import numpy as np

import synthia as syn

def main():
    import pyvinecopulib as pv

    n_samples = 100
    n_features = 50
    n_synthetic_samples = 500
    input_data = np.random.normal(size=(n_samples, n_features))

    generator = syn.CopulaDataGenerator()
    
    t0 = time.time()
    generator.fit(input_data, syn.GaussianCopula())
    generator.generate(n_synthetic_samples)
    print(f'synthia: {time.time() - t0}s')

    t0 = time.time()
    generator.fit(input_data, syn.VineCopula(pv.FitControlsVinecop(
        family_set=[pv.gaussian], trunc_lvl=1, select_trunc_lvl=False)))
    generator.generate(n_synthetic_samples)
    print(f'pyvinecopulib: {time.time() - t0}s')

if __name__ == '__main__':
    main()
