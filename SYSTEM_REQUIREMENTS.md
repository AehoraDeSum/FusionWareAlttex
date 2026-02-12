# System Requirements for Quantum Simulation

## Estimated Requirements for 10,000 Samples

### Minimum Requirements
- **CPU**: 4 cores, 2.0 GHz or higher
- **RAM**: 8 GB
- **Storage**: 500 MB free space
- **Estimated Time**: 4-8 hours (depending on CPU)

### Recommended Requirements
- **CPU**: 8+ cores, 3.0 GHz or higher (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16 GB
- **Storage**: 1 GB free space
- **Estimated Time**: 1-3 hours

### Optimal Requirements
- **CPU**: 16+ cores, 3.5+ GHz (Intel Xeon/AMD Threadripper or better)
- **RAM**: 32 GB
- **Storage**: 2 GB free space
- **Estimated Time**: 30 minutes - 1 hour

## Computational Complexity

Each sample requires:
- **Time steps**: 1,000 per sample (default)
- **Quantum system**: 2-level atom (2x2 Hilbert space - relatively small)
- **Solver**: QuTiP's `mesolve` (master equation solver)
- **Time-dependent Hamiltonian**: Requires interpolation at each time step

**Total operations**: 10,000 samples × 1,000 time steps = 10,000,000 quantum evolution steps

## Performance Optimization Options

1. **Reduce time steps** (faster, less resolution):
   ```bash
   python3 stage1_simulation/generate_dataset.py \
       --num_samples 10000 \
       --time_steps 500 \
       --output data/synthetic_dataset.h5
   ```

2. **Use parallel processing** (see optimized version below)

3. **Generate in batches** and combine:
   ```bash
   # Generate 4 batches of 2500 samples each
   python3 stage1_simulation/generate_dataset.py --num_samples 2500 --output data/batch1.h5
   python3 stage1_simulation/generate_dataset.py --num_samples 2500 --output data/batch2.h5
   # ... etc
   ```

4. **Start with smaller dataset** for testing:
   ```bash
   python3 stage1_simulation/generate_dataset.py --num_samples 1000 --output data/test_dataset.h5
   ```

## Memory Usage

- **Per sample**: ~10-50 KB (depends on time steps)
- **10,000 samples**: ~100-500 MB in memory
- **HDF5 file size**: ~50-200 MB (compressed)

## Tips

1. **Close other applications** to free up CPU and RAM
2. **Run overnight** for large datasets
3. **Monitor progress** - the script shows a progress bar
4. **Use SSD** for faster I/O when saving large datasets
5. **Consider cloud computing** (AWS, Google Cloud) for very large datasets
