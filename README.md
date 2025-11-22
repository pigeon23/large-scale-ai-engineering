# TODO
1. add distributed checkpoint save and load
2. now dataset only return one text per sample, should concat multiple texts in one sample. But if using iterable dataset, distributedsampler doesn't work because it only supports indexed dataset. Either implement a sampler for distributed training or implement an indexed datset that concats texts.
3. evaluation dataset? split current dataset or simply look for another public dataset?
4. performance evaluation method
5. tp model weight initialization under same random seed



# Start
Run `sbatch script.sbatch`. Remember to adjust environment and log path. 
