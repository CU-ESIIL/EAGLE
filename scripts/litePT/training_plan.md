I'm working on training representation models for aerial LiDAR. Input is USGS 3DEP via entwine and pdal

The training procedure can follow LitePT codebase. 

Eventually we will evaluate the representations on downstream task: classification of  on the vegbank dataset for ALS tiles. supervised single-target classification task of predicting the habitat type at the location of the vegetation survey from a lidar tile. I'll let you know when the train and validation datasets are ready

Please design and implement a training script/procedure using the LitePT architecture https://github.com/prs-eth/LitePT . 

I've implemented a dataset (dataset.py) and investigated parallel dataloader ingestion of the USGS point cloud tiles: test_streaming_speed.ipynb. You are welcome to implement alternative dataloading strategies to enhance data ingestion speeds and of course to implement augmentation routines. It may be best to cache medium-sized pools of pdal-aquired lidar tiles locally and re-use those tiles several times instead of always downloading new tiles for each training step. 

Some foundational experiments will be necessary: 
- setting up and testing use of the architecture, dataset/dataloader, loss fn, including benchmarking memory usage for batch size optimization; see LitePT/blob/main/engines/train.py
- checking for missing/unavailable/corrupted lidar sources and creating a version of the train/eval datasets that excludes missing data 
- script or notebook that generates some visuals of labeled training samples, so that I can visually check that the labels sensibly match the data
- setting up data augmentation routines and providing images that visualize the effect of each augmentation (augmentations and data preprocessing operations are defined in litept/datasets/transform.py)


Training will be conducted on the current computing cluster, where slurm jobs are used to run on GPU nodes of up to 8 GPUS each (cpus are limited to 5 per gpu). You can set up test and full runs, and run short jobs eg up to 6 hrs on 4 gpus, for longer jobs you should provide the slurm script and ask me to launch them. The current interactive job only has cpus and limited memory. 


Evaluation on held-out validation set will report classification accuracy. 