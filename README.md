### run localy from vscode
open folder /Users/hagitbenshoshan/Documents/PHD/Temporal-LPA-1 
under notebooks run "new.ipynb" 
This folder is synced with : https://github.com/hagitbs/Temporal-LPA
Python 3.7
### Installation

Run `pip install -r requirements.txt`. Make sure your Python version is up to date. 


### Data Exploration

A short exploartion of the data can be dound in [LOCO_data_exploration.ipynb](LOCO_data_exploration.ipynb).


### TLPA
To run the basic TLPA, run `python LOCO_TLPA.py`. The script does the following:
1. Function `create_freq` joins all noun-phrase frequency files in `data/loco/np_freq` to one main dataframe, and adds the metadata, so only data with from after 1990 is included, and only months with over 20 articles are included. Writes the filtered frecuency data to `results/base_freq.csv`
2. For each subcorpus (mainstream / conspiracy):
   1. Function `tw_freq` groups frequencies per time-window (the only available time-window currently is per month), for a given start and end date. Writes the grouped frecuencies to `results/{subcorpus}/tw_freq.csv`
   2. Function `dBTC` preforms δ-bounded timeline compression on the frequencies, using the Kullback-Leibler divergence and with the threshold being the median of the data. Writes a bar chart visualization to `results/{subcorpus}/bar_charts` for each step, writes a dataframe of the final distances between time-windows to `results/{subcorpus}/kldf.csv`, and the compressed frequency data to `results/{subcorpus}/squeezed_freq.csv`.
   3. Within the function `create_and_cut`:
      1.  The function `create_mdvr` creates a mean DVR by performing a monthly average for each element in the corpus.
      2.  Compressed frequencies are converted to vector format using the `create_arrays` function from the `LPA` module.
      3.  KLD distances are measured between vectors and DVR using the `create_distances` function from the `LPA` module, and a ± sign is added accordingly.
      4.  Returns a list of elements with the highest absolute distances, along with a list of signatures.
   4. Finally writes mDVR to `results/{subcorpus}/dvr.csv`, signatures to `results/{subcorpus}/sigs/sig_{timestamp}.csv`, and highest absolute distances to `results/{subcorpus}/max_distances.csv`

### Visualize results
Using a couple functions from `visualize.py`, go to [LOCO_results.ipynb](LOCO_results.ipynb) to view timelines of frequency data and KLD distances over time.