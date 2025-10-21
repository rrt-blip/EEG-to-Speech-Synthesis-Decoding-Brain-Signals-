from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt

# Load NWB file
nwb_path = "../data/raw/SingleWordProductionDutch-iBIDS/sub-01/ieeg/sub-01_task-wordProduction_ieeg.nwb"
io = NWBHDF5IO(nwb_path, 'r')
nwbfile = io.read()

# Access iEEG data
ieeg_data = nwbfile.acquisition['iEEG']
data = ieeg_data.data[:]  # shape: (timepoints, channels)

# Use timestamps stored in the NWB file
times = ieeg_data.timestamps[:]  # shape: (timepoints,)

# Plot first channel
plt.figure(figsize=(12, 4))
plt.plot(times, data[:, 0])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title("iEEG - Channel 0")
plt.show()


