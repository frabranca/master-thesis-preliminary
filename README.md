# Master Thesis: Optical Flow Determination using Integrate & Fire Neurons
**Project Task:** investigating on using the SynSense's [speck2e](https://www.synsense.ai/products/speck-2/) neuromorphic device for optical flow determination.
## Details
This repository contains the preliminary experiments of my master thesis project, specifically, the testing of the speck2e. To get familiar with Samna and Sinabs (libraries developed by SynSense), the speck is tested on the Neuromorphic MNIST dataset (N-MNIST). Other scripts are included to analyse the undergoing processes inside the device. For instance:

- read the membrane potential of the neurons and its decay (``speck_membrane_potential.py``, ``speck_bias_test.py``);
- visualize the output of the event-based camera (``event_based_video_tools/visualizer.py``);
- observe the output spikes from the device (``speck_events_read.py``).
- investigate on using concatenation recurrency inside the layers (``concatenation_config.py``).

In addition, the repository includes some other tools developed to work with h5 datasets and generate event-based videos.

## Resources
- Samna ([documentation](https://synsense-sys-int.gitlab.io/samna/index.html))
- sinabs-dynapcnn ([documenation](https://synsense-sys-int.gitlab.io/samna/index.html) and [repo](https://gitlab.com/synsense/sinabs-dynapcnn))
