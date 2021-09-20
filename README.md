# BeyondTracking

This repository contains the codes used for the pre-printed paper at \[[arXiv:2108.09394](https://arxiv.org/abs/2108.09394)\]: 

***Beyond Tracking: Using Deep Learning to Discover Novel Interactions in Biological Swarms***,
presented at the joint conference [DARS-SWARM2021](https://www.swarm-systems.com/), winning the [*Best Paper Award* at SWARM2021](https://www.swarm-systems.com/dars-swarm2021/awards#h.a6edhgurno03). 

<img src=imgs/scenario.jpg width="80%" class="center">

Basically, this BeyondTracking (BT) framework has been built to *visually* highlight local behaviors in complex multi-agent systems (e.g., ant colony) that a system-level state detector considers *informative* for inference of state transitions in global system. The ultimate goal of the method is set to inform humans of *unknown* individual behaviors to better understand complex social systems.

In the paper, therefore, BT is built upon [***IO-GEN***](https://github.com/ctyeong/IO-GEN), designed to detect abnormal states of ant colonies, while the model could only access observations of normal ant colonies. In particular, each input to IO-GEN is set as a sequence of video frames from the entire view of the focal system without spatial tracking annotations. So, IO-GEN is trained to utilise the global-view observations to detect abnormal ant colonies. 

BT can then be applied to discover specific behaviors (or ant entities) that indicate the abnormal global states. Technically, [Grad-CAM](https://ieeexplore.ieee.org/document/8237336) is an essential component in BT to visualise local motions of ants from sequential frame images, but as stated in the archived paper, only highly prioritised regions are displayed in BT by thresholding to only localise highly impactful individuals. 

# Contents

1. [Examples](https://github.com/ctyeong/BeyondTracking#examples)
2. [Installation](https://github.com/ctyeong/BeyondTracking#installation)
3. [Training](https://github.com/ctyeong/BeyondTracking#training)
4. [Test](https://github.com/ctyeong/BeyondTracking#test)
5. [Citation](https://github.com/ctyeong/BeyondTracking#citation)
6. [Contact](https://github.com/ctyeong/BeyondTracking#contact)


# Examples 



# Installation 

1. Clone the repository
    ```
    git clone https://github.com/ctyeong/IO-GEN.git
    ```

2. Install the required Python packages

    ```
    pip install -r requirements.txt
    ```
   - Python 3.6 is assumed to be installed already



# Training 



# Test 



# Citation 

```
@article{CPLP21B,
  title={Beyond Tracking: Using Deep Learning to Discover Novel Interactions in Biological Swarms},
  author={Choi, Taeyeong and Pyenson, Benjamin and Liebig, Juergen and Pavlic, Theodore P},
  journal={arXiv preprint arXiv:2108.09394},
  year={2021}
}
```


# Contact

If there is any question or suggestion, please do not hesitate to drop an email to tchoi@lincoln.ac.uk. Thanks!