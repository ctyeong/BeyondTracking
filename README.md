# Beyond Tracking

This repository contains the codes used for the pre-printed paper at \[[arXiv:2108.09394](https://arxiv.org/abs/2108.09394)\]: 

***"Beyond Tracking: Using Deep Learning to Discover Novel Interactions in Biological Swarms"***,
presented at the joint conference [DARS-SWARM2021](https://www.swarm-systems.com/), winning the [*Best Paper Award* at SWARM2021](https://www.swarm-systems.com/dars-swarm2021/awards#h.a6edhgurno03). 

<img src=imgs/scenario.jpg width="80%">

Basically, this BeyondTracking (BT) framework has been built to *visually* highlight local behaviors in complex multi-agent systems (e.g., ant colony) that a system-level state detector considers *informative* for inference of state transitions in global system. The ultimate goal of the method is set to inform humans of *unknown* individual behaviors to better understand complex social systems.

In the paper, therefore, BT is built upon two key components, publicly available online: 
1. [***IO-GEN***](https://github.com/ctyeong/IO-GEN)
2. [***Video data of Harpegnathos saltator ants in optical flow form***](https://github.com/ctyeong/OpticalFlows_HsAnts)
   
To be specific, IO-GEN was designed to detect abnormal states of ant colonies, while the model could only access observations of normal ant colonies. In particular, the video data of *Harpegnathos saltator* ants are used as input to IO-GEN, as each input is a sequence of *optical flow* frames from the entire view of the focal system without spatial tracking annotations. So, IO-GEN is trained to utilise the global-view observations to detect abnormal ant colonies. 

BT can then be applied to discover specific behaviors (or ant entities) that indicate the abnormal global states. Technically, [Grad-CAM](https://ieeexplore.ieee.org/document/8237336) is an essential component in BT to visualise local motions of ants from sequential frame images, but as stated in the archived paper, only highly prioritised regions are displayed in BT by thresholding to only localise highly impactful individuals. 

# Contents

1. [Expected Output](https://github.com/ctyeong/BeyondTracking#expected-output)
2. [Installation](https://github.com/ctyeong/BeyondTracking#installation)
4. [Execution](https://github.com/ctyeong/BeyondTracking#execution)
5. [Citation](https://github.com/ctyeong/BeyondTracking#citation)
6. [Contact](https://github.com/ctyeong/BeyondTracking#contact)


# Expected Output 

<img src=imgs/dueling.jpg width="70%">

Validations have been performed by finding whether known key interactions, such as "dueling", are highlighted successfully. An example of dueling is shown above (with yellow dash lines), in which two engaged ants face each other to move back and forth several times. In ant communities, this aggressive interaction is already known to be an *informative* behavioral signal to predict the abnormal state in colonies of *H. saltator* ants. Hence, BT is expected to be able to discover the occurrences of dueling as key local observations whilst IO-GEN is detecting abnormal states. 

<img src=imgs/fig5.jpg width="80%">

Four output examples are displayed above: (a) BT does not use thresholding for prioritisation of Grad-CAM outputs and (b)-(d) top 5\% heatmaps are observed near the ants engaged in dueling, even though spatial coordinate information was *not* provided as input.

<img src=imgs/fig6.jpg width="80%">

(a)-(b) above show the resultant heatmaps appear *only* around dueling interactions (yellow lines) whilst other ants simply moving swiftly (white lines) are *ignored*. 


# Installation 

1. Clone the repository
    ```
    $ git clone https://github.com/ctyeong/BeyondTracking.git
    ```

2. Install the required Python packages

    ```
    $ pip install -r requirements.txt
    ```
   - Python 3.6 is assumed to be installed already


# Execution 

We assume that a state detector has been trained with IO-GEN to be saved as `IO-GEN_Path/IO-GEN.h5` by following [this instruction](https://github.com/ctyeong/IO-GEN#training).


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