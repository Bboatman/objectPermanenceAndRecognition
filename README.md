# Red Light / Green Light : An Algorithmic Approach
## About
This project focuses on computer vision topics that have wide utility, motion tracking and entity detection, to re-imagine a version of the children's game Red Light, Green Light in which a computer acts as the 'stoplight' player. For this to be possible, the algorithm must be able to do more than track motion. Therefore, this algorithm is capable of identifying significant entities in the frame and remembering them as unique. This enables the computer to track multiple players about the screen and remain capable of telling them apart regardless of their location inside (or out of) the frame.

More can be read [here](https://docs.google.com/document/d/1riwWNXvnWbfoEWd7sR2uf_6dwW992XKponkObq-c-Bk/edit?usp=sharing)

## TODO
* insert unsupervised learning techniques in such a way that the hard coded values could be calibrated to be reflexive in relation to camera quality and lighting conditions
* investigate on the fly background imagery generation
* create human model (using SVM?) to detect body parts and notice specific human entities over simple motion
