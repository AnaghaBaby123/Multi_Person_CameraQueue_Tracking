A real-time multi-camera system that detects people in a queue (YOLOv8), tracks them across frames (Deep SORT), and maintains stable global identities (OSNet-based ReID). It fuses streams from two cameras into a unified view, displays per-person bounding boxes with confidence scores and global IDs, and reports the live queue length.

**reid_osnet.pt** : The fine tuned reid with osnet with the persons in the video. 
<figure>
  <img src="sample.png" alt="Queue detection demo" width="600" />
  <figcaption>Figure 1. The multi-camera queue detection in action. GID 0 and GID 1 is getting same ID's in both frames.</figcaption>
</figure>

