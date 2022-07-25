# DBT-Net
The audio demos with respect to the paper "DBT-Net: Dual-branch federative magnitude and phase estimation with attention-in-attention transformer for monaural speech enhancement" are provided (Accepted by IEEE TASLP).  The code and the pretained model is also released.

### Overall architecture:
  
  
  ![image](https://user-images.githubusercontent.com/51236251/153812164-e7ab16c3-bcb0-494e-99a8-70671c812eb4.png)

### Code:
You can use dual_aia_trans_merge_crm() in aia_trans.py for dual-branch SE, while aia_complex_trans_mag() and aia_complex_trans_ri() are single-branch aprroaches.
The trained weights on VB dataset, 30h WSJ0-SI84 datset and 300h 2020 DNS-Challenge are also provided. You can directly perform inference or finetune the model by using vb_aia_merge_new.pth.tar. 

### requirements:
	
	CUDA 10.1
	torch == 1.8.0
	pesq == 0.0.1
	librosa == 0.7.2
	SoundFile == 0.10.3

### How to train
### Step1
prepare your data. Run json_extract.py to generate json files, which records the utterance file names for both training and validation set

	# Run json_extract.py
	json_extract.py
	
### Step2
change the parameter settings accroding to your directory (within config_vb.py or config_dns.py)
	
### Step3
Network Training (you can also use aia_complex_trans_mag() and aia_complex_trans_ri() network in aia_trans.py for single-branch SE)

	# Run main_vb.py or main_dns.py to begin network training 
	# solver_merge.py and train_merge.py contain detailed training process
	main_vb.py


### Inference:
The trained weights are provided in BEST_MODEL. 

	# Run enhance_vb.py or enhance_wsj.py to enhance the noisy speech samples.
	enhance_vb.py 


## Experimental Results

### WSJ0-SI84 Dataset 
![lQLPDhtQojea5r3NAsTNBCGwwIWt6yPeAGcCVMiMrwD1AA_1057_708](https://user-images.githubusercontent.com/51236251/162556894-52356262-6145-4f92-82a9-5830e32a9d76.png)

### DNSMOS 

![816A7E97-B2AC-4a40-A1E7-5160BA631A1D](https://user-images.githubusercontent.com/51236251/162556923-4eb60b03-7129-47b1-b43e-953337a8301f.png)

### Voice-Bank + Demand dataset


![3D94CBE7-904A-4fd1-95F9-D640899F105F](https://user-images.githubusercontent.com/51236251/162556935-a1b9bff8-b831-4277-aaee-5ebc4047e271.png)


## Spectrogram Visualization

![lQLPDhtQogiEcxbNAm_NBl2wnOokv3Dc5iECVMg-l0D5AA_1629_623](https://user-images.githubusercontent.com/51236251/162556886-d159dfbd-37bc-4c66-9b78-b9a1761c2a46.png)

