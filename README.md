# GNN_E
Usefull links:
- Origina Brain_GNN: (link)[https://github.com/xxlya/BrainGNN_Pytorch/tree/main]
- Explained Brain_GNN: (link)[https://github.com/arashsm79/brain-opto-fmri-decoding-gnn/tree/main]
- Paper Brain_GNN: (link)[https://www.sciencedirect.com/science/article/pii/S1361841521002784]


# MICHAEL'S ANSWERS
- take abs of z-correct 
    - very differt sing of emotion
    - we **suppose** that the same graph would be elicited in the brain both for neg and pos values
- only use test-train   
    - no crossvalidation
- train/test split  
    - all emo in test
    - same distrubution in test
    - not random --> scenes not too close --> info leackage
    - in split horizonyllay, legit to have a little info leackge
- prdict only:
  - [‘Anger’,‘Guilt’,‘WarmHeartedness’,‘Disgust’,‘Happiness’,‘Fear’,‘Regard’,‘Anxiety’, ‘Satisfaction’,‘Pride’,‘Surprise’,‘Love’,‘Sad’]
-

- models
    - basline model
      - KNN   OK --> C
	  - RF    OK --> C
      - GCN   OK --> Z
      - FNN   OK --> G
    - lots of simple ones
        - GAT   OK --> G
        - transf   --> 
        - RF       --> 
        - OpenGSL  --> C
    - one complex from a paper
        - VIB-gls (autoencoder + classification)  --> GZC


	- images
		- plot labels distruvtuon in mvoies
		- image of VIB model
		- images of FN used
	- tables
		- reutls
  
- decide order results ans narrative

    - try to rpedcit emo, BUT new stuff
    	--> not just prediction, but also the activration/netowdok that elivct that emotion -- GSL
    	- use fMRI -- why? 
    		- more deep regions are related to emptions
    - thus we want to use GNN
    - problems:
    	- do not have a staring grpah
    	- emotion have difffert iems in elciting --> diffcult to makae windows/feattures
    	- emotion bold singlas mixes
    	- splititng in traina nd validation (not tto close, not random, kl diverge, balance)
    	- pay attention to oversmoothing
    - we started with baseline model (not so powerful)
      - KNN 
      	- random
      	- features that we used
      - RF
      	- 
      - FNN
      	- above chnage but not sol much 
      		- maybe just to chnage
      		- it is overftting
      	- exolain the structure
      - GCN
      	- strucre
      	- results
    - we diced to use more complex mdoelfs, that were also alble to descirbe the undelrung networks
        - GAT  
        	- why? beaocme the can ointerpest the attention weights as newtornds
        	- show the proposed strucutre (ine head ofr eahc emo)
        	- image??
    - lastly we decided to use a model not impe,mented by ourself, from a publication --> VIB
    	- why? 
    		- one complex from a paper --< paper looks promisng (used on simialr datasets to oiur)
    		- it has rgrpah classifcation
    		- it is GSL
	- what is it?
		- medel descriptions (GSL + VAE + classfier)
		- parmaters that we can choose)
			- whcih values did we choose --> why we cjhoose these values
	- results
		- many differt combiuantrions of paramters + intiial grpahs creration


WEDNESDAY
- C
	- put RF and KNN (both with feat and raw singaml )
		- on git
		- on latex
		- on presnation
		- with final results
- G
	- gat 
- Z
	- gcn
	- put resutl on latex and preseantion
- ALL
	- try some paramters of vib
	- pipe: 1 movie 1 sun --> 8 mvoes 1 sub --> all
- sevda
	- not cite articel trnaofmers
	- try with self loop


# presenation:  17th tuesday 14:00 --> 10 min + 5 min
