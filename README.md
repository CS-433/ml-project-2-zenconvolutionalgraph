# WHAT CAN WE CHANEG VIB
- backbone (gin, gat, gcn(
- lr
- clamp claucltation of std in farward()


# GNN_E
Usefull links:
- Origina Brain_GNN: (link)[https://github.com/xxlya/BrainGNN_Pytorch/tree/main]
- Explained Brain_GNN: (link)[https://github.com/arashsm79/brain-opto-fmri-decoding-gnn/tree/main]
- Paper Brain_GNN: (link)[https://www.sciencedirect.com/science/article/pii/S1361841521002784]

Feautres are the fmri singals aordinating the crretn timepotin with a symemtric winow of differ sizes.
- not know which is the correct **initial graph strucure** (fc, sc, fc window, clique, ...)
    - for egde atrtibutes
      -FC of the subjectcalculated with
        - in sliding window
        - using the whole movie movie - so constant
      - constant values
    - for nodes
      - use onyl subset of nodes
        - onyl the one in specific sunctionak netwokf, it is a for of a priori knolwdge (pleae not the the 14 subdornticla regions always there)
        - all 414 nodes
these intila grpah concctivies will be the refined using theicjek like GSL; GAT, ... Indeed one of our goals is to find what can abe an eindeirlyn hidden netowek for each emotion


## Hypotheisze solution
- transformer
  -(https://github.com/elyesmanai/simpletransformerss?tab=readme-ov-file --> need before to pass form numebr to text)
  -orsimpletrnafomer done by us
  - https://github.com/lucidrains/tab-transformer-pytorch/tree/main
- use GAT
  -(https://github.com/usccolumbia/deeperGATGNN)
- use GSL
  - open gsl


Usefull links:
- Origina Brain_GNN: (link)[https://github.com/xxlya/BrainGNN_Pytorch/tree/main]
- Explained Brain_GNN: (link)[https://github.com/arashsm79/brain-opto-fmri-decoding-gnn/tree/main]
- Paper Brain_GNN: (link)[https://www.sciencedirect.com/science/article/pii/S1361841521002784]




# Cross calidation
NO --> decision made with the lab


# MICHAEL'S ANSWERS
- take abs of z-correct 
    - very differt sing of emotion
    - we **suppose** that the same graph would be elicited in the brain both for neg and pos values
- classifiy each timepoint, not only the one where emotions are stable
- only use test-train   
    - no crossvalidation
- train/test split  
    - all emo in test
    - same distrubution in test
    - not random --> scenes not too close --> info leackage
    - in split horizonyllay, legit to have a little info leackge
- yes, we have 14 movies
- prdict only:
  - [‘Anger’,‘Guilt’,‘WarmHeartedness’,‘Disgust’,‘Happiness’,‘Fear’,‘Regard’,‘Anxiety’, ‘Satisfaction’,‘Pride’,‘Surprise’,‘Love’,‘Sad’]
-



# WEDNESDAY

  TODO
- OK --> change the df_movies to use only 13 emotios 
- OK: check distribution triana nd chekc with verticla and horizontal
- OK: try to classify between movie and rest --> FNN 87% accuracy
- OK --> mutiple label classfication  [1 0 1 0 1... 0   


- models
    - basline model
      - knn   OK --> C
      - GCN   OK --> Z
      - FNN   OK --> G


    - lots of simple ones
        - GAT   OK --> G
        - FNN   OK --> G
        - transf   --> 
        - RF       --> 
        - OpenGSL  --> C
    - one complex from a paper
        - VIB-gls  
            - autoencoder


Sunday: 10th  8:00
  - by then:
    - gnn Z
    - openGSL C
    - OK chnage GAT G
  - OK: implent clique for lymbic
  - OK implementation + interpetation VIB
tuesday: 19th 11:00
  - implement VIB https://github.com/RingBDStack/VIB-GSL
presenation:  17th tuesday 14:00 --> 10 min + 5 min