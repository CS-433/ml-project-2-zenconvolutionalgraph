
# AFTER THE RAIN
- duration sec = 382 --> 293 TR
- start sec = 76 --> 56 TR


# GNN_E

Usefull links:
- Origina Brain_GNN: (link)[https://github.com/xxlya/BrainGNN_Pytorch/tree/main]
- Explained Brain_GNN: (link)[https://github.com/arashsm79/brain-opto-fmri-decoding-gnn/tree/main]
- Paper Brain_GNN: (link)[https://www.sciencedirect.com/science/article/pii/S1361841521002784]

# INTRO
- what is emotion detection

# Dataset
- frmi acquistion parmaters
- preopsocessin frmi 
- movies, films, people (hown manu, gneder, ...)
- how labels have been create + z.scoring
- how labels have been aligned
    - emotion lables
        - regression
        - classification
            -thr (multilabel)
            -max value (singlelabel)

# Explanation of our model choice

## DIfficuluties in emo detection
- decide node features
  - singlw raw frmi value
  - window (both beofre and after, no real time prediction)
- not know which is the correct **initial graph strucure** (fc, sc, fc window, clique, ...)
  - a **priori knowldeg (we decide)**
    - FC of the subject, edges calculated with
        - in window
        - in all movie
    - FN (14 regions always connected via clique)
        - all edges constants
        - edge clausted with window
    - **GSL**
      - OpenGSL
        - still need intila one --> clique
        - not interpretable
    - **GAT** (model tries to learn most improtat)
      - still need intioa one
      - cannot use the clique (it is a transformer)
    - **transformer**
      - no need for intiial grpah
    - 
- differt emo differt time eliciting, difficult window size
- emotion labelling is correct but still controversial
- oversmoothing (geneal gnn)

## Hypotheisze solution
- transformer
  -(https://github.com/elyesmanai/simpletransformerss?tab=readme-ov-file --> need before to pass form numebr to text)
  -orsimpletrnafomer done by us
  - https://github.com/lucidrains/tab-transformer-pytorch/tree/main
- use GAT
  -(https://github.com/usccolumbia/deeperGATGNN)
- use GSL
  - open gsl

# Cross calidation
NO --> decision made with the lab

# ATTENTION how tor sploit trian and tets
cannot do random --> maybe some timepoints in traion and tets are close --> so they are same scene of the kvoie --> so it like a little of info lackage
Solution: use whole movirs ofr tets ste


# WEDNESDAY
# GNN_E

Usefull links:
- Origina Brain_GNN: (link)[https://github.com/xxlya/BrainGNN_Pytorch/tree/main]
- Explained Brain_GNN: (link)[https://github.com/arashsm79/brain-opto-fmri-decoding-gnn/tree/main]
- Paper Brain_GNN: (link)[https://www.sciencedirect.com/science/article/pii/S1361841521002784]

# INTRO
- what is emotion detection

# Dataset
- frmi acquistion parmaters
- preopsocessin frmi 
- movies, films, people (hown manu, gneder, ...)
- how labels have been create + z.scoring
- how labels have been aligned
    - emotion lables
        - regression
        - classification
            -thr (multilabel)
            -max value (singlelabel)

# Explanation of our model choice

## DIfficuluties in emo detection
- decide node features
  - singlw raw frmi value
  - window (both beofre and after, no real time prediction)
- not know which is the correct **initial graph strucure** (fc, sc, fc window, clique, ...)
  - a **priori knowldeg (we decide)**
    - FC of the subject, edges calculated with
        - in window
        - in all movie
    - FN (14 regions always connected via clique)
        - all edges constants
        - edge clausted with window
    - **GSL**
      - OpenGSL
        - still need intila one --> clique
        - not interpretable
    - **GAT** (model tries to learn most improtat)
      - still need intioa one
      - cannot use the clique (it is a transformer)
    - **transformer**
      - no need for intiial grpah
    - 
- differt emo differt time eliciting, difficult window size
- emotion labelling is correct but still controversial
- oversmoothing (geneal gnn)

## Hypotheisze solution
- transformer
  -(https://github.com/elyesmanai/simpletransformerss?tab=readme-ov-file --> need before to pass form numebr to text)
  -orsimpletrnafomer done by us
  - https://github.com/lucidrains/tab-transformer-pytorch/tree/main



# Cross calidation
NO --> decision made with the lab

# ATTENTION how tor sploit trian and tets
cannot do random --> maybe some timepoints in traion and tets are close --> so they are same scene of the kvoie --> so it like a little of info lackage
Solution: use whole movirs ofr tets ste

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
- check distribution triana nd chekc with verticla and horizontal
- OK: try to classify between movie and rest --> FNN 87% accuracy
- OK --> mutiple label classfication  [1 0 1 0 1... 0   


- models
    - lots of simple ones
        - GAT      --> G
        - FNN   OK --> G
        - knn   OK --> C
        - GNN      --> Z
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
    - chnage GAT G
  - implement VIB
tuesday: 19th 11:00
  - implement VIB https://github.com/RingBDStack/VIB-GSL
presenation:  17th tuesday 14:00 --> 10 min + 5 min