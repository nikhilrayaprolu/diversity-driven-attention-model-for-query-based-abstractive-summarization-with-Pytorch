# diversity-driven-attention-model-for-query-based-abstractive-summarization-with-Pytorch

## Data Download and Preprocessing
    * cd src/dataextraction_scripts
    * The model will extract the data for the categories mentioned in file debatepedia_categories
    * sh extract_data.sh
    
    ### To use the existing extracted dataset in dataset folder:
    * cd src/dataextraction_scripts
    * python make_folds.py ../../data <number_of_folds> <new_dir_for_10_folds> 
    * By default run : python make_folds.py ../../data 10 ../../data

## Get the Glove embeddings:
    mkdir Embedding
    cd Embedding
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
    echo 2196017 300 > temp_metadata
    cat temp_metadata glove.840B.300d.txt > embeddings
    rm temp_metadata
    
    
Experiments: Estimated time of completion 18th April <br/>
List of Experiments: <br/>
Implement the paper in pytorch <br/>
Add Cross selective Encoding instead of cross Attentions <br/>
Add pointer generator paradigm <br/>
