#!/bin/bash
# download W2C corpora, split text into lines and split these lines into disjoint training, devel and testing sets
# concurrent run is not guaranteed and will most likely fail

# four arguments are required -- STATMT data file identifier (e.g. cs.2015_22), language
# identificator for moses sentence splitter (e.g. cs -- most likely ISO 639-1), path to folder to store created dataset
# and whether to append created sentences to existing training sentences or keep them separate (y/n)

# for STATMT data files identifiers see http://data.statmt.org/ngrams/raw/
# for MOSES codes see https://github.com/moses-smt/mosesdecoder/tree/master/scripts/share/nonbreaking_prefixes

set -e

# testing -- et.2015_27
STATMT_ID=${1:-"cs.2015_22"}
LANG_MOSES=${2:-"cs"} # perl
DATA_FOLDER=${3:-"/home/arahusky/troja/w2c_data/cs"}
DO_APPEND=${4:-n} # append to existing training corpus?

# download and unzip STATMT wiki corpora for given language
echo "Downloading and unzipping data"
STATMT_XZ_FILENAME=${DATA_FOLDER}/temp_${STATMT_ID}.raw.xz
STATMT_EXTRACTED_FILENAME=${DATA_FOLDER}/statmt_data_${STATMT_ID}.txt
curl "http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/${LANG_MOSES}/raw/${LANG_MOSES}.${STATMT_ID}.raw.xz" -o ${STATMT_XZ_FILENAME}
xzcat ${STATMT_XZ_FILENAME} > ${STATMT_EXTRACTED_FILENAME}
rm ${STATMT_XZ_FILENAME}

# remove invalid UTF-8, all wiki and additional information
STATMT_FILTERED_DATA_FILENAME=${DATA_FOLDER}/statmt_filtered_data_${STATMT_ID}.txt
python3 preprocess_statmt.py ${STATMT_EXTRACTED_FILENAME} ${STATMT_FILTERED_DATA_FILENAME}
rm ${STATMT_EXTRACTED_FILENAME}

# split into sentences
echo "Splitting into sentences"
STATMT_SENTENCES_FILENAME=${DATA_FOLDER}/statmt_sentences_${STATMT_ID}.txt
cat ${STATMT_FILTERED_DATA_FILENAME} | perl split_sentences.perl -l ${LANG_MOSES} | sed -e '/<P>/d' > ${STATMT_SENTENCES_FILENAME}
rm ${STATMT_FILTERED_DATA_FILENAME}

# split into training, devel and testing sets
MIN_CHARS=100 # minimal number characters the sentence has to have to be included in the training set (short sentences are assumed to be noisiest)
MAX_CHARS=300 # maximal number characters the sentence has to have to be included in the training set (long sentences are assumed to be noisiest and also RNN does process long sentences too slowly)
STATMT_TRAINING_CANDIDATES_FILENAME=${DATA_FOLDER}/statmt_${STATMT_ID}_train_candidate_target_sentences.txt
echo "Lower-casing, striping and removing lines with low number of characters"
num_lines=$(cat ${DATA_FOLDER}/statmt_sentences_${STATMT_ID}.txt | wc -l)
cat ${STATMT_SENTENCES_FILENAME} | python3 create_dataset.py ${num_lines} ${DATA_FOLDER} \
    --min_chars ${MIN_CHARS} --max_chars ${MAX_CHARS} --dev_sentences 0 --test_sentences 0 --filename_train \
    ${STATMT_TRAINING_CANDIDATES_FILENAME} --filename_dev ${DATA_FOLDER}/pom_dev.txt --filename_test ${DATA_FOLDER}/pom_test.txt

rm ${DATA_FOLDER}/pom_dev.txt
rm ${DATA_FOLDER}/pom_test.txt
rm ${STATMT_SENTENCES_FILENAME}

# remove collisions with existing devel and testing sets
echo "Creating disjoint sets"
## new train -- dev
python3 make_disjoint_sets.py ${STATMT_TRAINING_CANDIDATES_FILENAME} ${DATA_FOLDER}/target_dev.txt
mv ${STATMT_TRAINING_CANDIDATES_FILENAME}_temp ${STATMT_TRAINING_CANDIDATES_FILENAME}

## new train -- test
python3 make_disjoint_sets.py ${STATMT_TRAINING_CANDIDATES_FILENAME} ${DATA_FOLDER}/target_test.txt
mv ${STATMT_TRAINING_CANDIDATES_FILENAME}_temp ${STATMT_TRAINING_CANDIDATES_FILENAME}

# strip diacritics
TRAIN_INPUT_SENTENCES_FILENAME=${DATA_FOLDER}/statmt_${STATMT_ID}_train_input_sentences.txt
TRAIN_TARGET_SENTENCES_FILENAME=${DATA_FOLDER}/statmt_${STATMT_ID}_train_target_sentences.txt
echo "Stripping diacritics"
cat ${STATMT_TRAINING_CANDIDATES_FILENAME} | python3 diacritization_stripping.py \
    ${TRAIN_INPUT_SENTENCES_FILENAME} ${LANG_MOSES} --uninames --diacritics_percentage \
    --out_input_file ${TRAIN_TARGET_SENTENCES_FILENAME}

rm ${STATMT_TRAINING_CANDIDATES_FILENAME}

# append to training data if requested
if [ ${DO_APPEND} = "y" ]; then
    echo "Appending to training sentences"
    cat ${TRAIN_INPUT_SENTENCES_FILENAME} >> ${DATA_FOLDER}/input_train.txt
    cat ${TRAIN_TARGET_SENTENCES_FILENAME} >> ${DATA_FOLDER}/target_train.txt

    rm ${TRAIN_INPUT_SENTENCES_FILENAME}
    rm ${TRAIN_TARGET_SENTENCES_FILENAME}
fi