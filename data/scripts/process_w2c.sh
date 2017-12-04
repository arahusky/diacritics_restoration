#!/bin/bash
# download W2C corpora, split text into lines and these lines split into disjoint training, devel and testing sets
# concurrent run is not guaranteed and will most likely fail

# three arguments are required -- language identificator for W2C (e.g. ces -- ISO 639-3 format), language
# identificator for moses sentence splitter (e.g. cs -- most likely ISO 639-1) and the third argument is path to folder
# to store created dataset

# for W2C codes see https://ufal.mff.cuni.cz/~majlis/w2c/download.html
# for MOSES codes see https://github.com/moses-smt/mosesdecoder/tree/master/scripts/share/nonbreaking_prefixes

LANG_W2C=${1:-"ces"}
LANG_MOSES=${2:-"cs"} # perl
DATA_FOLDER=${3:-"/home/arahusky/troja/w2c_data/cs"}

# download and unzip W2C wiki corpora for given language
echo "Downloading and unzipping data"
mkdir -p ${DATA_FOLDER}
# curl "http://ufal.mff.cuni.cz/~majlis/w2c/download.php?lang=${LANG_W2C}&type=wiki" -o ${DATA_FOLDER}/temp_w2c_.${LANG_W2C}.txt.gz
cp /net/data/W2C/W2C_WIKI/2011-11-50000-10000/${LANG_W2C}_50000.txt.gz ${DATA_FOLDER}/temp_w2c_.${LANG_W2C}.txt.gz
gunzip -c /${DATA_FOLDER}/temp_w2c_.${LANG_W2C}.txt.gz > ${DATA_FOLDER}/w2c_data.txt
rm${DATA_FOLDER}/temp_w2c_.${LANG_W2C}.txt.gz

# split into sentences
echo "Splitting into sentences"
cat ${DATA_FOLDER}/w2c_data.txt | perl split_sentences.perl -l ${LANG_MOSES} | sed -e '/<P>/d' > ${DATA_FOLDER}/w2c_sentences.txt
#rm ${DATA_FOLDER}/w2c_data.txt

# split into training, devel and testing sets
echo "Splitting into train/dev/test sets"
num_lines=$(cat ${DATA_FOLDER}/w2c_sentences.txt | wc -l)
(echo ${num_lines}; cat ${DATA_FOLDER}/w2c_sentences.txt) | python3 create_dataset.py ${DATA_FOLDER}
#rm ${DATA_FOLDER}/w2c_sentences.txt

# ensure that the three created sets are disjoint
echo "Creating disjoint sets"
## dev - train
python3 make_disjoint_sets.py ${DATA_FOLDER}/target_dev.txt ${DATA_FOLDER}/target_test.txt
mv ${DATA_FOLDER}/target_dev.txt_temp ${DATA_FOLDER}/target_dev.txt

## train - dev
python3 make_disjoint_sets.py ${DATA_FOLDER}/target_train.txt ${DATA_FOLDER}/target_dev.txt
mv ${DATA_FOLDER}/target_train.txt_temp ${DATA_FOLDER}/target_train.txt

## train - test
python3 make_disjoint_sets.py ${DATA_FOLDER}/target_train.txt ${DATA_FOLDER}/target_test.txt
mv ${DATA_FOLDER}/target_train.txt_temp ${DATA_FOLDER}/target_train.txt

# strip diacritics
echo "Stripping diacritics"
cat ${DATA_FOLDER}/target_train.txt | python3 diacritization_stripping.py ${DATA_FOLDER}/input_train.txt ${LANG_MOSES} --uninames
cat ${DATA_FOLDER}/target_dev.txt | python3 diacritization_stripping.py ${DATA_FOLDER}/input_dev.txt ${LANG_MOSES} --uninames
cat ${DATA_FOLDER}/target_test.txt | python3 diacritization_stripping.py ${DATA_FOLDER}/input_test.txt ${LANG_MOSES} --uninames
