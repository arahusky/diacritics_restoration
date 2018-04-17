#!/bin/bash
# prepare data (both w2c and statmt) for given language
# call me like bash prepare_data_for_language.sh cs ces

set -e

languages_short=${1:-"cs"}
# abbreviations from http://data.statmt.org/ngrams/raw/
# vietnames romanian latvian czech slovak irish french hungarian polish slovenian croatian spanish turkish serbian
# vi ro lv cs sk ga fr hu pl sl hr es tr sr

languages_long=${2:-"ces"}
# abbreviations from https://ufal.mff.cuni.cz/~majlis/w2c/download.html
# vie ron lat ces slk gle fra hun pol slv hrv spa tur srp

save_dir=${3:-"/tmp"}

statmt_datafiles=(2017_17) # 2016_50)

lang_save_dir=${save_dir}/${languages_short}
bash process_w2c.sh ${languages_long} ${languages_short} ${lang_save_dir}

for statmt_datafile in ${statmt_datafiles}
do
    bash process_statmt.sh ${statmt_datafile} ${languages_short} ${lang_save_dir} n
done

