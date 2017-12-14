#!/bin/bash

set -e

languages_short=(cs) # vi de)
languages_long=(ces) # vie deu)
save_dir=~/troja/diacritization_data/

statmt_datafiles=(2017_17) # 2016_50)

for (( i=0; i<${#languages_short[@]}; ++i))
do
    lang_save_dir=${save_dir}/${languages_short[$i]}
    bash process_w2c.sh ${languages_long[$i]} ${languages_short[$i]} ${lang_save_dir}

    for statmt_datafile in ${statmt_datafiles}
    do
        bash process_statmt.sh ${statmt_datafile} ${languages_short[$i]} ${lang_save_dir} n
    done
done

