cd ../data || exit
unzip wiki_zh_2019.zip
cd small_wiki_zh || exit

for dir in ./*; do
  for file in ${dir}/*; do
    cat $file | jq ".text" | sed -e 's/^"//' -e 's/"$//' >> ../training_data.txt
  done
done
