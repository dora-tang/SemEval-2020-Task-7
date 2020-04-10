data_dir=semeval_data
mkdir $data_dir
cd $data_dir
# brew install wget
# train extra
wget https://www.cs.rochester.edu/u/nhossain/funlines/semeval-2020-task-7-extra-training-data.zip -O set2.zip
unzip -o set2.zip
# train + dev + test_without_label
wget https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data-full.zip -O set3.zip
unzip -o set3.zip
# test_with_label
wget https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data-test_with_labels.zip -O set4.zip
unzip -o set4.zip

rm -r set2.zip set3.zip set4.zip
rm -r __MACOSX
