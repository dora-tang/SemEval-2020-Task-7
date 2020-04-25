data_dir=${1:-semeval_data}
mkdir -p $data_dir
cd $data_dir
# brew install wget
echo -e "\n===== Data: extra train ======\n"
wget https://www.cs.rochester.edu/u/nhossain/funlines/semeval-2020-task-7-extra-training-data.zip -O set2.zip
unzip -q -o set2.zip
echo -e "\n===== Data: train, dev, test without label ======\n"
# train + dev + test_without_label
wget https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data-full.zip -O set3.zip
unzip -q -o set3.zip
# test_with_label
echo -e "\n===== Data: test with label ======\n"
wget https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data-test_with_labels.zip -O set4.zip
unzip -q -o set4.zip

# remove unnessary files
rm -r set2.zip set3.zip set4.zip
rm -r __MACOSX

echo -e "\n===== Data is downloaded to directory: $data_dir ======\n"
