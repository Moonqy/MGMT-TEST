# MGMT-TEST
python -c "import monai" || pip install -q "monai-weekly[nibabel, tqdm]"
python -c "import matplotlib" || pip install -q matplotlib
pip install -q "monai-weekly[einops]"
%matplotlib inline
