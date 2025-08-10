git pull 
rm -rf dist
python -m build &&  python -m pip install --upgrade dist/*.gz
echo "updated the package"