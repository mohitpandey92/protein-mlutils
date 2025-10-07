#update the version in pyproject.toml
# and commit the changes
# usage: ./update_package.sh <new_version>
if [ -z "$2" ]; then
  echo "Usage: $0 <new_version> <commit_message>"
  exit 1
fi  
NEW_VERSION=$1
commit_message=$2

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i "s/version = .*/version = \"$NEW_VERSION\"/" pyproject.toml
elif [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/version = .*/version = \"$NEW_VERSION\"/" pyproject.toml
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi
echo "Recognized $OSTYPE. Updated version to $NEW_VERSION in pyproject.toml"
rm -rf dist
python -m build &&  python -m pip install --upgrade dist/*.gz
git add pyproject.toml src/proteinmlutils/*.py README.md notebook/*.ipynb *.sh
git commit -m "Update version to $NEW_VERSION. $commit_message"
git push 
echo "Version updated to $NEW_VERSION, package built and changes committed."