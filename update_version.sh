#update the version in pyproject.toml
# and commit the changes
# usage: ./update_package.sh <new_version>
if [ -z "$1" ]; then
  echo "Usage: $0 <new_version>"
  exit 1
fi  
NEW_VERSION=$1
sed -i '' "s/version = .*/version = \"$NEW_VERSION\"/" pyproject.toml
rm -rf dist
python -m build &&  python -m pip install --upgrade dist/*.gz
git add pyproject.toml src/proteinmlutils/
git commit -m "Update version to $NEW_VERSION"
git push 
echo "Version updated to $NEW_VERSION, package built and changes committed."