## Github CheatSheet

```
git clone <url>
```


Once you have new files and then make a custom branch and push it
```
git checkout -b v1
git add .
git commit -m "Added new files"
git push origin v1
```


Make sure the github is updated in the UI and cross confirm it.

lets say you made a wrong push and wanted to revert back
```
git log
git reset --hard <commit_id>
git push origin v1 --force
```


> use pyclean to remove all the cache.
> use gitignore to ignore the files that you dont want to push.
> dont upload heavy files and too many images to github, please log them in google drive and share up link the readme.

make sure you are dealing things in the right way if you are working in a team and dont push to main branch.


lets say if you want to delete a branh locally
```
git branch -d <branch_name>
```