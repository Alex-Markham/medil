TODO:

- got docs pipline working for docs website, but still need to restrict it to master branch somehow:
```
only:
    changes:
    - docs/*
  - master
 ```
 
- finish documentation
- docs for different versions of medil
- put on conda

 -------------------------------------------------------------------------------
 
For viewing docs site locally:
 
 docs/: make clean; make html
 
 docs/_build/: python -m http.server

-------------------------------------------------------------------------------

[for conda package](https://stackoverflow.com/questions/49474575/how-to-install-my-own-python-module-package-via-conda-and-watch-its-changes)
[conda](https://enterprise-docs.anaconda.com/en/latest/data-science-workflows/packages/upload.html)
[for uploading to pip](https://packaging.python.org/tutorials/packaging-projects/)

-------------------------------------------------------------------------------

Pushing new release: don't forget to update version number in setup.py 
```bash
git push origin develop
git tag -a v0.X.0 -m "Releasing version 0.X.0"
git push origin v0.X.0
git push origin develop:master
```

-------------------------------------------------------------------------------

Structuring the Project (in descending order of importance/usefulness/detail):
  * [seems to say everything, but is a lot of reading](https://docs.python-guide.org/writing/structure/)
[testing in python](https://realpython.com/python-testing/)
[more testing](https://docs.python-guide.org/writing/tests/)
