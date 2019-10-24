Structuring the Project (in descending order of importance/usefulness/detail):

  * [seems to say everything, but is a lot of reading](https://docs.python-guide.org/writing/structure/)
  * [extensive---probably useful for a while](https://python-packaging.readthedocs.io/en/latest/minimal.html)
  * [short---covered in other refs---think i've already done it all]( https://able.bio/SamDev14/how-to-structure-a-python-project--685o1o6)

Readthedocs website for medil.causal.dev:
  * [adding custom domain](https://docs.readthedocs.io/en/stable/custom_domains.html)
  * [specifying canonical URL](https://docs.readthedocs.io/en/stable/guides/canonical.html)

Code:
  * could make a load module containing funcs such as from\_dict and from\_edges and from\_adj\_matrix etc, but not a big priority---network does it all well already, and medil really just needs to work with the output from the perm test methods
  * maybe change UndirectedDependenceGraph object init argument to be a adj matrix instead?
