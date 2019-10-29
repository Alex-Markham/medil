Structuring the Project (in descending order of importance/usefulness/detail):

  * [seems to say everything, but is a lot of reading](https://docs.python-guide.org/writing/structure/)
  * [extensive---probably useful for a while](https://python-packaging.readthedocs.io/en/latest/minimal.html)
  * [short---covered in other refs---think i've already done it all]( https://able.bio/SamDev14/how-to-structure-a-python-project--685o1o6)

Readthedocs website for medil.causal.dev:
  * [adding custom domain](https://docs.readthedocs.io/en/stable/custom_domains.html)
  * [specifying canonical URL](https://docs.readthedocs.io/en/stable/guides/canonical.html)

<<<<<<< HEAD
Check out [this](http://signal.ee.psu.edu/mrf.pdf) slidedeck for a nice summary of MRF and related graph theory concepts

See [this](https://en.wikipedia.org/wiki/Markov_random_field) as well as factor graphs and clique factorization

Undirected dependency graph is complement of Markov Random Field with global Markov property? And so cliques of UDG would be independent sets---ecc of UDG corresponds to sufficient factorization of MRF into independent sets

[testing in python](https://realpython.com/python-testing/)
[more testing](https://docs.python-guide.org/writing/tests/)
=======
Code:
  * could make a load module containing funcs such as from\_dict and from\_edges and from\_adj\_matrix etc, but not a big priority---network does it all well already, and medil really just needs to work with the output from the perm test methods
  * maybe change UndirectedDependenceGraph object init argument to be a adj matrix instead?
>>>>>>> 5ceda7d09247b1dda81403085c71a4e219b52210
