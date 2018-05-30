seqdist
========

Utility for pairwise identity and similarity calculations from
MSAs.

Usage
-----

Calculate the pairwise percent identity between aligned sequences. Currently
does NOT support a2m format (containing lowercase unaligned residues), so make
sure to input a cleaned alignment (``sparalign.py`` output option ``-a``). ::

    $ seqdist.py aligned_seqs.afa -o aligned_seqs.dist

Calculate the pairwise non-similarity between aligned sequences::

    $ seqdist.py aligned_seqs.afa -m nonsim -o aligned_seqs.nonsim.dist

The implementation is Python/numpy and can be slow, but multiprocessing is
enabled. Calculate nonsimilarity with 8 cores::

    $ seqdist.py aligned_seqs.afa -m nonsim -n 8 -o aligned_seqs.nonsim.dist

On compute1 with 16 cores, ``seqdist.py`` calculates a similarity matrix for 5700 OMTs in ~20min.

By default, similarity is defined as amino-acid substitutions with score >= 1
in BLOSUM62 matrix. The above is equivalent to::

    $ seqdist.py aligned_seqs.afa -m nonsim --subsmat blosum62 --score-thresh 1 -o aligned_seqs.nonsim.dist

List available substitution matrices::

    $ seqdist.py --list-subsmat

More help::

    $ seqdist.py -h

