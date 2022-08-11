# The Apportionment Problem

Class: MAT 630
Created: March 31, 2022 4:33 PM
Materials: discrete.pdf
Reviewed: No
Type: Seminar

# The Apportionment Problem

In every method of apportioning the House of Representatives, the ideal quota for a state
is now calculated by the formula 435·(state population/total population). Because this is
not likely to be an integer, it is necessary to either round up to the upper quota or round
down to the lower quota to obtain a meaningful result. A method of apportionment must
specify exactly how this rounding is to be done.

Another quantity of significance in apportionment is the ideal district size, which is the total
population divided by the total number of representatives. The 2000 census puts this figure
at 646 952 = 281 424 177/435. This is how many constituents each representative should
have (and would have, if Congressional districts were allowed to cross state boundaries).

1. **What do you get if you divide a state’s population by the ideal district size?**
    
    The simplest method of apportionment was proposed in 1790 by Alexander Hamilton, and
    it is so intuitively appealing that you may have thought of it yourself already: Calculate
    each state’s share of the total number of available seats, based on population proportions,
    and give each state as many seats as prescribed by the integer part of its ideal quota. The
    remaining fractional parts of the quotas add up to a whole number of uncommitted seats,
    which are awarded to those states that have the largest fractional parts.
    
    Apply the Hamilton method to the following small, three-state examples. (The names of
    the states are simply A, B, and C.) You should notice some interesting anomalies.
    
2. **Suppose that the populations are A = 453000, B = 442000, and C = 105000, and that there are 100 delegates to be assigned to these states on the basis of their populations.**
3. **Suppose that the populations are A = 453000, B = 442000, and C = 105000, and that there are 101 delegates to be assigned to these states on the basis of their populations.**
4. **Suppose that the populations are A = 647000, B = 247000, and C = 106000, and that there are 100 delegates to be assigned to these states on the basis of their populations.**
5. **Suppose that the populations are A = 650000, B = 255000, and C = 105000, and that there are 100 delegates to be assigned to these states on the basis of their populations.**