import random
def parent_selection(pop,m):
    try:
        ranks = [i for i in range(len(pop))]
        probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
        parents = random.choices(pop, weights=probs, k=m)
    except:
        import pdb; pdb.set_trace()
    return parents