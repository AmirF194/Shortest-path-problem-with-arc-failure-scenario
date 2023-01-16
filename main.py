import math
import matplotlib.pyplot as plt
import time
import networkx as nx
from networkx.classes.function import path_weight
def PRIMARYMAIN(G, s, t, Isc):
    ctr, pp, UpperBound, s1, X, TotalCost, Y, pred, S, rp, sp, c, a, p = {}, {}, math.inf, [], {}, \
        {}, {}, {}, {}, {}, {}, {}, [], {}
    for n in G.nodes:
        ctr[n] = 0
    ctr[s] = 1
    # Assign 1 to s1
    s1.append(1)
    pp[f'[{s1[0]}]'] = 0;
    X[str(s1)] = list(Isc.keys())
    TotalCost[str(s1)] = 0;
    Y[str(s1)] = [];
    pred[f'[{s1[0]}]'] = f'[{s1[0]}]'
    LIST = []
    LIST.append(f'[{s1[0]}]')
    aj = returnaj(G, s)
    aj[s1[0]] = [s1]

    # Python program to count number of items
    # in a dictionary value that is a list.
    # count = 0
    # for x in aj:
    #     if isinstance(aj[x], list):
    #         count += len(aj[x])
    # print(count)

    # Calculating Totalcost aj
    for a, values in aj.items():
        for v in values:
            pp[str(v)] = path_weight(G, v, weight="weight")
            vOrderedPair = []
            v1 = v.copy()
            while len(v1) > 1:
                vOrderedPair.append((v1[0], v1[1]))
                vOrderedPair.append((v1[1], v1[0]))
                v1.remove(v1[0])
            if str(v) not in X.keys():
                X[str(v)] = []
            if str(v) not in Y.keys():
                Y[str(v)] = []
            for sc, path in Isc.items():
                checker = True
                for vvv in vOrderedPair:
                    if vvv in path[0]:
                        checker = False
                if checker:
                    X[str(v)].append(sc)
                else:
                    Y[str(v)].append(sc)
            TotalCost[str(v)] = 0
            for ii in X[str(v)]:
                p[ii] = Isc[ii][-1]
                TotalCost[str(v)] += p[ii] * pp[str(v)]
            for iii in Y[str(v)]:
                firstF = returnFirstOccurance(vOrderedPair, Isc[iii][0])
                v_modified = rpPath(v, firstF)
                p[iii] = Isc[iii][-1]
                sp[str(v), iii] = SECONDARY(G.copy(), iii, Isc, firstF[0], v[-1])
                rp[str(v), iii] = path_weight(G, v_modified, weight="weight")
                TotalCost[str(v)] += p[iii] * (rp[str(v), iii] + sp[str(v), iii])
    result = []
    # Main Code
    for v in LIST:
        if LIST and TotalCost[pred[str(v)]] < UpperBound:
            tmp = {}
            for l in LIST:
                tmp[l] = TotalCost[l]
            if len([min(tmp, key=tmp.get)][0]) == 4:
                removed_node = int([min(tmp, key=tmp.get)][0][0])
            else:
                removed_node = 1
            LIST.remove((min(tmp, key=tmp.get)))
            result.append(v)
            if v == min(tmp, key=tmp.get):
                for a, b, w in G.edges(data=True):
                    if a == removed_node and b != 1:
                        tmp, X[b], kkk = [], [], []
                        TotalCost[b] = 0
                        checker = True
                        for sc, path in Isc.items():
                            if (removed_node, b) not in path[0] or (b, removed_node) not in path[0]:
                                tmp.append(sc)
                        X[b] = list(set(X[str(extractpred(v, pred, []))]).intersection(tmp))
                        Y[b] = [x for x in list(Isc.keys()) if (x not in X[b])]
                        S[b] = [x for x in Y[b] if (x not in Y[str(extractpred(v, pred, []))])]
                        for iii in Y[str(extractpred(v, pred, []))]:
                            rp[b, iii] = rp[str(extractpred(v, pred, [])), iii]
                            sp[b, iii] = sp[str(extractpred(v, pred, [])), iii]
                        if S[b]:
                            for i in S[b]:
                                rp[b, i] = pp[str(extractpred(v, pred, []))]
                                sp[b, i] = SECONDARY(G.copy(), i, Isc, removed_node, t)
                        pp[b] = pp[str(extractpred(v, pred, []))] + w['weight']
                        for ii in X[b]:
                            p[ii] = Isc[ii][-1]
                            TotalCost[b] += p[ii] * pp[b]
                        for iii in Y[b]:
                            tmp = extractpred(v, pred, [])
                            if b not in tmp:
                                tmp.append(b)
                            TotalCost[b] += p[iii] * (rp[str(tmp), iii] + sp[str(tmp), iii])
                        for k in aj[b]:
                            X[f'{b}[{aj[b].index(k) + 1}]'] = []
                            TotalCost[f'{b}[{aj[b].index(k) + 1}]'] = TotalCost[str(k)]
                            if X[b] == X[f'{b}[{aj[b].index(k) + 1}]']:
                                checker = False

                        if ctr[b] == 0 or checker and b not in extractpred(v, pred, []):
                            ctr[b] = ctr[b] + 1
                            pred[f'{b}[{ctr[b]}]'] = v
                            LIST.append(f'{b}[{ctr[b]}]')
                            if b == t and TotalCost[b] <= UpperBound:
                                UpperBound = TotalCost[b]
                        for k in LIST:
                            if X[b] == X[pred[k]]:
                                if TotalCost[b] <= TotalCost[pred[k]]:
                                    pred[k] = v
                                    if k not in LIST:
                                        LIST.append(k)
                                    if b == t and TotalCost[b] <= UpperBound:
                                        UpperBound = TotalCost[b]
    return UpperBound
def PRIMARYMAINWithDominanceCriteria(G, s, t, Isc):
    ctr, pp, UpperBound, s1, X, TotalCost, Y, pred, S, rp, sp, c, a, p = {}, {}, math.inf, [], {}, \
        {}, {}, {}, {}, {}, {}, {}, [], {}
    for n in G.nodes:
        ctr[n] = 0
    ctr[s] = 1
    # Assign 1 to s1
    s1.append(1)
    pp[f'[{s1[0]}]'] = 0;
    X[str(s1)] = list(Isc.keys())
    TotalCost[str(s1)] = 0;
    Y[str(s1)] = [];
    pred[f'[{s1[0]}]'] = f'[{s1[0]}]'
    LIST = []
    LIST.append(f'[{s1[0]}]')
    aj = returnaj(G, s)
    aj[s1[0]] = [s1]
    # Calculating Totalcost aj
    for a, values in aj.items():
        for v in values:
            pp[str(v)] = path_weight(G, v, weight="weight")
            vOrderedPair = []
            v1 = v.copy()
            while len(v1) > 1:
                vOrderedPair.append((v1[0], v1[1]))
                vOrderedPair.append((v1[1], v1[0]))
                v1.remove(v1[0])
            if str(v) not in X.keys():
                X[str(v)] = []
            if str(v) not in Y.keys():
                Y[str(v)] = []
            for sc, path in Isc.items():
                checker = True
                for vvv in vOrderedPair:
                    if vvv in path[0]:
                        checker = False
                if checker:
                    X[str(v)].append(sc)
                else:
                    Y[str(v)].append(sc)
            TotalCost[str(v)] = 0
            for ii in X[str(v)]:
                p[ii] = Isc[ii][-1]
                TotalCost[str(v)] += p[ii] * pp[str(v)]
            for iii in Y[str(v)]:
                firstF = returnFirstOccurance(vOrderedPair, Isc[iii][0])
                v_modified = rpPath(v, firstF)
                p[iii] = Isc[iii][-1]
                sp[str(v), iii] = SECONDARY(G.copy(), iii, Isc, firstF[0], v[-1])
                rp[str(v), iii] = path_weight(G, v_modified, weight="weight")
                TotalCost[str(v)] += p[iii] * (rp[str(v), iii] + sp[str(v), iii])
    result = []
    # Main Code
    for v in LIST:
        if LIST and TotalCost[pred[str(v)]] < UpperBound:
            tmp = {}
            for l in LIST:
                tmp[l] = TotalCost[l]
            if len([min(tmp, key=tmp.get)][0]) == 4:
                removed_node = int([min(tmp, key=tmp.get)][0][0])
            else:
                removed_node = 1
            LIST.remove((min(tmp, key=tmp.get)))
            result.append(v)
            if v == min(tmp, key=tmp.get):
                for a, b, w in G.edges(data=True):
                    if a == removed_node and b != 1:
                        tmp, X[b] = [], []
                        TotalCost[b] = 0
                        checker = True
                        for sc, path in Isc.items():
                            if (removed_node, b) not in path[0] or (b, removed_node) not in path[0]:
                                tmp.append(sc)
                        X[b] = list(set(X[str(extractpred(v, pred, []))]).intersection(tmp))
                        Y[b] = [x for x in list(Isc.keys()) if (x not in X[b])]
                        S[b] = [x for x in Y[b] if (x not in Y[str(extractpred(v, pred, []))])]
                        for iii in Y[str(extractpred(v, pred, []))]:
                            rp[b, iii] = rp[str(extractpred(v, pred, [])), iii]
                            sp[b, iii] = sp[str(extractpred(v, pred, [])), iii]
                        if S[b]:
                            for i in S[b]:
                                rp[b, i] = pp[str(extractpred(v, pred, []))]
                                sp[b, i] = SECONDARY(G.copy(), i, Isc, removed_node, t)
                        pp[b] = pp[str(extractpred(v, pred, []))] + w['weight']
                        for ii in X[b]:
                            p[ii] = Isc[ii][-1]
                            TotalCost[b] += p[ii] * pp[b]
                        for iii in Y[b]:
                            tmp = extractpred(v, pred, [])
                            if b not in tmp:
                                tmp.append(b)
                            TotalCost[b] += p[iii] * (rp[str(tmp), iii] + sp[str(tmp), iii])
                        for k in aj[b]:
                            X[f'{b}[{aj[b].index(k) + 1}]'] = []
                            TotalCost[f'{b}[{aj[b].index(k) + 1}]'] = TotalCost[str(k)]
                            if X[b] == X[f'{b}[{aj[b].index(k) + 1}]']:
                                checker = False
                        if ctr[b] == 0 or checker and b not in extractpred(v, pred, []):
                            Flag = 0
                            for k in LIST:
                                if set(X[b]).issubset(set(X[pred[k]])):
                                    if TotalCost[b] <= TotalCost[pred[k]]:
                                        if Flag == 0:
                                            pred[k] = v
                                        if k not in LIST:
                                            LIST.append(k)
                                        Flag = 1
                                        if b == t and TotalCost[pred[k]] <= UpperBound:
                                            UpperBound = TotalCost[pred[k]]
                                    else:
                                        LIST.remove(k)
                            if Flag == 0:
                                ctr[b] = ctr[b] + 1
                                pred[f'{b}[{ctr[b]}]'] = v
                                LIST.append(f'{b}[{ctr[b]}]')
                                if b == t and TotalCost[b] <= UpperBound:
                                    UpperBound = TotalCost[b]
                        for k in LIST:
                            if X[b] == X[pred[k]]:
                                if TotalCost[b] <= TotalCost[pred[k]]:
                                    pred[k] = v
                                    if k not in LIST:
                                        LIST.append(k)
                                    if b == t and TotalCost[b] <= UpperBound:
                                        UpperBound = TotalCost[b]
    return UpperBound
def extractpred(inn, pred, result):
    if len(inn) == 4:
        result.insert(0, int(inn[0]))
    if len(pred[inn]) == 3:
        result.insert(0, 1)
    else:
        extractpred(pred[inn], pred, result)
    return result
def returnFirstOccurance(l1, l2):
    for e in l1:
        if e in l2:
            return e
def rpPath(l, firstF):
    l1 = l[:l.index(firstF[0]) + 1]
    l2 = l[:l.index(firstF[1]) + 1]
    if len(l1) < len(l2):
        return l1
    else:
        return l2
def SECONDARY(G1, iii, I, s, t):
    for m in I[iii][0]:
        if (m[0], m[1]) in G1.edges:
            G1.remove_edge(m[0], m[1])
        if (m[1], m[0]) in G1.edges:
            G1.remove_edge(m[1], m[0])
    # nx.draw(G1, with_labels=True)
    # plt.show()
    try:
        cost = nx.shortest_path_length(G1, source=s, target=t, weight='weight', method='dijkstra')
    except:
        cost = math.pow(10, 100)
    return cost
def returnaj(G, s):
    A = {}
    a = list(G)
    a.remove(s)
    paths = list(nx.all_simple_paths(G, source=s, target=a))
    for path in paths:
        if path[-1] not in A.keys():
            A[path[-1]] = []
        A[path[-1]].append(path)
    return A
G = nx.complete_graph(12)
G = G.to_directed()
for a,b in G.edges():
    G[a][b]['weight'] = random.randint(1, G.number_of_nodes())

Isc = {
    "sc1": [[(1, 2), (2, 4), (2, 1), (4, 2)], 0.2],
    "sc2": [[(3, 5), (5, 3)], 0.1],
    "sc3": [[(2, 5), (1, 4), (5, 2), (4, 1), (1, 7), (7, 1), (2, 6), (6, 2)], 0.7]
}
s = 1
t = 8
start_time = time.time()
print( PRIMARYMAIN(G, s, t, Isc))
print((time.time() - start_time))

start_time = time.time()
print( PRIMARYMAINWithDominanceCriteria(G, s, t, Isc))
print((time.time() - start_time))