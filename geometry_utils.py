# Middle line between two parallel lines.
def middle_line(l1, l2):
    a1, b1, d1 = l1
    a2, b2, d2 = l2

    am = (a1 + a2) / 2.0
    bm = (b1 + b2) / 2.0
    dm = (d1 + d2) / 2.0
    return am, bm, dm
