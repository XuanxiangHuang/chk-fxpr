
def toPBO(fpointer, cnf, pb_constrs, comment=None):
    """
        save encoding into PBO format
    """
    nc = len(cnf.clauses) + len(pb_constrs)
    if comment is not None:
        print('\n'.join(comment), file=fpointer)
        print('*', file=fpointer)
        nc += len(comment)
    else:
        print(f'* #variable= {cnf.nv} #constraint= {nc}', file=fpointer)
        nc += 1

    def __str__(pbc):
        def fmt_lit(wl):
            x, w = wl
            if x > 0:
                return f'{w} x{x}'
            return f'{w} ~x{-x}'

        return (' '.join(map(fmt_lit, zip(c.lhs, c.coefs)))) + f' {c.comp} {c.rhs} ;'

    def fmt_lit(wl):
        x, w = wl
        if x > 0:
            return f'{w} x{x}'
        return f'{w} ~x{-x}'

    for cl in cnf:
        print((' '.join(map(fmt_lit, zip(cl, [1] * len(cl)))) + ' >= 1 ;'), file=fpointer)
    # for c in pb_constrs:
    #    fpointer.write((' '.join(map(fmt_lit, zip(c.lits, c.coefs)))+f' >= {c.bound}\n'))
    for c in pb_constrs:
        print(__str__(c), file=fpointer)

    return nc  # num lines


