#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Render NOTICE from NOTICE.template for the drafter lineage that ships.

    make_notice.py --lineage clean --drafter-name <name> [--template NOTICE.template] [--out NOTICE]

The template holds one data-attribution block per lineage between BEGIN:/END: markers. The only lineage is 'clean'
(permissively licensed sources: Tatoeba CC BY 2.0 FR, OASST1/2 Apache-2.0, e-Gov statutes, Aozora Bunko and Project
Gutenberg public domain). Nothing about a drafter version is hard-coded: give the package name that release.json records.
"""
import argparse
import os
import re
import sys

LINEAGES = ('clean',)


def render(template, lineage, drafter_name):
    if lineage not in LINEAGES:
        raise ValueError('lineage must be one of %s' % (LINEAGES,))
    out = template
    for other in LINEAGES:
        if other != lineage:
            out = re.sub(r'<!-- BEGIN:%s -->.*?<!-- END:%s -->\n?' % (other, other), '', out, flags=re.S)
    out = re.sub(r'<!-- (?:BEGIN|END):%s -->\n?' % lineage, '', out)
    out = out.replace('@DRAFTER_NAME@', drafter_name)
    if '@DRAFTER_NAME@' in out:
        raise ValueError('unreplaced placeholder')
    return out


def main(argv=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lineage', choices=LINEAGES, required=True)
    ap.add_argument('--drafter-name', required=True)
    ap.add_argument('--template', default=os.path.join(here, 'NOTICE.template'))
    ap.add_argument('--out', default=os.path.join(here, 'NOTICE'))
    a = ap.parse_args(argv)
    with open(a.template, encoding='utf-8') as f:
        text = render(f.read(), a.lineage, a.drafter_name)
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print('wrote %s (%s lineage, drafter %s)' % (a.out, a.lineage, a.drafter_name))
    return 0


if __name__ == '__main__':
    sys.exit(main())
