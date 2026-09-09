# Week 2 code — SQL on two tables

The in-class demo for **Wednesday, Sep 2, 2026**.

```
code/
├── data/
│   ├── listings.csv    42 used cars — Week 1's cleaned file plus a dealer_id column
│   └── dealers.csv     6 dealerships
└── 02_sql_intro.ipynb  the demo, already executed
```

Run it with nothing installed:

```bash
jupyter lab 02_sql_intro.ipynb
```

`sqlite3` is in the Python standard library, so the only real dependency is pandas.
Everything works identically in Colab — upload the two CSVs and the notebook runs
unchanged.

## Where the data came from

`listings.csv` is `projects/csc381-cars/data/clean/used_cars_clean.csv` — the file
the class cleaned together on Aug 26 — with **one column added**, `dealer_id`.
`dealers.csv` is invented for this class. The towns are real places in Chester and
Delaware counties; the businesses are not, apart from the `CarMax` value that Week
1's synthetic data already contained.

Both are rebuilt by `../_build/make_sql_tables.py`.

## Why two tables, and why they break

The whole lesson is the join, and the join is worth teaching because it is
**dangerous in a way that produces no error message**:

| | rows |
|---|---|
| `listings` | 42 |
| `listings JOIN dealers` (inner) | **22** |
| `listings LEFT JOIN dealers` | 42 |

Twenty listings — 16 private-party and 4 auction sales — have no dealer behind
them, so their `dealer_id` is empty and an inner join drops them silently. A
student who goes straight from that join to `AVG(price)` publishes an average
used-car price that excludes every private seller in the market, and nothing
anywhere tells them so.

It fires from the other side too: `D-06 Route 30 Motors` has zero listings on
purpose, so it vanishes from any inner-joined report — and "the dealer with no
inventory" is often exactly the row somebody needed to see.

This is Week 1's *never drop rows silently* habit, in a different language. That
is the point of the whole night.

## Other things planted in the data

- **`flag_any` survives into SQL.** Week 1 flagged the seven impossible rows
  instead of deleting them; `WHERE flag_any = 1` now pulls them back in one line,
  which is the payoff for that decision. `WHERE flag_any = 0` is used throughout
  so the `$0` and `$1` cars stop dragging the averages down.
- **SQLite has no boolean type**, so `flag_any` comes back as `1`/`0`, not
  `True`/`False`. One column, two ideas about what a value is — the same problem
  as last week, except now the database is doing it to you.
- **`COUNT(*)` vs `COUNT(column)`.** After a left join, Route 30 Motors has one
  row of NULLs: `COUNT(*)` says 1, `COUNT(l.listing_id)` says 0. `COUNT(column)`
  skips NULLs.
- **A count that quietly changes.** In the final combined query Gay Street Auto
  shows 3 cars rather than 4, because `WHERE flag_any = 0` removed one of them.
  Worth pointing at live.

## Rebuilding

```bash
python ../_build/make_sql_tables.py     # the two CSVs
python ../_build/make_sql_notebook.py   # the notebook, executed
```

The notebook builder runs every code cell as it writes the file, so a broken
query fails the build instead of the class. There is no Jupyter installed in this
environment — `../_build/nbwrite.py` is a small notebook writer that executes
cells in-process and stores the real outputs.
