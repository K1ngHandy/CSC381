# Homework 1 — Clean a messy file

**CSC 381/576 · Data Science · Fall 2026**

**Due Monday, September 14, 11:59 PM, on D2L** · 100 points · about two hours

---

## The file

`philly_rentals_raw.csv` — 30 apartment listings from five Philadelphia
neighborhoods, plus one row that is not a listing. Columns: `listing_id`,
`neighborhood`, `bedrooms`, `bathrooms`, `sqft`, `rent`.

**There are five things wrong with it.** Every one is something we fixed in class
on August 26, on a different column. There are no unit conversions, no date
formats, and no duplicate rows — don't go looking for them. Never edit the raw
file; every change lives in your code.

---

## Part 1 — Find four things wrong (20 points)

Before you change anything, look at the file and write down what is wrong with it.
Diagnosis first, treatment second — **do not fix anything in this part.**

1. **Load it and look at it.** Run `df.head(10)`, `df.tail(3)`, `df.info()`, and
   `df['neighborhood'].value_counts()` — and the same `value_counts` on the other
   text columns. Those four calls show you everything you need for this part.
2. **Add a markdown cell to your notebook** titled *Part 1 — what is wrong with
   this file*, and list four problems in it, numbered 1 to 4.
3. **For each of the four, write three things:** (a) which column or row it is in,
   (b) what is wrong with it, in one sentence, and (c) which of the six categories
   below it belongs to.
4. **Show what made you notice.** Under each problem, point at the output that
   gave it away — the `value_counts`, the dtype, the row you spotted. One line is
   plenty.

> **The six categories:** type problems · unit problems · encoding drift ·
> missing values · impossible values · structural junk

**A complete answer looks like this:**

> ***2. `rent` — type problem.** The column holds `$1,450`, `1750` and `1,900` in
> three different formats, so pandas read the whole column as text instead of
> numbers. `df.info()` reports it as `object`, and `df['rent'].mean()` fails.*

**Two rules.** Four is enough — there are five, you do not have to find them all.
And the four have to be four *different* problems: "rent has dollar signs" and
"rent has commas" are one problem written twice, not two.

---

## Part 2 — Fix them (45 points)

In this order:

1. **Drop the row that is not a listing.** Print the row count before and after:
   **31 rows in, 30 out.**
2. **Make `rent` a real number.** Check: `df['rent'].dtype` is no longer `object`.
3. **Clean `neighborhood`.** Check: you end up with five neighborhoods, not eight.
4. **Find every missing `bedrooms` value.** This file says "we do not know" in
   three different ways and pandas only recognises two of them, so `isna()`
   under-reports. Run `df['bedrooms'].value_counts()` to see the third. Turn all
   three into real missing values with `replace()`, then make the column numeric.

**Check:** `df['bedrooms'].isna().sum()` should now say 3. Filling them in is
Part 3.

Then save: `df.to_csv('rentals_clean.csv', index=False)` — after Part 3, so the
file you hand in has the bedrooms filled in.

---

## Part 3 — Fill in the three missing bedroom counts (20 points)

Fill each of the three with **the median number of bedrooms for that listing's own
neighborhood** — not the median of the whole file. One line does it:

```python
df['bedrooms'] = df['bedrooms'].fillna(
    df.groupby('neighborhood')['bedrooms'].transform('median')
)
```

**Read it from the inside out.** `groupby('neighborhood')['bedrooms']` splits the
column into five neighborhood-sized pieces. `transform('median')` takes the median
of each piece and hands back a column the same length as the DataFrame, holding
that row's own neighborhood median. `fillna()` then copies a value across only
where bedrooms is missing, and leaves every other row alone.

**Check your work.** `df['bedrooms'].isna().sum()` is now 0, and the three rows
you filled are exactly these:

| Listing | Neighborhood | bedrooms after filling |
|---|---|---|
| `R-2007` | Manayunk | **2** |
| `R-2014` | University City | **1** |
| `R-2020` | Old City | **2** |

**If all three came out as 2**, you filled with the median of the whole file
instead of the neighborhood. University City is a student neighborhood full of
one-bedrooms, and its median is 1. That is the entire reason we grouped first.

---

## Part 4 — Two SQL queries (15 points)

Covered in class on Wednesday, September 2. Paste this in — there is nothing to
install:

```python
import sqlite3
con = sqlite3.connect(':memory:')
df.to_sql('listings', con, index=False)

def q(sql):
    return pd.read_sql(sql, con)
```

**Q1.** How many listings are in each neighborhood, most to fewest?
*Your answer has five rows.*

**Q2.** What is the average rent of the two-bedroom listings?
*Your answer is one number.*

---

## What to submit

Two files on D2L:

| File | Notes |
|---|---|
| `rentals_clean.csv` | Written by your code, not edited by hand in Excel. |
| `HW1_yourlastname.pdf` | Your notebook, printed to PDF with all output showing. |

Run everything from the top before you print, so the output in the PDF matches the
code above it. **Colab:** File → Print → Save as PDF. **VS Code or Jupyter:**
Ctrl-P in the browser preview, then Save as PDF.

**Everything I grade is in the PDF**, so check that your outputs are actually
visible in it before you upload.

---

**Grading** Part 1 (20) · Part 2 (45) · Part 3 (20) · Part 4 (15).
**Late work** 10% per day, up to 3 days.
**AI** is allowed; you must be able to explain any line you submit, and I may ask
you to.
**Stuck?** This should take about two hours. If it is taking four, post on D2L —
that is my problem to fix, not yours to suffer through.
