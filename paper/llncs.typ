// LLNCS-style template for Typst
// Replicates Springer's Lecture Notes in Computer Science formatting
// Based on llncs.cls v2.24 (2024/01/29)

// ── Helper: create an institute ──────────────────────────────────────────────
#let institute(name, addr: none, email: none, url: none) = (
  name: name,
  addr: addr,
  email: email,
  url: url,
)

// ── Helper: create an author ─────────────────────────────────────────────────
#let author(name, insts: (), orcid: none) = (
  name: name,
  insts: insts,
  orcid: orcid,
)

// ── Theorem-like environment factory ─────────────────────────────────────────
// `kind` controls the counter grouping; `head-style` / `body-style` control fonts.

#let _theorem-counter = counter("llncs-theorem")
#let _definition-counter = counter("llncs-definition")

#let theorem-env(name, cnt: _theorem-counter, numbered: true, head-style: strong, body-style: emph) = {
  // Returns a function (body, supplement: none) => content
  (body, supplement: none) => {
    if numbered { cnt.step() }
    v(7pt)
    block(width: 100%, {
      if numbered {
        context {
          let num = cnt.get().first()
          head-style[#name #num]
          if supplement != none {
            head-style[ ]
            head-style[(#supplement)]
          }
          [. ]
        }
      } else {
        head-style[#name]
        if supplement != none {
          head-style[ ]
          head-style[(#supplement)]
        }
        [. ]
      }
      body-style(body)
    })
    v(3pt)
  }
}

// Pre-defined environments
#let theorem = theorem-env("Theorem", head-style: strong, body-style: emph)
#let lemma = theorem-env("Lemma", head-style: strong, body-style: emph)
#let corollary = theorem-env("Corollary", head-style: strong, body-style: emph)
#let proposition = theorem-env("Proposition", head-style: strong, body-style: emph)
#let definition = theorem-env("Definition", cnt: _definition-counter, head-style: strong, body-style: emph)

#let remark = theorem-env("Remark", head-style: emph, body-style: it => it)
#let example = theorem-env("Example", head-style: emph, body-style: it => it)
#let note = theorem-env("Note", head-style: emph, body-style: it => it)

#let proof(body) = {
  v(7pt)
  block(width: 100%, {
    emph[Proof.]
    [ ]
    body
    h(1fr)
    $square.stroked$
  })
  v(3pt)
}


// ── Main document function ───────────────────────────────────────────────────

#let lncs(
  title: "Contribution Title",
  running-title: none,
  subtitle: none,
  authors: (),
  running-authors: none,
  abstract: none,
  keywords: (),
  acknowledgments: none,
  disclosure: none,
  bib: none,
  body,
) = {
  // ── Collect unique institutes ──────────────────────────────────────────────
  let all-insts = ()
  for a in authors {
    for inst in a.insts {
      let found = false
      for existing in all-insts {
        if existing.name == inst.name { found = true }
      }
      if not found { all-insts.push(inst) }
    }
  }

  // Helper: get 1-based index for an institute
  let inst-index(inst) = {
    for (i, existing) in all-insts.enumerate() {
      if existing.name == inst.name { return i + 1 }
    }
    return 0
  }

  // ── Running head strings ───────────────────────────────────────────────────
  let run-title = if running-title != none { running-title } else { title }
  let run-authors = if running-authors != none {
    running-authors
  } else if authors.len() <= 2 {
    authors.map(a => a.name).join(" and ")
  } else {
    authors.first().name + " et al."
  }

  // ── Page setup ─────────────────────────────────────────────────────────────
  // llncs.cls: \textwidth = 12.2cm, \textheight = 19.3cm
  // \oddsidemargin = \evensidemargin = 63pt ≈ 2.21cm
  // \headsep = 16pt
  // On A4 (21cm × 29.7cm): left/right margin = (21 - 12.2) / 2 = 4.4cm
  // top margin:  1in (2.54cm) + 0pt (\topmargin default) + 12pt (\headheight) + 16pt (\headsep) ≈ 3.52cm
  // bottom margin: 29.7 - 3.52 - 19.3 = 6.88cm (but LaTeX also has 1in offset)
  // Actual LaTeX: top text starts at ~5.2cm from top edge, so top margin ≈ 5.2cm
  // Let's match empirically: A4 paper with the correct text block centered

  set page(
    paper: "a4",
    margin: (
      left: 4.4cm,
      right: 4.4cm,
      top: 5.1cm,
      bottom: 5.3cm,
    ),
    header: context {
      let page-num = counter(page).get().first()
      if page-num > 1 {
        set text(size: 9pt)
        if calc.odd(page-num) {
          h(1fr)
          run-title
          h(1.166cm)
          str(page-num)
        } else {
          str(page-num)
          h(1.166cm)
          run-authors
          h(1fr)
        }
      }
    },
    header-ascent: 16pt,
    numbering: none, // we handle page numbers in the header manually
  )

  // ── Font ───────────────────────────────────────────────────────────────────
  set text(
    font: "New Computer Modern",
    size: 10pt,
    lang: "en",
  )

  // French spacing (uniform word spacing)
  set text(spacing: 100%)

  // ── Paragraph ──────────────────────────────────────────────────────────────
  set par(
    justify: true,
    first-line-indent: (amount: 1.5em, all: false),
    leading: 0.55em,
  )

  // ── Headings ───────────────────────────────────────────────────────────────
  // secnumdepth=2: only section (level 1) and subsection (level 2) are numbered
  // Subsubsection (level 3): 10pt bold, run-in, NOT numbered
  // Paragraph (level 4): 10pt italic, run-in, NOT numbered

  set heading(numbering: (..nums) => {
    let n = nums.pos()
    if n.len() <= 2 {
      numbering("1.1", ..n)
    }
    // level 3+ gets no number (secnumdepth=2)
  })

  show heading.where(level: 1): it => {
    v(18pt)
    block({
      set text(size: 12pt, weight: "bold")
      if it.numbering != none {
        counter(heading).display(it.numbering)
        h(0.7em)
      }
      it.body
    })
    v(12pt)
  }

  show heading.where(level: 2): it => {
    v(18pt)
    block({
      set text(size: 10pt, weight: "bold")
      if it.numbering != none {
        counter(heading).display(it.numbering)
        h(0.7em)
      }
      it.body
    })
    v(8pt)
  }

  // Level 3 — run-in bold heading (NOT numbered, per secnumdepth=2)
  show heading.where(level: 3): it => {
    v(18pt)
    {
      set text(size: 10pt, weight: "bold")
      it.body
      [. ]
    }
  }

  // Level 4 — run-in italic heading (unnumbered)
  show heading.where(level: 4): it => {
    v(12pt)
    {
      set text(size: 10pt, weight: "regular", style: "italic")
      it.body
      [. ]
    }
  }

  // ── Figures & Tables ───────────────────────────────────────────────────────
  // LaTeX \@makecaption: small font, bold "Fig. N." / "Table N.",
  // short captions centered, long captions justified (left-aligned)
  // Figure: \figurename\thinspace\thefigure  → "Fig." + thinspace + number
  // Table:  \tablename~\thetable              → "Table" + ~ + number

  show figure.caption: it => {
    set text(size: 9pt)
    let label-text = if it.kind == table {
      strong({
        it.supplement
        [~]
        context it.counter.display(it.numbering)
        [.]
      })
    } else {
      strong({
        it.supplement
        h(0.16667em) // \thinspace = 1/6 em
        context it.counter.display(it.numbering)
        [.]
      })
    }
    // Replicate LaTeX logic: if label+body fits on one line → center;
    // otherwise → left-aligned justified paragraph
    layout(size => {
      let full = label-text + [ ] + it.body
      let m = measure(full)
      if m.width <= size.width {
        align(center, full)
      } else {
        label-text
        [ ]
        it.body
      }
    })
  }

  // Set the figure supplement names
  set figure(supplement: [Fig.])
  set figure.caption(separator: none)

  // Figures: caption below (default). Tables: caption above with "Table" supplement.
  show figure.where(kind: table): set figure(supplement: [Table])
  show figure.where(kind: table): set figure.caption(position: top)

  // ── Bibliography ───────────────────────────────────────────────────────────
  set bibliography(style: "springer-lecture-notes-in-computer-science")

  // ── Equations ──────────────────────────────────────────────────────────────
  set math.equation(numbering: "(1)")

  // ── Lists ──────────────────────────────────────────────────────────────────
  set list(marker: [--])
  set enum(numbering: "1.")

  // ── Footnotes ──────────────────────────────────────────────────────────────
  // LaTeX: \footnoterule is 2cm wide
  set footnote.entry(separator: line(length: 2cm, stroke: 0.5pt))

  // ── Table column spacing ───────────────────────────────────────────────────
  // LaTeX: \tabcolsep = 1.4pt
  set table(column-gutter: 2.8pt) // tabcolsep is per-side, so total gap = 2×1.4pt

  // ═══════════════════════════════════════════════════════════════════════════
  // TITLE BLOCK
  // ═══════════════════════════════════════════════════════════════════════════
  {
    set par(first-line-indent: 0pt)

    // Title
    align(center, {
      set text(size: 14pt, weight: "bold")
      title
    })

    // Subtitle
    if subtitle != none {
      v(-0.65cm + 0.8cm)
      align(center, {
        set text(size: 12pt, weight: "bold")
        subtitle
      })
    }

    v(0.8cm)

    // Authors
    align(center, {
      for (ai, a) in authors.enumerate() {
        a.name
        // Superscript institute numbers
        if a.insts.len() > 0 {
          let indices = a.insts.map(inst => str(inst-index(inst)))
          super(indices.join(","))
        }
        // ORCID
        if a.orcid != none {
          super[\[#a.orcid\]]
        }
        // Separator
        if ai < authors.len() - 2 {
          [, ]
        } else if ai == authors.len() - 2 {
          [ and ]
        }
      }
    })

    v(0.35cm)

    // Institutes
    {
      set text(size: 9pt)
      set par(leading: 0.45em)
      align(center, {
        for (ii, inst) in all-insts.enumerate() {
          if all-insts.len() > 1 {
            super(str(ii + 1))
            [ ]
          }
          if inst.addr != none {
            inst.name
            [, ]
            inst.addr
          } else {
            inst.name
          }
          if inst.email != none {
            linebreak()
            raw(inst.email)
          }
          if inst.url != none {
            linebreak()
            link(inst.url)
          }
          if ii < all-insts.len() - 1 {
            linebreak()
          }
        }
      })
    }

    v(0.5cm)

    // Abstract
    if abstract != none {
      pad(left: 1cm, right: 1cm, {
        set text(size: 9pt)
        set par(justify: true, first-line-indent: 0pt)
        strong[Abstract.]
        [ ]
        abstract

        // Keywords
        if keywords.len() > 0 {
          v(1em)
          strong[Keywords:]
          [ ]
          keywords.join(" · ")
        }
      })
    }

    v(0.8cm)
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // BODY
  // ═══════════════════════════════════════════════════════════════════════════
  body

  // ═══════════════════════════════════════════════════════════════════════════
  // CREDITS
  // ═══════════════════════════════════════════════════════════════════════════
  if acknowledgments != none or disclosure != none {
    v(12pt)
    set text(size: 9pt)
    set par(first-line-indent: 0pt)

    if acknowledgments != none {
      {
        strong[Acknowledgments.]
        [ ]
        acknowledgments
      }
      v(8pt)
    }

    if disclosure != none {
      {
        strong[Disclosure of Interests.]
        [ ]
        disclosure
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // BIBLIOGRAPHY
  // ═══════════════════════════════════════════════════════════════════════════
  if bib != none {
    v(12pt)
    set text(size: 9pt)
    bib
  }
}
