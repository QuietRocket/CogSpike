// Shared Typst style for the Nengo spiking-compression experiment reports.
// Import in each report:  #import "../report_style.typ": *

#let setup(doc) = {
  set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
  set text(font: "New Computer Modern", size: 10.5pt)
  set par(justify: true)
  set heading(numbering: "1.")
  set math.equation(numbering: "(1)")
  doc
}

#let blockbox(fill, stroke, body) = block(width: 100%, inset: 8pt, fill: fill,
  stroke: (left: 2pt + stroke), radius: 2pt, body)

#let claim(body) = blockbox(rgb("#eef3fb"), rgb("#4a90d9"), [📌 *Claim under test.* #body])
#let finding(body) = blockbox(rgb("#eefbf0"), rgb("#3a9d5d"), [✅ *Finding.* #body])
#let honest(body) = blockbox(rgb("#fff7f0"), rgb("#d98a4a"), [⚖️ *Honesty check.* #body])
#let intuition(body) = blockbox(rgb("#f0f7ff"), rgb("#4a90d9"), [💡 *Intuition.* #body])
#let gap(body) = blockbox(rgb("#f5f0ff"), rgb("#8a6ad9"), [🔬 *Spiking-reality gap.* #body])
#let method(body) = blockbox(luma(247), luma(150), [🛠 *Method.* #body])

// Standard report header.
#let report_header(id, title, subtitle) = {
  align(center)[
    #text(size: 9pt, fill: luma(110))[CogSpike · research/compression/nengo_experiments · #id]
    #v(0.2em)
    #text(size: 15pt, weight: "bold")[#title]
    #v(0.2em)
    #text(size: 10.5pt, style: "italic")[#subtitle]
  ]
  v(0.4em)
}

// A PASS/FAIL acceptance row table.
#let accept_table(rows) = table(
  columns: (auto, 1fr),
  align: (center, left),
  stroke: 0.5pt + luma(190),
  table.header[*verdict*][*acceptance check*],
  ..rows.map(r => (
    if r.at(0) { text(fill: rgb("#2a8a4a"))[*PASS*] } else { text(fill: rgb("#c0392b"))[*FAIL*] },
    r.at(1),
  )).flatten()
)
