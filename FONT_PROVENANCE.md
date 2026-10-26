# Font provenance

The unmodified Noto fonts in `code/fonts/` were copied from the repository's
existing Arabic Reconciliation Bench font bundle. Its acquisition record says
Noto Project fonts bundled with LibreOffice, with terms verified in each TTF name
table. The original record is retained as `code/fonts/manifest.json`; it lists
the larger original bundle, including a Latin bold file that this project does
not use or redistribute. This project's actual files and hashes are recorded
in `build-manifest.json`.

| File | SHA-256 |
| --- | --- |
| NotoSansArabic-Regular.ttf | d6efa4c9fac8ecefd2de19676acc3a396ba7aee99ba7a2d513950246841b6725 |
| NotoSansArabic-Bold.ttf | 457926cb001ac787dc7415c89e6f5b1c4ce3a0f2c91d0cabc1bc19a05638e7de |
| NotoSans-Regular.ttf | f5f552c8c5edb61fe6efb824baf4d4de47b1a8689ab4925ff43f7bd6a4ebece5 |

Arabic fonts: Copyright 2022 The Noto Project Authors
([Noto Arabic](https://github.com/notofonts/arabic)), version 2.010.
Latin font: Copyright 2022 The Noto Project Authors
([Noto Latin/Greek/Cyrillic](https://github.com/notofonts/latin-greek-cyrillic)),
version 2.015. Font software remains under the original SIL Open Font License
1.1; its full notice is retained in `code/fonts/OFL-1.1.txt`. No font name or file
was modified. Rendered fixture images and original annotations are CC BY 4.0.

Every drawn string is checked against its actual chosen font's Unicode cmap.
Arabic words use RAQM/HarfBuzz shaping. ASCII IDs use Noto Sans. Arabic-Indic
numeric runs use the Arabic font and explicit left-to-right nonjoining glyph
placement, preserving literal ISO date component order without invisible bidi
characters. This is a clean synthetic renderer, not a simulation of all natural
Arabic typography or document bidirectional behavior.
