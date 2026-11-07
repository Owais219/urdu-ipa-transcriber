# Transcription standard

Default profile: **Standard Pakistani Urdu** — educated urban speech, careful single-word citation.

## Three representations

| Layer | Notation | Where it comes from |
|---|---|---|
| Orthographic | Urdu script | Speech recogniser (Whisper/MMS), or the researcher's *expected word* |
| Phonemic | `/…/` | Lexicon citation form for the word. Shown **only** when the word is in the lexicon. It is the dictionary form, not a claim about this token. |
| Broad phonetic (primary output) | `[…]` | The candidate realisation best supported by *this recording*, after acoustic scoring and measurement. |
| Narrow phonetic (optional) | `[…]` | Shown only when every segment is acoustically supported; adds allophonic detail the model can resolve ([ɦ] for /h/, [ɾ] for /r/). |

## Symbol conventions

**Consonants**

| | Bilabial | Labiodental | Dental | Alveolar | Retroflex | Post-alv./Palatal | Velar | Uvular | Glottal |
|---|---|---|---|---|---|---|---|---|---|
| Stop/affricate | p pʰ b bʱ | | t̪ t̪ʰ d̪ d̪ʱ | | ʈ ʈʰ ɖ ɖʱ | t͡ʃ t͡ʃʰ d͡ʒ d͡ʒʱ | k kʰ ɡ ɡʱ | q | (ʔ) |
| Nasal | m | | | n | (ɳ) | | (ŋ) | | |
| Fricative | | f | | s z | | ʃ ʒ | x ɣ | | h (ɦ) |
| Flap/trill | | | | r | ɽ ɽʱ | | | | |
| Approximant | | ʋ | | l | | j | | | |

* Dentals always carry the dental diacritic (t̪, d̪). /n s z r l/ are written without it.
* Voiceless aspiration: ʰ. Voiced (breathy) aspiration: ʱ.
* Affricates use the tie bar: t͡ʃ, d͡ʒ.
* Gemination: length mark on the consonant. Aspirated geminates are written base + ː + ʰ/ʱ, e.g. `ət͡ʃːʰɑː` (اچھا), `məkːʰiː` (مکھی). The tokenizer treats `t͡ʃʰː` and `t͡ʃːʰ` as the same segment.
* ɳ and ŋ are treated as positional allophones of /n/ before retroflex and velar stops: ٹھنڈا is `/ʈʰənɖɑː/` phonemically and `[ʈʰəɳɖɑː]` phonetically.
* ع and ح: not realised as pharyngeals in this profile (عید /iːd̪/, حال /hɑːl/). ħ and ʕ exist in the inventory for other profiles.
* /h/ is phonemic; its common voiced realisation [ɦ] appears only in narrow output.

**Vowels**

| | Front | Central | Back |
|---|---|---|---|
| High | iː ɪ | | uː ʊ |
| Mid | eː | ə | oː |
| Low-mid | ɛː | | ɔː |
| Low | | | ɑː ɑ |

* The specification's vowel list omitted /eː/ and /oː/; they are contrastive in Urdu (بیٹا `beːʈɑː`, دو `d̪oː`) and are included.
* Nasalised vowels: combining tilde before the length mark (`ɑ̃ː`, `ĩː`, `ə̃`).
* Short ɛ and ɔ are in the inventory for allophonic/regional use only.
* **Word-final ہ after a consonant** (جرگہ, بچّہ, کمرہ) is a short **[ɑ]**, distinct from final ا = [ɑː] (کھانا [kʰɑːnɑː]).
  Whether a breathy release follows is measured in every recording: if at least 50 ms of
  voiceless noise follows the end of voicing, the output is [d͡ʒɪrɡɑh]; if 20 ms or less, [d͡ʒɪrɡɑ];
  in between, the segment is marked unresolved. Lexicon entries are written without the [h]
  (`d͡ʒɪrɡɑ`), so the phonemic form does not claim it. ہ after a vowel (شاہ, راہ) is an ordinary /h/.
* Word-final ھ is never a vowel: it marks aspiration of the preceding consonant (ہاتھ [hɑːt̪ʰ], آٹھ [ɑːʈʰ]).

## Rules the software follows

* It never writes Romanisation or English-style respellings.
* Every symbol output belongs to the inventory above (`ipa.validate`).
* It does not normalise away aspiration, retroflexion, length or nasalisation. When the evidence cannot decide one of these, the segment is flagged `unresolved` or `model_blind`, and the whole word is refused if too many segments are undecided.
* A realisation that differs from the dictionary (e.g. [kələm] for قلم) is reported as an **acoustic variant**, never corrected to the dictionary form.
