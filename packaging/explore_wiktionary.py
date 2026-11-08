"""Print statistics and samples of Urdu IPA in the kaikki.org Wiktionary extract."""
import collections, json, random, sys, unicodedata
path = sys.argv[1]
n = with_ipa = 0
tags = collections.Counter(); syms = collections.Counter(); samples = []
for line in open(path, encoding="utf-8"):
    e = json.loads(line); n += 1
    ipas = [s for s in e.get("sounds", []) if s.get("ipa")]
    if not ipas:
        continue
    with_ipa += 1
    for s in ipas:
        tags[tuple(s.get("tags", []))] += 1
        for ch in unicodedata.normalize("NFD", s["ipa"]):
            syms[ch] += 1
    samples.append((e.get("word"), e.get("pos"), [(s["ipa"], s.get("tags")) for s in ipas]))
print("entries", n, "with ipa", with_ipa, "distinct words with ipa", len({w for w, _, _ in samples}))
print("tags", tags.most_common(15))
print("symbols", " ".join(f"{c}(U+{ord(c):04X}):{k}" for c, k in syms.most_common(90)))
random.seed(1)
for w in ["سلام", "بچہ", "جرگہ", "کمرہ", "اچھا", "ہاتھ", "ماں", "نہیں", "پتا", "محبت", "قلم", "عید", "ٹھنڈا", "رنگ", "کتاب", "دوست", "زندگی"]:
    print("CHECK", w, [s[2] for s in samples if s[0] == w][:2])
for s in random.sample(samples, 60):
    print("SAMPLE", s)
