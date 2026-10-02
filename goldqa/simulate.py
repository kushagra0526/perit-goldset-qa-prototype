"""Synthetic gold sets in Perit's 9 locales + simulated annotators with planted drift. All made up."""
from __future__ import annotations

import random

from .score import FILLERS

# short, call-centre style sentences; CJK pre-segmented into words with spaces
BANK = {
    "de": ["ich möchte meine bestellung ändern", "die lieferung kam gestern nicht an", "können sie mir die rechnung schicken",
           "mein konto wurde zweimal belastet", "wann öffnet die filiale morgen"],
    "fr": ["je voudrais changer mon rendez-vous", "le colis n'est pas encore arrivé", "pouvez-vous m'envoyer la facture",
           "mon compte a été débité deux fois", "à quelle heure ouvre l'agence demain"],
    "it": ["vorrei cambiare il mio ordine", "il pacco non è ancora arrivato", "potete mandarmi la fattura",
           "mi avete addebitato due volte", "a che ora apre la filiale domani"],
    "es": ["quiero cambiar mi pedido", "el paquete no ha llegado todavía", "me puede enviar la factura",
           "me cobraron dos veces este mes", "a qué hora abre la oficina mañana"],
    "pt": ["quero mudar o meu pedido", "a encomenda ainda não chegou", "pode me enviar a fatura",
           "fui cobrado duas vezes este mês", "que horas abre a loja amanhã"],
    "hi": ["मुझे अपना ऑर्डर बदलना है", "पार्सल अभी तक नहीं आया", "क्या आप बिल भेज सकते हैं",
           "मेरे खाते से दो बार पैसे कटे", "कल दुकान कितने बजे खुलेगी"],
    "zh": ["我 想 修改 我的 订单", "包裹 还 没有 到", "你 能 把 发票 发给 我 吗", "我的 账户 被 扣 了 两次", "明天 门店 几点 开门"],
    "ja": ["注文 を 変更 したい です", "荷物 が まだ 届いて いません", "請求書 を 送って もらえ ます か",
           "二回 引き落とさ れ ました", "明日 は 何時 に 開き ます か"],
    "ko": ["주문 을 변경 하고 싶어요", "택배 가 아직 안 왔어요", "청구서 를 보내 주실 수 있나요", "두 번 결제 되었어요", "내일 몇 시 에 문 을 열어요"],
}

ANNOTATORS = {  # id -> (locale, behaviour)
    "A01": ("de", "clean"), "A02": ("fr", "clean"), "A03": ("es", "drops_fillers_w5"), "A04": ("it", "clean"),
    "A05": ("pt", "ts_offset_w3"), "A06": ("hi", "tag_variants_w4"), "A07": ("zh", "cjk_segmentation"),
    "A08": ("ja", "clean"), "A09": ("ko", "clean"), "A10": ("de", "careless_w6"), "A11": ("fr", "clean"), "A12": ("hi", "clean"),
}
WEEKS = 8
ITEMS_PER_WEEK = 10


def gold_item(rng: random.Random, locale: str) -> tuple[list[dict], int]:
    words = []
    for sent in rng.sample(BANK[locale], 5):
        ws = sent.split()
        if rng.random() < 0.7:
            ws.insert(rng.randrange(len(ws)), rng.choice(sorted(FILLERS[locale])))
        words += ws
        words.append(rng.choice(["[pause]", "[pause]", "[overlap]", "[keyboard]", None]))
    t, toks = 300, []
    for w in words:
        if w is None:
            continue
        dur = rng.randint(400, 900) if w.startswith("[") else rng.randint(180, 450)
        toks.append({"w": w, "s": t, "e": t + dur})
        t += dur + rng.randint(20, 120)
    return toks, t + 400


def annotate(rng: random.Random, gold: list[dict], locale: str, behaviour: str, week: int) -> list[dict]:
    p_sub = 0.004
    if behaviour == "careless_w6" and week >= 6:
        p_sub = 0.14
    offset = 120 if behaviour == "ts_offset_w3" and week >= 3 else 0
    out = []
    vocab = [w for s in BANK[locale] for w in s.split()]
    for t in gold:
        w = t["w"]
        if w in FILLERS[locale] and behaviour == "drops_fillers_w5" and week >= 5 and rng.random() < 0.85:
            continue
        if not w.startswith("[") and rng.random() < p_sub:
            w = rng.choice(vocab)
        if w.startswith("[") and behaviour == "tag_variants_w4" and week >= 4:
            w = {"[pause]": "(pause)", "[overlap]": "[Overlap]"}.get(w, w)
        jitter = rng.gauss(0, 12)
        out.append({"w": w, "s": max(0, int(t["s"] + offset + jitter)), "e": max(0, int(t["e"] + offset + jitter))})
    if behaviour == "cjk_segmentation":          # writes 2-word compounds as one word: same characters, different spaces
        merged, i = [], 0
        while i < len(out):
            a = out[i]
            if i + 1 < len(out) and not a["w"].startswith("[") and not out[i + 1]["w"].startswith("[") and rng.random() < 0.5:
                b = out[i + 1]
                merged.append({"w": a["w"] + b["w"], "s": a["s"], "e": b["e"]})
                i += 2
            else:
                merged.append(a)
                i += 1
        out = merged
    return out


def generate(seed: int = 7):
    """-> list of dicts {annotator, locale, week, item, gold, hyp, duration}"""
    rng = random.Random(seed)
    rows = []
    for week in range(1, WEEKS + 1):
        for ann, (loc, beh) in ANNOTATORS.items():
            for k in range(ITEMS_PER_WEEK):
                gold, dur = gold_item(rng, loc)
                rows.append({"annotator": ann, "locale": loc, "week": week, "item": f"{loc}-w{week}-{ann}-{k}",
                             "gold": gold, "hyp": annotate(rng, gold, loc, beh, week), "duration": dur})
    return rows
