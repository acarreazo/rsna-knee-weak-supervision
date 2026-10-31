"""02 — Extractor de etiquetas débiles a partir del reporte radiológico (reglas multilingües).

Diseño (ver 01_decisiones/D003):
  * Normalización: minúsculas + eliminación de diacríticos (NFKD), igual para texto y léxico.
  * Segmentación en cláusulas (punto, punto y coma, salto de línea, "pero/but/aber/...").
  * Para cada hallazgo se buscan términos en ~11 idiomas (en, de, es, pt, fr, nl, it, tr, el, ru).
      - Hallazgos "autocontenidos" (derrame, sinovitis, Baker, contusión, fractura, artrosis):
        el término ya es el hallazgo.
      - Hallazgos "estructura + patología" (LCA, LCM, meniscos): se exige término de anatomía y un
        término de patología (rotura, desgarro, Riss, déchirure, разрыв, ...) a <= 8 tokens.
      - Artrosis: se asigna al compartimento (medial / lateral / patelofemoral) más cercano en la
        cláusula; si no se nombra compartimento, se reparte como "posible" a los tres.
  * Negación tipo NegEx: claves previas (no, sin, kein, pas de, geen, δεν, без, ...) hasta 6 tokens
    antes, y claves posteriores (izlenmedi, nicht nachweisbar, не выявлено, intact, ...) hasta 5
    tokens después. Las claves de duda (posible, fraglich, suspected, ...) tienen prioridad.
  * Severidad: leve / moderado / severo modulan el puntaje -> la salida es ordinal, no binaria,
    lo que mejora el AUC (el AUC solo mira el orden).

Salida por estudio y hallazgo:
  score::<label>   puntaje ordinal en [0, 1]   (0.05 negado ... 0.95 severo)
  conf::<label>    confianza en [0, 1]          (se usa para ponderar la pérdida)
  weak::<label>    etiqueta binaria = score >= 0.5
  ev::<label>      tipo de evidencia: pos / neg / duda / ambiguo / no_mencionado

Archivo: 05_datos/etiquetas_debiles_reglas.csv
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

import config as C

log = C.Logger("02_extractor_etiquetas_debiles")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.replace("œ", "oe").replace("æ", "ae").replace("ß", "ss").replace("’", "'")


# ---------------------------------------------------------------------------------------------
# Léxico. Cada entrada es una expresión regular (sobre texto normalizado). Se compila con
# (?<!\w) al inicio para que funcione como "prefijo de palabra" (captura flexiones).
# ---------------------------------------------------------------------------------------------
PATOLOGIA = [
    # en
    r"tear", r"torn", r"ruptur", r"disrupt", r"lesion", r"injur", r"sprain", r"avuls", r"discontinu",
    r"deficien", r"insufficien", r"bucket", r"flap", r"partial", r"full.thickness", r"complete",
    r"grade (3|iii|2|ii)\b", r"surface", r"cleavage", r"fissur",
    # de
    r"riss", r"einriss", r"lasion", r"zerrung", r"distorsion", r"insuffizien", r"korbhenkel",
    r"lappenriss", r"kontinuitatsunterbrech", r"partialruptur", r"grad (3|iii|2|ii)\b", r"teilruptur",
    # es / pt
    r"rotura", r"desgarro", r"esguince", r"fisura", r"lesao", r"estiramento", r"entorse",
    r"grado (3|iii|2|ii)\b", r"ruptura",
    # fr
    r"dechirure", r"desinsertion", r"anse de seau", r"elongation",
    # nl
    r"scheur", r"ruptuur", r"laesie", r"letsel", r"distorsie", r"verstuik",
    # it
    r"rottura", r"lesione", r"lacerazion", r"distrazion",
    # tr
    r"yırtık", r"yirtik", r"hasar", r"lezyon", r"kopuk", r"parsiyel", r"burkul",
    # el
    r"ρηξ", r"κακωσ", r"βλαβ", r"διαστρεμ", r"θλασ",
    # ru
    r"разрыв", r"надрыв", r"повреждени", r"растяжени", r"отрыв",
]
PATOLOGIA_DEBIL = [  # degeneración sin rotura: evidencia débil (0.40)
    r"degenera", r"degenerat", r"degeneracion", r"degeneracao", r"degenerescence", r"degeneratie",
    r"degenerazion", r"dejenera", r"εκφυλ", r"дегенерат", r"mucoid", r"mukoid", r"mucoide",
    r"signal(anhebung|alteration|veranderung)", r"periligament", r"edema", r"odem", r"oedem",
]

ANATOMIA = {
    "ACL": [r"acl\b", r"lca\b", r"vkb\b", r"anterior cruciate", r"cruciate ligament,? anterior",
            r"kreuzbandr", r"vordere[sn]? kreuzband", r"vorderes? kreuzband", r"kreuzband,? vorder", r"ligamento cruzado anterior",
            r"ligament croise anterieur", r"voorste kruisband", r"legamento crociato anteriore",
            r"on capraz bag", r"ön capraz bag", r"προσθιο\w* χιαστ", r"передн\w* крестообразн",
            r"ligamentum cruciatum anterius"],
    "MCL": [r"mcl\b", r"mkl\b", r"lli\b", r"lcm\b", r"mcb\b", r"medial collateral", r"tibial collateral",
            r"innenband", r"mediale[sn]? seitenband", r"ligamento colateral (medial|interno|tibial)",
            r"ligament collateral (medial|interne|tibial)", r"ligament lateral interne",
            r"mediale collaterale band", r"legamento collaterale (mediale|tibiale)",
            r"medial kollateral", r"ic yan bag", r"iç yan bag", r"εσω πλαγι\w* συνδεσμ",
            r"медиальн\w* (коллатеральн|боков)", r"большеберцов\w* (коллатеральн|боков)",
            r"внутренн\w* (коллатеральн|боков)"],
    "Medial Meniscus": [r"medial\w* menisc", r"menisc\w* medial", r"innenmenisk", r"medialer? menisk",
                        r"menisco (medial|interno|mediale)", r"menisque (interne|medial)",
                        r"mediale meniscus", r"meniscus medialis", r"\bim[- ]?hinterhorn", r"imhh\b",
                        r"medial menisku", r"ic menisku", r"iç menisku", r"εσω μηνισκ", r"μηνισκ\w* εσω",
                        r"медиальн\w* мениск", r"внутренн\w* мениск", r"мениск\w* (медиальн|внутренн)"],
    "Lateral Meniscus": [r"lateral\w* menisc", r"menisc\w* lateral", r"aussenmenisk", r"lateraler? menisk",
                         r"menisco (lateral|externo|laterale)", r"menisque (externe|lateral)",
                         r"laterale meniscus", r"meniscus lateralis", r"\bam[- ]?hinterhorn", r"amhh\b",
                         r"lateral menisku", r"dıs menisku", r"dis menisku", r"εξω μηνισκ", r"μηνισκ\w* εξω",
                         r"латеральн\w* мениск", r"наружн\w* мениск", r"мениск\w* (латеральн|наружн)"],
}
MENISCO_GENERICO = [r"menisc", r"menisk", r"menisq", r"μηνισκ", r"мениск"]

AUTOCONTENIDOS = {
    "Effusion": [r"effusion", r"hydrops", r"hydarthr", r"erguss", r"gelenkerguss", r"derrame", r"epanchement",
                 r"effusie", r"versamento", r"efuzyon", r"sıvı art", r"sivi art", r"συλλογ", r"υδαρθρ",
                 r"выпот", r"increased (joint )?fluid", r"vermehrt\w* (gelenk)?flussigkeit",
                 r"aumento (de|del) liquido", r"liquido articular aumentad", r"vocht"],
    "Synovitis": [r"synovit", r"synovialit", r"sinovit", r"synoviit", r"υμενιτ", r"συνοβιτ", r"синовит",
                  r"synovial (thicken|hypertroph|prolifer)", r"hipertrofia sinovial", r"engrosamiento sinovial",
                  r"synoviale? verdick", r"epaississement synovial", r"synoviale hypertrof",
                  r"sinoviyal kalınlas", r"sinoviyal kalinlas", r"синовиальн\w* (утолщ|гипертроф)"],
    "Baker's": [r"baker", r"popliteal cyst", r"poplitealzyste", r"quiste popliteo", r"kyste poplite",
                r"cisti poplitea", r"popliteale cyste", r"bakerkyste", r"бейкер", r"popliteal kist",
                r"gastrocnem\w*[- ]semimembran", r"semimembran\w*[- ]gastrocnem"],
    "Contusion": [r"bone bruis", r"bruise", r"contusion", r"kontusion", r"contusao", r"contusie", r"contusione",
                  r"kontuzyon", r"контуз", r"ушиб", r"(bone )?marrow (o)?edema", r"knochenmark\w*odem",
                  r"knochenprell", r"markraumodem", r"knochenodem", r"edema oseo", r"edema osseo",
                  r"edema (de )?(la )?medula", r"edema medular", r"edema midollare", r"oedeme osseu",
                  r"oedeme (de la )?medul", r"beenmergoedeem", r"botkneuz", r"kemik iligi odem",
                  r"kemik odem", r"οιδημα\w* (του )?(οστικου )?μυελ", r"οστικ\w* (οιδημα|θλαση)",
                  r"отек\w* (костного )?(мозга|кост)"],
    "Fracture": [r"fractur", r"fraktur", r"fractuur", r"frattur", r"fratur", r"kırık", r"kirik", r"καταγμ",
                 r"перелом", r"breuk", r"segond", r"infraktion", r"infraction"],
}
OA_TERMINOS = [
    r"osteoarthr", r"arthros", r"osteoarthros", r"gonarthr", r"chondropath", r"chondromalac", r"chondral",
    r"cartilag\w* (loss|defect|thinn|damage|fissur|ulcer)", r"(loss|thinning|defect) of (the )?cartilag",
    r"joint space narrow", r"osteophyt", r"knorpel(schad|defekt|verlust|ausdunn|glatz|ulcer)", r"chondromalaz",
    r"artrosis", r"osteoartrosis", r"condropat", r"condromalac", r"osteofit", r"perdida (de )?cartilag",
    r"artrose", r"chondropathie", r"chondromalacie", r"gonarthrose", r"osteophyt", r"pincement",
    r"kraakbeen(verlies|defect|schade|dunn)", r"osteofyt", r"artrosi", r"condropatia",
    r"osteoartrit", r"artroz", r"gonartroz", r"kondropat", r"kondromalaz", r"kıkırdak", r"kikirdak",
    r"οστεοαρθρ", r"αρθροσ", r"χονδροπαθ", r"χονδρομαλακ", r"οστεοφυτ",
    r"остеоартр", r"артроз", r"гонартроз", r"хондропат", r"хондромаляц", r"остеофит", r"хрящ",
]
COMPARTIMENTO = {
    "Medial OA": [r"medial", r"innen", r"intern", r"inner", r"iç\b", r"ic\b", r"εσω", r"медиальн", r"внутренн"],
    "Lateral OA": [r"lateral", r"aussen", r"extern", r"outer", r"dıs\b", r"dis\b", r"εξω", r"латеральн", r"наружн"],
    "PF OA": [r"patel", r"\w*patell", r"retropatell", r"rotul", r"trochle", r"troclea", r"trokle", r"knieschijf",
              r"femoropatel", r"patellofem", r"επιγονατ", r"надколен", r"бедренно-надколен"],
}

NEG_PRE = [
    r"no\b", r"not\b", r"without", r"absen", r"negative", r"neither", r"nor\b", r"none\b",
    r"kein", r"ohne", r"nicht\b", r"weder",
    r"sin\b", r"ni\b", r"ausencia", r"nao\b", r"sem\b", r"ausencia",
    r"pas\b", r"sans\b", r"absence", r"aucun", r"geen\b", r"zonder", r"niet\b",
    r"non\b", r"senza", r"assenza", r"δεν\b", r"χωρις", r"ουτε", r"ουδεμ", r"απουσ",
    r"нет\b", r"без\b", r"не\b", r"отсутств",
]
NEG_POST = [
    r"not (seen|identified|demonstrated|present|detected)", r"absent", r"excluded", r"ruled out",
    r"nicht (nachweisbar|abgrenzbar|erkennbar|vorhanden)", r"ausgeschlossen",
    r"no se (observa|identifica|aprecia|evidencia)", r"descartad", r"ausente", r"nao (se )?(observa|identifica)",
    r"absent", r"exclu", r"niet (aantoonbaar|zichtbaar)", r"non (si )?(osserva|evidenzia)",
    r"yok", r"izlenme", r"saptanma", r"gorulme", r"rastlanma", r"degil",
    r"δεν (παρατηρ|αναδεικ)", r"не (выявлен|определя|визуализ|обнаруж)", r"отсутств",
]
NORMAL = [  # estado normal de la estructura (pre o post)
    r"intact", r"normal", r"unremarkable", r"preserved", r"regelrecht", r"unauffallig", r"intakt",
    r"integr[oa]", r"conservad", r"preservad", r"respect", r"normaal", r"normale", r"integro",
    r"olagan", r"dogal", r"saglam", r"φυσιολογ", r"ακεραι", r"норм", r"интактн", r"physiolog",
    r"fisiologic", r"fysiolog", r"физиологич", r"sans particularite", r"sin alteracion",
    r"o\.b\.", r"ob\b",
]
DUDA = [
    r"possib", r"probab", r"likely", r"suspect", r"suspicious", r"questionab", r"cannot be excluded",
    r"may\b", r"equivocal", r"fraglich", r"verdacht", r"v\. ?a\.", r"moglich", r"wahrscheinlich",
    r"nicht (sicher|auszuschliessen)", r"posible", r"probable", r"sospech", r"dudos", r"descartar",
    r"possivel", r"provavel", r"suspeit", r"suspicion", r"evoqu", r"douteu", r"mogelijk", r"waarschijnlijk",
    r"verdenking", r"possibil", r"sospett", r"supheli", r"olası", r"olasi", r"muhtemel", r"πιθαν", r"υποψ",
    r"возможн", r"вероятн", r"подозрени", r"сомнительн", r"нельзя исключить",
]
SEV_LEVE = [r"mild", r"minimal", r"minor", r"small", r"slight", r"trace", r"discret", r"subtle", r"low.grade",
            r"gering", r"leicht", r"klein", r"wenig", r"schmal", r"leve", r"ligero", r"pequen", r"escas",
            r"minim", r"leger", r"petit", r"faible", r"licht", r"lieve", r"modest", r"piccol", r"hafif",
            r"az miktar", r"kucuk", r"ince", r"ηπι", r"μικρ", r"ελαφρ", r"небольш", r"незначительн", r"минимальн"]
SEV_MOD = [r"moderat", r"intermedi", r"massig", r"mittelgrad", r"moderad", r"modere", r"matig", r"orta",
           r"μετρι", r"умеренн"]
SEV_SEV = [r"sever", r"marked", r"large", r"extensiv", r"advanced", r"complete", r"full.thickness", r"high.grade",
           r"massiv", r"prominent", r"significant", r"considerab", r"grad[eoe]? ?(3|4|iii|iv)\b",
           r"tam kat", r"tam\b", r"ausgepragt",
           r"deutlich", r"hochgradig", r"gross", r"komplett", r"vollstandig", r"fortgeschritten", r"erheblich",
           r"importante?", r"abundant", r"extens", r"avanzad", r"avancad", r"complet", r"marcad", r"acentuad",
           r"volumos", r"abondant", r"marque", r"ernstig", r"fors\b", r"uitgebreid", r"volledig", r"uitgesproken",
           r"gevorderd", r"abbondant", r"marcat", r"belirgin", r"ileri", r"buyuk", r"yaygın", r"ciddi",
           r"σοβαρ", r"μεγαλ", r"εκτεταμεν", r"πληρ", r"εντον", r"выраженн", r"значительн", r"больш", r"полн",
           r"распространенн", r"массивн"]

SPLIT_RE = re.compile(
    r"(?<=\w\w)\.(?=\s|$)|[;\n!?]|\s(?:but|however|aber|jedoch|pero|sin embargo|mais|maar|echter|ancak|fakat|"
    r"αλλα|но|однако|ma|porem)\s"
)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _compile(patterns):
    pats = [norm(p) for p in patterns]
    return re.compile(r"(?<!\w)(?:" + "|".join(pats) + r")", re.UNICODE)


RX_PAT = _compile(PATOLOGIA)
# compuestos alemanes/neerlandeses: "Innenmeniskusriss", "Kreuzbandruptur", "meniscusscheur"
RX_PAT_COMP = re.compile(r"\w(?:riss|ruptur|lasion|zerrung|insuffizienz|scheur|laesie)", re.UNICODE)
RX_PAT_DEB = _compile(PATOLOGIA_DEBIL)
RX_ANAT = {k: _compile(v) for k, v in ANATOMIA.items()}
RX_MEN_GEN = _compile(MENISCO_GENERICO)
RX_AUTO = {k: _compile(v) for k, v in AUTOCONTENIDOS.items()}
RX_OA = _compile(OA_TERMINOS)
RX_COMP = {k: _compile(v) for k, v in COMPARTIMENTO.items()}
RX_NEG_PRE = _compile(NEG_PRE)
RX_NEG_POST = _compile(NEG_POST)
RX_NORMAL = _compile(NORMAL)
RX_DUDA = _compile(DUDA)
RX_LEVE, RX_MOD, RX_SEV = _compile(SEV_LEVE), _compile(SEV_MOD), _compile(SEV_SEV)

SCORE = {"neg": 0.05, "no_mencionado": 0.15, "ambiguo": 0.30, "generico": 0.40, "debil": 0.40,
         "duda": 0.50, "leve": 0.65, "pos": 0.80, "moderado": 0.85, "severo": 0.95}
PRIORIDAD = ["severo", "moderado", "pos", "leve", "duda", "debil", "generico", "neg", "ambiguo", "no_mencionado"]
W_PRE, W_POST, W_PAT = 6, 5, 8


class Clause:
    def __init__(self, text: str, starts: list):
        self.text = text        # cláusula normalizada
        self.starts = starts    # offset de inicio de cada token

    def tok_at(self, off: int) -> int:
        # índice del token que contiene (o sigue a) el offset
        lo, hi = 0, len(self.starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.starts[mid] <= off:
                lo = mid + 1
            else:
                hi = mid
        return max(lo - 1, 0)

    def hits(self, rx):
        return [self.tok_at(m.start()) for m in rx.finditer(self.text)]


def make_clauses(report: str):
    t = norm(report)
    out = []
    for piece in SPLIT_RE.split(t):
        if piece and piece.strip():
            starts = [m.start() for m in TOKEN_RE.finditer(piece)]
            if starts:
                out.append(Clause(piece, starts))
    return out


def _near(pos_list, anchor, before, after):
    return any(anchor - before <= p <= anchor + after for p in pos_list)


def _polarity(cl: Clause, anchors: list) -> str:
    """Devuelve 'duda' | 'neg' | 'pos' para un hallazgo localizado en los tokens `anchors`."""
    duda = cl.hits(RX_DUDA)
    if any(_near(duda, a, W_PRE, W_POST) for a in anchors):
        return "duda"
    neg_pre = cl.hits(RX_NEG_PRE)
    neg_post = cl.hits(RX_NEG_POST)
    if any(_near(neg_pre, a, W_PRE, 0) for a in anchors) or any(_near(neg_post, a, 0, W_POST) for a in anchors):
        return "neg"
    return "pos"


def _severity(cl: Clause, anchors: list) -> str:
    for rx, name in ((RX_SEV, "severo"), (RX_MOD, "moderado"), (RX_LEVE, "leve")):
        h = cl.hits(rx)
        if any(_near(h, a, 4, 4) for a in anchors):
            return name
    return "pos"


def extract(report: str) -> dict:
    """Devuelve {label: evidencia} con evidencia en SCORE.keys()."""
    ev = {l: [] for l in C.LABELS}
    if not isinstance(report, str) or not report.strip():
        return {l: "no_mencionado" for l in C.LABELS}
    for cl in make_clauses(report):
        pat = sorted(set(cl.hits(RX_PAT) + cl.hits(RX_PAT_COMP)))
        pat_deb = cl.hits(RX_PAT_DEB)
        normal = cl.hits(RX_NORMAL)

        # 1) estructura + patología (LCA, LCM, meniscos)
        side_men_found = False
        for lab, rx in RX_ANAT.items():
            for a in cl.hits(rx):
                if "Meniscus" in lab:
                    side_men_found = True
                near_pat = [p for p in pat if abs(p - a) <= W_PAT]
                near_deb = [p for p in pat_deb if abs(p - a) <= W_PAT]
                if near_pat:
                    pol = _polarity(cl, [a] + near_pat)
                    ev[lab].append(_severity(cl, near_pat) if pol == "pos" else pol)
                elif _near(normal, a, 4, 4):
                    ev[lab].append("neg")
                elif near_deb:
                    pol = _polarity(cl, [a] + near_deb)
                    ev[lab].append("debil" if pol == "pos" else pol)
                else:
                    pol = _polarity(cl, [a])
                    ev[lab].append("neg" if pol == "neg" else "ambiguo")
        # menisco sin lado -> "genérico" para ambos
        if not side_men_found:
            for a in cl.hits(RX_MEN_GEN):
                near_pat = [p for p in pat if abs(p - a) <= W_PAT]
                val = None
                if near_pat:
                    pol = _polarity(cl, [a] + near_pat)
                    val = "generico" if pol == "pos" else pol
                elif _near(normal, a, 4, 4) or _polarity(cl, [a]) == "neg":
                    val = "neg"            # "the menisci are intact", "keine Meniskusläsion"
                if val:
                    ev["Medial Meniscus"].append(val)
                    ev["Lateral Meniscus"].append(val)

        # 2) hallazgos autocontenidos
        for lab, rx in RX_AUTO.items():
            for a in cl.hits(rx):
                pol = _polarity(cl, [a])
                if pol == "pos" and _near(normal, a, 3, 3):
                    pol = "neg"  # "physiological amount of fluid", "normal bone marrow"...
                ev[lab].append(_severity(cl, [a]) if pol == "pos" else pol)

        # 3) artrosis por compartimento
        comp_hits = {k: cl.hits(rx) for k, rx in RX_COMP.items()}
        for a in cl.hits(RX_OA):
            pol = _polarity(cl, [a])
            val = _severity(cl, [a]) if pol == "pos" else pol
            dists = {k: min((abs(h - a) for h in hs), default=None) for k, hs in comp_hits.items()}
            dists = {k: d for k, d in dists.items() if d is not None and d <= 10}
            if dists:
                dmin = min(dists.values())
                for k, d in dists.items():
                    if d == dmin:
                        ev[k].append(val)
            else:
                gen = "generico" if val not in ("neg", "duda") else val
                for k in COMPARTIMENTO:
                    ev[k].append(gen)

    out = {}
    for lab, lst in ev.items():
        if not lst:
            out[lab] = "no_mencionado"
        else:
            out[lab] = min(lst, key=PRIORIDAD.index)
    return out


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, rep in zip(df[C.STUDY_COL], df[C.REPORT_COL]):
        e = extract(rep)
        n_words = len(str(rep).split()) if isinstance(rep, str) else 0
        r = {C.STUDY_COL: uid}
        for lab in C.LABELS:
            s = SCORE[e[lab]]
            if e[lab] in ("pos", "leve", "moderado", "severo", "neg"):
                conf = 1.0
            elif e[lab] == "no_mencionado":
                conf = 0.6 if n_words >= 80 else 0.3   # reportes cortos no listan negativos (D003)
            elif e[lab] == "duda":
                conf = 0.5
            else:
                conf = 0.4
            r[f"score::{lab}"] = s
            r[f"conf::{lab}"] = conf
            r[f"weak::{lab}"] = int(s >= 0.5)
            r[f"ev::{lab}"] = e[lab]
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    root = C.data_root()
    tr = pd.read_csv(root / "train.csv")
    out = to_frame(tr)
    out.to_csv(C.DIR_DATOS / "etiquetas_debiles_reglas.csv", index=False)
    prev = {l: float(out[f"weak::{l}"].mean()) for l in C.LABELS}
    menc = {l: float((out[f"ev::{l}"] != "no_mencionado").mean()) for l in C.LABELS}
    log("prevalencia débil (todos los estudios):", {k: round(v, 3) for k, v in prev.items()})
    log("tasa de mención en reportes:", {k: round(v, 3) for k, v in menc.items()})
    C.save_json({"prevalencia_debil": prev, "tasa_mencion": menc}, C.DIR_RES / "02_extractor_resumen.json")
    C.append_bitacora(f"02 extractor reglas: {len(out)} reportes etiquetados")


if __name__ == "__main__":
    main()
