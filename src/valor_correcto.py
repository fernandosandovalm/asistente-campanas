MESES = {"01": "enero", "02": "febrero", "03": "marzo", "04": "abril", "05": "mayo", "06": "junio",
         "07": "julio", "08": "agosto", "09": "septiembre", "10": "octubre", "11": "noviembre", "12": "diciembre"}


def _variantes_texto(v: str) -> list[str]:
    """'2026-08' -> agosto; '2026-08-14' -> 14 de agosto, 14/08, agosto 14..."""
    v = str(v).lower()
    m = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", v)
    if not m:
        return [v]
    _, mes, dia = m.groups()
    nombre = MESES[mes]
    if dia is None:
        return [v, nombre]
    d = str(int(dia))
    return [v, f"{d} de {nombre}", f"{nombre} {d}", f"{dia}/{mes}", f"{d}/{int(mes)}", f"{mes}-{dia}"]


def valor_correcto(outputs, reference_outputs):
    """Numéricas: el número de referencia aparece en la respuesta (tolerancia 1 %).
    Texto: aparece el valor o una variante (fechas y meses en español)."""
    v = reference_outputs.get("valor")
    if v is None:
        return {"key": "valor_correcto", "score": None}
    respuesta = outputs["respuesta"]
    if isinstance(v, (int, float)):
        txt = respuesta.replace(",", "")
        nums = [float(n) for n in re.findall(r"\d+\.?\d*", txt)]
        return {"key": "valor_correcto", "score": any(abs(n - v) <= abs(v) * 0.01 for n in nums)}
    txt = respuesta.lower()
    return {"key": "valor_correcto", "score": any(var in txt for var in _variantes_texto(v))}
