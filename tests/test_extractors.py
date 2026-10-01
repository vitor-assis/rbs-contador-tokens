import io

import pytest

from core.extractors import decode_text, extract, html_to_text


def test_txt_utf8_com_bom():
    f = extract("a.txt", "﻿Olá, mundo".encode("utf-8"))
    assert f.ok and f.text == "Olá, mundo" and f.chars == 10


def test_txt_cp1252():
    assert decode_text("ação".encode("cp1252")) == "ação"


@pytest.mark.parametrize("name", ["a.md", "a.csv", "a.json", "a.xml"])
def test_formatos_texto(name):
    f = extract(name, b"conteudo 123")
    assert f.ok and f.text == "conteudo 123"


def test_html_remove_tags_script_e_style():
    html = "<html><head><style>p{color:red}</style><script>alert(1)</script></head>" \
           "<body><h1>Título</h1><p>Olá <b>mundo</b></p></body></html>"
    assert html_to_text(html) == "Título\nOlá mundo"
    f = extract("pagina.html", html.encode("utf-8"))
    assert f.ok and "alert" not in f.text and f.warnings


def test_docx():
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Primeiro parágrafo")
    doc.add_paragraph("Segundo")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "A"
    table.rows[0].cells[1].text = "B"
    buf = io.BytesIO()
    doc.save(buf)
    f = extract("doc.docx", buf.getvalue())
    assert f.ok
    assert f.text == "Primeiro parágrafo\nSegundo\nA | B"


def _pdf_com_texto(texto: str) -> bytes:
    """Gera um PDF mínimo com uma página de texto (sem dependências extras)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return out.getvalue()


def test_pdf_extrai_texto_e_paginas():
    pytest.importorskip("pypdf")
    f = extract("manual.pdf", _pdf_com_texto("Politica de trocas"))
    assert f.ok, f.error
    assert f.pages == 1
    assert "Politica de trocas" in f.text


def test_pdf_corrompido_nao_quebra():
    f = extract("ruim.pdf", b"isto nao e um pdf")
    assert f.error and not f.ok


def test_imagem_invalida_da_erro():
    f = extract("foto.png", b"\x89PNG....")
    assert f.is_image and f.error and not f.ok


def test_formato_nao_suportado():
    f = extract("planilha.xlsx", b"...")
    assert f.error and "não suportado" in f.error
