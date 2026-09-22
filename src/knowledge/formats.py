"""Local EPUB reading and explicitly requested PDF OCR."""
from html.parser import HTMLParser
from io import BytesIO
from pathlib import PurePosixPath, Path
import posixpath
import shutil
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET
from zipfile import ZipFile


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.hidden += 1
        if tag in {'p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'blockquote'}:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {'p', 'div', 'li', 'h1', 'h2', 'h3', 'blockquote'}:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def epub_pages(raw):
    with ZipFile(BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) > 10000 or sum(m.file_size for m in members) > 200_000_000:
            raise ValueError('EPUB exceeds local import size limit')
        if len({m.filename for m in members}) != len(members):
            raise ValueError('EPUB contains duplicate member paths')
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise ValueError('Unsafe EPUB member path')
        def xml(name):
            data = archive.read(name)
            if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
                raise ValueError('EPUB metadata must not declare entities')
            return ET.fromstring(data)
        container = xml('META-INF/container.xml')
        roots = container.findall('.//{*}rootfile')
        if not roots:
            raise ValueError('EPUB has no package document')
        package = roots[0].get('full-path', '')
        opf = xml(package)
        manifest = {item.get('id'): item for item in opf.findall('.//{*}manifest/{*}item')}
        pages = []
        for itemref in opf.findall('.//{*}spine/{*}itemref'):
            item = manifest.get(itemref.get('idref'))
            if item is None:
                raise ValueError('EPUB spine references a missing item')
            if item.get('media-type') not in {'application/xhtml+xml', 'text/html'}:
                raise ValueError('EPUB spine contains unsupported non-text content')
            href = urlsplit(item.get('href', ''))
            name = posixpath.normpath(posixpath.join(posixpath.dirname(package), unquote(href.path)))
            if href.scheme or href.netloc or name.startswith(('/', '../')):
                raise ValueError('Unsafe EPUB spine reference')
            parser = TextParser()
            parser.feed(archive.read(name).decode('utf-8-sig'))
            text = ''.join(parser.parts).strip()
            if not text:
                raise ValueError(f'EPUB chapter has no readable text: {name}')
            pages.append((f'spine {len(pages)+1}: {name}', text, 'epub'))
        return pages


def ocr_page(path, number):
    if not shutil.which('pdftoppm') or not shutil.which('tesseract'):
        raise ValueError('PDF OCR requires local pdftoppm (Poppler) and tesseract executables')
    with tempfile.TemporaryDirectory(prefix='neo-ocr-') as directory:
        output = str(Path(directory) / 'page')
        subprocess.run(['pdftoppm', '-f', str(number), '-l', str(number), '-singlefile',
                        '-scale-to', '3000', '-png', str(path), output],
                       check=True, capture_output=True, timeout=120)
        result = subprocess.run(['tesseract', output + '.png', 'stdout'],
                                check=True, capture_output=True, timeout=120)
        text = result.stdout.decode('utf-8').strip()
        if not text:
            raise ValueError(f'OCR found no text on PDF page {number}')
        return text
