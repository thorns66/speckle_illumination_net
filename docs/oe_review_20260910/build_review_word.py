from pathlib import Path
import re, zipfile, copy, json, html
from lxml import etree as E
from lxml import html as H
import markdown
from docutils.utils.math.latex2mathml import tex2mathml

SRC = Path('/tmp/oe_nature_review_20260910/OE_nature_reviewer_report_20260910.md')
OUT = Path('/tmp/oe_nature_review_20260910/OE_nature_reviewer_report_20260910.docx')
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
XMLSPACE='{http://www.w3.org/XML/1998/namespace}space'

def el(name, **attrs):
    prefix, local = name.split(':')
    e = E.Element('{%s}%s'%(NS[prefix],local))
    for k,v in attrs.items(): e.set('{%s}%s'%(NS[prefix],k),str(v))
    return e

def add(parent,name,**attrs):
    e=el(name,**attrs);parent.append(e);return e

def mr(text, variant=None, normal=False):
    r=el('m:r');props=add(r,'m:rPr')
    if normal: add(props,'m:nor')
    if variant=='script': add(props,'m:scr',val='script');add(props,'m:sty',val='p')
    elif variant=='bold': add(props,'m:sty',val='b')
    elif variant=='normal': add(props,'m:sty',val='p')
    t=add(r,'m:t');t.set(XMLSPACE,'preserve');t.text=text
    return [r]

def into(parent, child):
    for x in mm(child):parent.append(x)

def mm(n):
    tag=E.QName(n).localname; kids=list(n);txt=n.text or ''
    if tag in ('math','mrow','mtd'):
        # Convert explicitly stretching, paired delimiters as one editable structure.
        if tag=='mrow' and len(kids)>=2:
            first=kids[0];last=kids[-1]
            left=(first.text or '') if E.QName(first).localname=='mo' else ''
            right=(last.text or '') if E.QName(last).localname=='mo' else ''
            pairs={'(':')','[':']','{':'}','⟨':'⟩','|':'|','‖':'‖'}
            if left in pairs and first.get('stretchy')!='false' and (right==pairs[left] or (left=='{' and E.QName(last).localname=='mtable')):
                d=el('m:d');pr=add(d,'m:dPr');add(pr,'m:begChr',val=left)
                paired=right==pairs[left];add(pr,'m:endChr',val=right if paired else '')
                add(pr,'m:grow',val='1');e=add(d,'m:e')
                for k in (kids[1:-1] if paired else kids[1:]):into(e,k)
                return [d]
        return [v for k in kids for v in mm(k)]
    if tag in ('mi','mn','mo','mtext'):
        if txt in ('\u2061','\u2062'):return []
        return mr(txt,n.get('mathvariant'),normal=(tag in ('mtext','mn','mo') or (len(txt)>1 and not n.get('mathvariant'))))
    if tag=='mspace':
        width=n.get('width','0')
        if width.startswith('-'):return []
        return mr('\u2003' if width.startswith('2') else '\u2009',normal=True)
    if tag=='mfrac':
        f=el('m:f');add(add(f,'m:fPr'),'m:type',val='bar')
        into(add(f,'m:num'),kids[0]);into(add(f,'m:den'),kids[1]);return [f]
    if tag in ('msub','msup','msubsup'):
        typ={'msub':'sSub','msup':'sSup','msubsup':'sSubSup'}[tag]
        f=el('m:'+typ);into(add(f,'m:e'),kids[0])
        if tag in ('msub','msubsup'):into(add(f,'m:sub'),kids[1])
        if tag in ('msup','msubsup'):into(add(f,'m:sup'),kids[-1])
        return [f]
    if tag=='mover':
        marker=''.join(kids[1].itertext())
        if marker in ('^','ˆ','̂','¯','→','˜','~'):
            a=el('m:acc');pr=add(a,'m:accPr');add(pr,'m:chr',val={'^':'\u0302','ˆ':'\u0302','~':'\u0303','˜':'\u0303'}.get(marker,marker))
            into(add(a,'m:e'),kids[0]);return [a]
        a=el('m:limUpp');into(add(a,'m:e'),kids[0]);into(add(a,'m:lim'),kids[1]);return [a]
    if tag=='munder':
        a=el('m:limLow');into(add(a,'m:e'),kids[0]);into(add(a,'m:lim'),kids[1]);return [a]
    if tag=='mtable':
        a=el('m:m');pr=add(a,'m:mPr');add(pr,'m:baseJc',val='center')
        mcs=add(pr,'m:mcs');mc=add(mcs,'m:mc');mcp=add(mc,'m:mcPr')
        add(mcp,'m:count',val=max(len(r) for r in kids));add(mcp,'m:mcJc',val='left')
        for row in kids:
            r=add(a,'m:mr')
            for cell in row:into(add(r,'m:e'),cell)
        return [a]
    raise ValueError('Unsupported MathML '+tag)

maths={};eqtags={}
def protect(match,block=False):
    text=match.group(1).strip();idx=str(len(maths))
    tags=re.findall(r'\\tag\{(\d+)\}',text)
    if tags:eqtags[idx]=tags[0]
    text=re.sub(r'\\tag\{\d+\}','',text)
    xml=tex2mathml(text,inline=not block).replace('&ApplyFunction;','\u2061').replace('&InvisibleTimes;','\u2062')
    maths[idx]=E.fromstring(xml.encode())
    return ('\n\n<div data-equation="'+idx+'"></div>\n\n') if block else '<span data-math="'+idx+'"></span>'

source=SRC.read_text()
source=re.sub(r'\\\[(.*?)\\\]',lambda m:protect(m,True),source,flags=re.S)
source=re.sub(r'\\\((.*?)\\\)',protect,source,flags=re.S)
html_doc=H.fragment_fromstring(markdown.markdown(source,extensions=['tables']),create_parent='div')
doc=E.Element('{%s}document'%NS['w'],nsmap=NS);body=add(doc,'w:body')
links=[]

def props(parent, style=None, align=None, keep=None):
    p=add(parent,'w:pPr')
    if style:add(p,'w:pStyle',val=style)
    if align:add(p,'w:jc',val=align)
    if keep:add(p,'w:keepNext')
    return p

def wr(parent,text,bold=False,italic=False,code=False,size=None,color=None):
    if not text:return
    r=add(parent,'w:r');p=add(r,'w:rPr')
    if code:add(p,'w:rFonts',ascii='Liberation Mono',hAnsi='Liberation Mono');add(p,'w:sz',val=18)
    if bold:add(p,'w:b')
    if italic:add(p,'w:i')
    if size:add(p,'w:sz',val=size)
    if color:add(p,'w:color',val=color)
    t=add(r,'w:t');t.set(XMLSPACE,'preserve');t.text=text

def inline(parent,node,bold=False,italic=False,code=False):
    tag=node.tag if isinstance(node.tag,str) else ''
    if tag=='span' and node.get('data-math') is not None:
        m=add(parent,'m:oMath');into(m,maths[node.get('data-math')]);return
    if tag=='a':
        rid='rIdLink'+str(len(links)+1);links.append((rid,node.get('href')))
        link=add(parent,'w:hyperlink');link.set('{%s}id'%NS['r'],rid)
        label=''.join(node.itertext())
        wr(link,'['+label+']' if label.isdigit() else label,color='245887');return
    bold=bold or tag in ('strong','b');italic=italic or tag in ('em','i');code=code or tag=='code'
    wr(parent,node.text,bold,italic,code)
    for child in node:
        if child.tag=='br':add(add(parent,'w:r'),'w:br')
        else:inline(parent,child,bold,italic,code)
        wr(parent,child.tail,bold,italic,code)

def paragraph(node,style='Normal',parent=body,keep=False):
    p=add(parent,'w:p');props(p,style,keep=keep)
    inline(p,node);return p

def table_border(pr,kind='single',color='C7CFD7'):
    b=add(pr,'w:tblBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:add(b,'w:'+side,val=kind,sz=4,color=color)

def math_para(parent,idx):
    p=add(parent,'w:p');pr=props(p,'Equation')
    om=add(p,'m:oMathPara');add(add(om,'m:oMathParaPr'),'m:jc',val='center')
    root=maths[idx];children=list(root)
    # Separate long two-clause equations at their existing inter-clause space.
    if eqtags.get(idx) in {'2','7','8','9','11','14'}:
        split=next((i for i,c in enumerate(children) if E.QName(c).localname=='mspace' and c.get('width') in ('2em','1em')),None)
    else:split=None
    if split is not None:
        omath=add(om,'m:oMath');array=add(omath,'m:eqArr');add(add(array,'m:eqArrPr'),'m:baseJc',val='center')
        for part in [children[:split],children[split+1:]]:
            e=add(array,'m:e')
            for child in part:into(e,child)
    else:
        into(add(om,'m:oMath'),root)
    return p

def equation(idx):
    t=add(body,'w:tbl');pr=add(t,'w:tblPr');add(pr,'w:tblW',w=9360,type='dxa');add(pr,'w:tblLayout',type='fixed');table_border(pr,'nil')
    grid=add(t,'w:tblGrid')
    for w in [8700,660]:add(grid,'w:gridCol',w=w)
    row=add(t,'w:tr');add(add(row,'w:trPr'),'w:cantSplit')
    for n,w in enumerate([8700,660]):
        c=add(row,'w:tc');cp=add(c,'w:tcPr');add(cp,'w:tcW',w=w,type='dxa');add(cp,'w:vAlign',val='center')
        if n==0:math_para(c,idx)
        else:
            p=add(c,'w:p');props(p,'Equation',align='right');wr(p,'('+eqtags[idx]+')' if idx in eqtags else '')

def table(node):
    rows=node.xpath('./thead/tr | ./tbody/tr | ./tr');ncols=len(rows[0]);t=add(body,'w:tbl')
    pr=add(t,'w:tblPr');add(pr,'w:tblW',w=9360,type='dxa');add(pr,'w:tblLayout',type='fixed');table_border(pr)
    margins=add(pr,'w:tblCellMar')
    for side in ['top','bottom']:add(margins,'w:'+side,w=75,type='dxa')
    for side in ['left','right']:add(margins,'w:'+side,w=90,type='dxa')
    widths={4:[1350,2350,3210,2450],5:[1150,1800,2450,1960,2000]}.get(ncols,[9360//ncols]*ncols)
    grid=add(t,'w:tblGrid')
    for w in widths:add(grid,'w:gridCol',w=w)
    for ri,row in enumerate(rows):
        tr=add(t,'w:tr');rp=add(tr,'w:trPr');add(rp,'w:cantSplit')
        if ri==0:add(rp,'w:tblHeader')
        for ci,cell in enumerate(row):
            tc=add(tr,'w:tc');cp=add(tc,'w:tcPr');add(cp,'w:tcW',w=widths[ci],type='dxa');add(cp,'w:vAlign',val='center')
            if ri==0:add(cp,'w:shd',fill='E8EEF4',val='clear')
            p=add(tc,'w:p');props(p,'TableText',align='left' if ci==0 or ncols==6 else 'center')
            inline(p,cell,bold=ri==0)
    p=add(body,'w:p');pr=props(p);add(pr,'w:spacing',after=80,before=0)

for node in html_doc:
    if node.tag=='div' and node.get('data-equation') is not None:equation(node.get('data-equation'))
    elif node.tag=='h1':paragraph(node,'Title')
    elif node.tag in ('h2','h3','h4','h5','h6'):
        pp=paragraph(node,{'h2':'Heading1','h3':'Heading2','h4':'Heading3','h5':'Heading3','h6':'Heading3'}[node.tag])
        if node.tag=='h2' and (''.join(node.itertext()).startswith('Reviewer ') or ''.join(node.itertext()).startswith('Cross-review synthesis')):add(pp.find('{%s}pPr'%NS['w']),'w:pageBreakBefore')
    elif node.tag=='blockquote':
        for n in node:
            if n.tag=='p':paragraph(n,'DraftNote')
    elif node.tag=='p':
        text=''.join(node.itertext());is_cap=text.startswith('Table ')
        paragraph(node,'Caption' if is_cap or text.startswith('Fig. ') else 'Normal',keep=is_cap)
    elif node.tag=='table':table(node)
    elif node.tag in ('ul','ol'):
        def render_list(listnode,level=0):
            for i,n in enumerate(listnode):
                plain=copy.deepcopy(n)
                for child in list(plain):
                    if child.tag in ('ul','ol'):plain.remove(child)
                p=add(body,'w:p');pr=props(p);add(pr,'w:ind',left=220+level*220,hanging=180)
                wr(p,str(i+1)+'. ' if listnode.tag=='ol' else '• ');inline(p,plain)
                for child in n:
                    if child.tag in ('ul','ol'):render_list(child,level+1)
        render_list(node)
    elif node.tag=='hr':
        p=add(body,'w:p');pr=props(p);add(add(pr,'w:pBdr'),'w:bottom',val='single',sz=4,color='B4BCC4')
    else:raise ValueError('Unhandled HTML block '+node.tag)

sec=add(body,'w:sectPr')
add(sec,'w:headerReference',type='default').set('{%s}id'%NS['r'],'rIdHeader')
add(sec,'w:footerReference',type='default').set('{%s}id'%NS['r'],'rIdFooter')
add(sec,'w:pgSz',w=12240,h=15840)
add(sec,'w:pgMar',top=1320,right=1440,bottom=1320,left=1440,header=640,footer=640,gutter=0)

styles=E.Element('{%s}styles'%NS['w'],nsmap={'w':NS['w']})
defaults=add(styles,'w:docDefaults');rp=add(add(defaults,'w:rPrDefault'),'w:rPr')
add(rp,'w:rFonts',ascii='Times New Roman',hAnsi='Times New Roman',eastAsia='Noto Serif CJK SC')
add(rp,'w:sz',val=21);add(rp,'w:szCs',val=21);add(rp,'w:lang',val='en-US',eastAsia='zh-CN')
pp=add(add(defaults,'w:pPrDefault'),'w:pPr');add(pp,'w:spacing',after=110,line=264,lineRule='auto');add(pp,'w:widowControl')

def style(name,size=21,bold=False,font=None,color=None,after=100,before=0,keep=False):
    st=add(styles,'w:style',type='paragraph',styleId=name);add(st,'w:name',val=name)
    if name!='Normal':add(st,'w:basedOn',val='Normal')
    pp=add(st,'w:pPr');add(pp,'w:spacing',after=after,before=before)
    if keep:add(pp,'w:keepNext');add(pp,'w:keepLines')
    rr=add(st,'w:rPr');add(rr,'w:sz',val=size);add(rr,'w:szCs',val=size)
    if bold:add(rr,'w:b')
    if font:add(rr,'w:rFonts',ascii=font,hAnsi=font,eastAsia='Noto Sans CJK SC')
    if color:add(rr,'w:color',val=color)
    return st

style('Normal');style('Title',32,True,'Arial',after=200,keep=True)
for name,size,before,level in [('Heading1',24,180,0),('Heading2',22,140,1),('Heading3',21,100,2)]:
    st=style(name,size,True,'Arial',before=before,keep=True);add(st.find('{%s}pPr'%NS['w']),'w:outlineLvl',val=level)
st=style('DraftNote',18,False,None,'31516A',after=140,before=60)
pp=st.find('{%s}pPr'%NS['w']);add(pp,'w:shd',fill='F0F5F9',val='clear');add(pp,'w:ind',left=150,right=120)
add(add(pp,'w:pBdr'),'w:left',val='single',sz=14,space=6,color='7194AD')
style('Caption',19,after=80,before=130)
st=style('TableText',17,after=30);st.find('{%s}pPr'%NS['w']).find('{%s}spacing'%NS['w']).set('{%s}line'%NS['w'],'240')
style('Equation',20,after=140,before=100)
style('Running',16,color='66717A',after=0)

header=E.Element('{%s}hdr'%NS['w'],nsmap={'w':NS['w']});p=add(header,'w:p');props(p,'Running')
wr(p,'Optics Express  |  投稿前模拟审阅  |  2026-09-10')
footer=E.Element('{%s}ftr'%NS['w'],nsmap={'w':NS['w']});p=add(footer,'w:p');props(p,'Running',align='right')
wr(p,'三份独立模拟报告及审后综合  ·  ');f=add(p,'w:fldSimple',instr='PAGE');wr(f,'1')

settings=E.Element('{%s}settings'%NS['w'],nsmap=NS);add(settings,'w:zoom',percent=100)
mp=add(settings,'m:mathPr');add(mp,'m:mathFont',val='Cambria Math');add(mp,'m:dispDef');add(mp,'m:smallFrac',val='0')

PKG='http://schemas.openxmlformats.org/package/2006/relationships'
def rels(items):
    r=E.Element('{%s}Relationships'%PKG,nsmap={None:PKG})
    for ident,typ,target,external in items:
        a=E.SubElement(r,'{%s}Relationship'%PKG,Id=ident,Type=typ,Target=target)
        if external:a.set('TargetMode','External')
    return r

base='http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
wordrels=rels([('rIdStyles',base+'styles','styles.xml',False),('rIdSettings',base+'settings','settings.xml',False),('rIdHeader',base+'header','header1.xml',False),('rIdFooter',base+'footer','footer1.xml',False)]+[(rid,base+'hyperlink',url,True) for rid,url in links])
rootrels=rels([('rIdDocument',base+'officeDocument','word/document.xml',False)])
CT='http://schemas.openxmlformats.org/package/2006/content-types'
ct=E.Element('{%s}Types'%CT,nsmap={None:CT})
E.SubElement(ct,'{%s}Default'%CT,Extension='rels',ContentType='application/vnd.openxmlformats-package.relationships+xml')
E.SubElement(ct,'{%s}Default'%CT,Extension='xml',ContentType='application/xml')
for part,typ in [('document','document.main'),('styles','styles'),('settings','settings'),('header1','header'),('footer1','footer')]:
    E.SubElement(ct,'{%s}Override'%CT,PartName='/word/'+part+'.xml',ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.'+typ+'+xml')

parts={'[Content_Types].xml':ct,'_rels/.rels':rootrels,'word/document.xml':doc,'word/_rels/document.xml.rels':wordrels,'word/styles.xml':styles,'word/settings.xml':settings,'word/header1.xml':header,'word/footer1.xml':footer}
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for name,xml in parts.items():z.writestr(name,E.tostring(xml,xml_declaration=True,encoding='UTF-8',standalone=True))
report={'output':str(OUT),'bytes':OUT.stat().st_size,'math_sources':len(maths),'numbered_equations':len(eqtags),'word_equations':len(doc.findall('.//m:oMath',NS)),'native_tables':len(doc.findall('.//w:tbl',NS)),'hyperlinks':len(links),'text_paragraphs':len(doc.findall('.//w:p',NS))}
Path('/tmp/oe_nature_review_20260910/conversion_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
