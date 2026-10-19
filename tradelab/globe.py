"""Интерактивный 3D-глобус на чистом Canvas/JS (без внешних библиотек и CDN, работает офлайн).

Страны раскрашиваются по метрике, биржи светятся (открытые пульсируют), по дугам большого круга бегут частицы:
толщина и скорость потока зависят от силы связи. Мышь: вращение, колесо: масштаб, наведение: подсказка.
"""
import json
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"
_geo_cache = {}


def countries():
    """Контуры стран как список {iso, rings:[[[lon,lat],...]]} (кэшируется)."""
    if "c" not in _geo_cache:
        data = json.loads((ASSETS / "countries.geojson").read_text(encoding="utf-8"))
        out = []
        for f in data["features"]:
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            rings = [ring for poly in polys for ring in poly[:1]]  # только внешние контуры
            if f["properties"]["iso3"] != "ATA":
                out.append({"iso": f["properties"]["iso3"], "rings": rings})
        _geo_cache["c"] = out
    return _geo_cache["c"]


def lerp_color(a, b, t):
    t = float(np.clip(t, 0, 1))
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def diverging(value, scale):
    """Красный (падение) - тёмный - зелёный (рост); scale = значение полной насыщенности."""
    if value is None or not np.isfinite(value):
        return None
    t = float(np.clip(value / scale, -1, 1))
    dark, red, green = (30, 38, 52), (255, 77, 79), (38, 194, 129)
    return lerp_color(dark, green if t >= 0 else red, abs(t))


def sequential(value, vmin, vmax):
    if value is None or not np.isfinite(value):
        return None
    t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    return lerp_color((30, 38, 52), (255, 159, 26), t)


def great_circle(lat1, lon1, lat2, lon2, n=48):
    """Точки дуги большого круга: список [lon, lat]."""
    p1, p2 = np.radians([lat1, lon1]), np.radians([lat2, lon2])
    v1 = np.array([np.cos(p1[0]) * np.cos(p1[1]), np.cos(p1[0]) * np.sin(p1[1]), np.sin(p1[0])])
    v2 = np.array([np.cos(p2[0]) * np.cos(p2[1]), np.cos(p2[0]) * np.sin(p2[1]), np.sin(p2[0])])
    omega = np.arccos(np.clip(v1 @ v2, -1, 1))
    if omega < 1e-9:
        return [[lon1, lat1]] * 2
    t = np.linspace(0, 1, n)[:, None]
    pts = (np.sin((1 - t) * omega) * v1 + np.sin(t * omega) * v2) / np.sin(omega)
    return [[float(np.degrees(np.arctan2(y, x))), float(np.degrees(np.arcsin(np.clip(z, -1, 1))))]
            for x, y, z in pts]


def build(markets, arcs, country_colors, title="", legend=(), height=700, lon0=40.0, info=None, mode="globe"):
    """markets: [{name, iso, city, lat, lon, label, color, open}], arcs: [{a, b, w, color, label}] (a,b: индексы),
    country_colors: {iso3: '#rrggbb'}. Возвращает HTML-строку."""
    for arc in arcs:
        m1, m2 = markets[arc["a"]], markets[arc["b"]]
        arc["pts"] = great_circle(m1["lat"], m1["lon"], m2["lat"], m2["lon"])
    data = {"countries": countries(), "colors": country_colors, "markets": markets, "arcs": arcs,
            "title": title, "legend": list(legend), "lon0": lon0, "info": info or {}, "mode": mode}
    return (_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
            .replace("__H__", str(int(height))))


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:#0A0D12;overflow:hidden;font-family:monospace;color:#C9D1DC}
#g{width:100%;height:__H__px;display:block;cursor:grab}
#tip{position:absolute;pointer-events:none;background:rgba(19,25,35,.96);border:1px solid #FF9F1A;border-radius:4px;
padding:8px 10px;font-size:12px;display:none;max-width:320px;line-height:1.5}
#title{position:absolute;left:14px;top:10px;font-size:15px;color:#FF9F1A;letter-spacing:1px;font-weight:bold}
#legend{position:absolute;left:14px;bottom:12px;font-size:12px;line-height:18px}
#legend i{display:inline-block;width:11px;height:11px;margin-right:7px;vertical-align:-1px;border-radius:2px}
#hint{position:absolute;right:14px;bottom:12px;font-size:11px;color:#6f7b8b}
#modes{position:absolute;right:14px;top:10px}
#modes button{background:#131923;color:#C9D1DC;border:1px solid #2a3547;padding:4px 12px;font-family:monospace;cursor:pointer}
#modes button.on{background:#FF9F1A;color:#000;border-color:#FF9F1A}
</style></head><body>
<canvas id="g"></canvas><div id="tip"></div><div id="title"></div><div id="legend"></div>
<div id="modes"><button id="bg">Глобус</button><button id="bf">Карта</button></div>
<div id="hint">тяни — вращать/двигать · колесо — масштаб · наведи на страну или биржу</div>
<script>
const D=__DATA__;
const cv=document.getElementById('g'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
document.getElementById('title').textContent=D.title;
document.getElementById('legend').innerHTML=D.legend.map(l=>'<div><i style="background:'+l[1]+'"></i>'+l[0]+'</div>').join('');
const rad=Math.PI/180;
let W,H,R,cx,cy,dpr,zoom=1,lon0=D.lon0,lat0=22,auto=true,lastTouch=0,drag=null,mouse=null,mode=D.mode,SX=1;
const stars=Array.from({length:140},()=>[Math.random(),Math.random(),Math.random()*1.2+.2,Math.random()]);
function setMode(m){mode=m;document.getElementById('bg').className=m=='globe'?'on':'';document.getElementById('bf').className=m=='flat'?'on':'';
if(m=='flat'){lat0=15;}else{lat0=22;}resize();}
function resize(){dpr=window.devicePixelRatio||1;W=cv.clientWidth;H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;
ctx.setTransform(dpr,0,0,dpr,0,0);cx=W/2;cy=H/2;R=Math.min(W,H)*0.44*zoom;SX=Math.min(W/360,H/150)*zoom;}
function dl(lon){return ((lon-lon0+540)%360+360)%360-180;}
function proj(lon,lat,lift){lift=lift||1;
if(mode=='flat'){return [cx+dl(lon)*SX,cy-(lat-15)*SX-(lift-1)*SX*70,1];}
const l=(lon-lon0)*rad,p=lat*rad,p0=lat0*rad,cp=Math.cos(p),sp=Math.sin(p);
const x=cp*Math.sin(l),y=Math.cos(p0)*sp-Math.sin(p0)*cp*Math.cos(l),z=Math.sin(p0)*sp+Math.cos(p0)*cp*Math.cos(l);
return [cx+R*lift*x,cy-R*lift*y,z];}
function ringPath(coords,shift){let pts=[],any=false,prev=null;
for(const c of coords){let p;
if(mode=='flat'){let d=dl(c[0]);if(prev!==null){while(d-prev>180)d-=360;while(d-prev<-180)d+=360;}prev=d;p=[cx+(d+shift)*SX,cy-(c[1]-15)*SX,1];any=true;}
else{p=proj(c[0],c[1]);if(p[2]>=0)any=true;}pts.push(p);}
if(!any)return false;ctx.beginPath();
pts.forEach((p,i)=>{let x=p[0],y=p[1];if(mode!='flat'&&p[2]<0){const dx=x-cx,dy=y-cy,d=Math.hypot(dx,dy)||1;x=cx+dx/d*R;y=cy+dy/d*R;}
i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.closePath();return true;}
function graticule(){ctx.strokeStyle='rgba(120,140,170,.13)';ctx.lineWidth=.6;
for(let lo=-180;lo<180;lo+=30){ctx.beginPath();let on=false;for(let la=-85;la<=85;la+=5){const p=proj(lo,la);
if(p[2]>0){on?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);on=true}else on=false}ctx.stroke()}
for(let la=-60;la<=60;la+=30){ctx.beginPath();let on=false;for(let lo=-180;lo<=180;lo+=5){const p=proj(lo,la);
if(p[2]>0){on?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);on=true}else on=false}ctx.stroke()}}
function arcPts(a){const out=[];let prev=null;
a.pts.forEach((q,i)=>{const lift=1+.16*Math.sin(Math.PI*i/(a.pts.length-1));let p=proj(q[0],q[1],lift);
if(mode=='flat'){let d=dl(q[0]);if(prev!==null){while(d-prev>180)d-=360;while(d-prev<-180)d+=360;}prev=d;p[0]=cx+d*SX;}
out.push(p);});return out;}
let hoverIso=null;
function draw(t){
ctx.clearRect(0,0,W,H);hoverIso=null;
if(mode=='globe'){
for(const s of stars){ctx.fillStyle='rgba(200,215,255,'+(0.25+0.5*Math.abs(Math.sin(t/1500+s[3]*9)))+')';ctx.fillRect(s[0]*W,s[1]*H,s[2],s[2]);}
let g=ctx.createRadialGradient(cx,cy,R*.98,cx,cy,R*1.18);g.addColorStop(0,'rgba(77,163,255,.35)');g.addColorStop(1,'rgba(77,163,255,0)');
ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,R*1.18,0,7);ctx.fill();
g=ctx.createRadialGradient(cx-R*.35,cy-R*.35,R*.1,cx,cy,R);g.addColorStop(0,'#16314f');g.addColorStop(1,'#07111f');
ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.fill();
ctx.save();ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.clip();graticule();}
else{ctx.fillStyle='#07111f';ctx.fillRect(0,0,W,H);ctx.save();graticule2();}
ctx.lineWidth=.5;ctx.strokeStyle='rgba(170,190,220,.45)';
const shifts=mode=='flat'?[-360,0,360]:[0];
for(const c of D.countries){const col=D.colors[c.iso];ctx.fillStyle=col||'#1b2536';
for(const r of c.rings){for(const sh of shifts){if(ringPath(r,sh)){ctx.fill();ctx.stroke();
if(mouse&&!hoverIso&&ctx.isPointInPath(mouse[0]*dpr,mouse[1]*dpr)&&D.info[c.iso])hoverIso=c.iso;}}}}
ctx.restore();
const time=t/1000;
for(const a of D.arcs){const pts=arcPts(a);
for(const pass of [[a.w*5+2,.10],[a.w*2+.8,.55]]){ctx.strokeStyle=a.color;ctx.globalAlpha=pass[1];ctx.lineWidth=pass[0]*.6;
ctx.beginPath();let on=false;for(const p of pts){if(p[2]>0){on?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);on=true}else on=false}ctx.stroke();}
ctx.globalAlpha=1;const n=2+Math.round(a.w*4),sp=.07+a.w*.18;
for(let k=0;k<n;k++){const u=((time*sp+k/n)%1),i=u*(pts.length-1),i0=Math.floor(i),f=i-i0,p0=pts[i0],p1=pts[Math.min(i0+1,pts.length-1)];
const z=p0[2]+(p1[2]-p0[2])*f;if(z<=0)continue;const x=p0[0]+(p1[0]-p0[0])*f,y=p0[1]+(p1[1]-p0[1])*f;
const gr=ctx.createRadialGradient(x,y,0,x,y,7);gr.addColorStop(0,'#fff');gr.addColorStop(.3,a.color);gr.addColorStop(1,'rgba(0,0,0,0)');
ctx.fillStyle=gr;ctx.beginPath();ctx.arc(x,y,7,0,7);ctx.fill();}}
let hover=null,best=16;
D.markets.forEach((m,idx)=>{const p=proj(m.lon,m.lat);if(p[2]<=0.02)return;
const gr=ctx.createRadialGradient(p[0],p[1],0,p[0],p[1],16);gr.addColorStop(0,m.color);gr.addColorStop(1,'rgba(0,0,0,0)');
ctx.globalAlpha=.55;ctx.fillStyle=gr;ctx.beginPath();ctx.arc(p[0],p[1],16,0,7);ctx.fill();ctx.globalAlpha=1;
ctx.fillStyle=m.color;ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.arc(p[0],p[1],4,0,7);ctx.fill();ctx.stroke();
if(m.open){const ph=(time*0.9+idx*.17)%1;ctx.strokeStyle=m.color;ctx.globalAlpha=1-ph;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(p[0],p[1],4+ph*16,0,7);ctx.stroke();ctx.globalAlpha=1;}
if(p[2]>.4){ctx.fillStyle='rgba(230,236,245,.9)';ctx.font='11px monospace';ctx.fillText(m.name,p[0]+8,p[1]-7);}
if(mouse){const d=Math.hypot(mouse[0]-p[0],mouse[1]-p[1]);if(d<best){best=d;hover=[m,p];}}});
if(hover&&!drag){const m=hover[0];tip.style.display='block';tip.style.left=Math.min(hover[1][0]+14,W-300)+'px';tip.style.top=(hover[1][1]+8)+'px';
tip.innerHTML='<b style="color:#FF9F1A">'+m.name+'</b> · '+m.city+'<br>'+m.label+'<br>'+(m.open?'<span style="color:#26C281">● торги идут</span>':'<span style="color:#7a8697">○ закрыта</span>');}
else if(hoverIso&&!drag&&mouse){tip.style.display='block';tip.style.left=Math.min(mouse[0]+16,W-320)+'px';tip.style.top=(mouse[1]+12)+'px';tip.innerHTML=D.info[hoverIso];}
else tip.style.display='none';
}
function graticule2(){ctx.strokeStyle='rgba(120,140,170,.13)';ctx.lineWidth=.6;
for(let lo=-180;lo<=180;lo+=30){const p=proj(lo,0),q=proj(lo,-80);ctx.beginPath();ctx.moveTo(p[0],cy-(85-15)*SX);ctx.lineTo(p[0],cy+(80+15)*SX);ctx.stroke();}
for(let la=-60;la<=80;la+=30){const y=cy-(la-15)*SX;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}
let prev=0;function loop(t){if(mode=='globe'&&auto&&!drag&&t-lastTouch>2500&&!hoverIso){lon0=(lon0+(t-prev)*0.012)%360;}prev=t;draw(t);requestAnimationFrame(loop);}
cv.addEventListener('mousedown',e=>{drag=[e.clientX,e.clientY,lon0,lat0];cv.style.cursor='grabbing';});
window.addEventListener('mouseup',()=>{drag=null;lastTouch=performance.now();cv.style.cursor='grab';});
cv.addEventListener('mousemove',e=>{const b=cv.getBoundingClientRect();mouse=[e.clientX-b.left,e.clientY-b.top];
if(drag){if(mode=='flat'){lon0=drag[2]-(e.clientX-drag[0])/SX;}else{lon0=drag[2]-(e.clientX-drag[0])*.4;lat0=Math.max(-70,Math.min(70,drag[3]+(e.clientY-drag[1])*.3));}}});
cv.addEventListener('mouseleave',()=>{mouse=null;});
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.6,Math.min(3,zoom*(e.deltaY<0?1.08:.93)));resize();lastTouch=performance.now();},{passive:false});
document.getElementById('bg').onclick=()=>setMode('globe');document.getElementById('bf').onclick=()=>setMode('flat');
window.addEventListener('resize',resize);setMode(mode);requestAnimationFrame(loop);
</script></body></html>"""


def heat(value, vmax):
    """Шкала риска: тёмный -> янтарный -> красный (value в долях от vmax)."""
    if value is None or not np.isfinite(value):
        return None
    t = float(np.clip(value / vmax, 0, 1)) if vmax > 0 else 0.0
    dark, amber, red = (30, 38, 52), (255, 159, 26), (255, 60, 60)
    return lerp_color(dark, amber, t * 2) if t < 0.5 else lerp_color(amber, red, (t - 0.5) * 2)
