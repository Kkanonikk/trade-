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


def build(markets, arcs, country_colors, title="", legend=(), height=700, lon0=40.0, info=None, mode="flat",
          ships=None, ship_points=None, chokes=None, conflicts=None, grad=None, ranking=None, subtitle="", auto_rotate=True):
    """markets: [{name, iso, city, lat, lon, label, color, open}], arcs: [{a, b, w, color, label}] (a,b: индексы),
    country_colors: {iso3: '#rrggbb'}. Возвращает HTML-строку."""
    for arc in arcs:
        m1, m2 = markets[arc["a"]], markets[arc["b"]]
        arc["pts"] = great_circle(m1["lat"], m1["lon"], m2["lat"], m2["lon"])
    data = {"countries": countries(), "colors": country_colors, "markets": markets, "arcs": arcs,
            "title": title, "legend": list(legend), "lon0": lon0, "info": info or {}, "mode": mode,
            "ships": ships or [], "shipPoints": ship_points or [], "chokes": chokes or [], "conflicts": conflicts or [],
            "grad": grad, "ranking": ranking, "subtitle": subtitle, "autoRotate": bool(auto_rotate)}
    return (_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
            .replace("__H__", str(int(height))))


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><style>
:root{--bg:#0A0D12;--panel:#10151D;--line:#222C3B;--amber:#FF9F1A;--muted:#7D8898;--text:#D7DEE8}
html,body{margin:0;background:var(--bg);overflow:hidden;font-family:'JetBrains Mono','IBM Plex Mono',Consolas,monospace;color:var(--text);font-size:12px}
#wrap{position:relative;width:100%;height:__H__px;border:1px solid var(--line);background:#07111f}
#g{position:absolute;left:0;top:0;width:100%;height:100%;display:block;cursor:grab}
#tip{position:absolute;pointer-events:none;background:rgba(16,21,29,.97);border:1px solid var(--amber);padding:8px 10px;font-size:12px;display:none;max-width:330px;line-height:1.5;z-index:5}
#bar{position:absolute;left:0;right:0;top:0;display:flex;align-items:center;gap:10px;padding:6px 10px;background:linear-gradient(180deg,rgba(10,13,18,.92),rgba(10,13,18,0));z-index:3}
#title{color:var(--amber);letter-spacing:.14em;font-weight:700;font-size:12px;text-transform:uppercase}
#sub{color:var(--muted);font-size:11px}
#sp{flex:1}
.seg{display:flex}.seg button,.chip{background:rgba(16,21,29,.92);color:var(--muted);border:1px solid var(--line);padding:4px 11px;font-family:inherit;font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;cursor:pointer;margin-left:-1px}
.seg button.on,.chip.on{background:var(--amber);color:#0A0D12;border-color:var(--amber);font-weight:700}
.chip{margin:0 4px 0 0}.chip:not(.on){opacity:.75}
#layers{position:absolute;left:10px;top:38px;z-index:3;display:flex;flex-wrap:wrap}
#legend{position:absolute;left:10px;bottom:10px;z-index:3;background:rgba(16,21,29,.9);border:1px solid var(--line);padding:7px 10px;font-size:11px;line-height:17px;max-width:46%}
#legend i{display:inline-block;width:10px;height:10px;margin-right:7px;vertical-align:-1px}
#gradbar{height:9px;width:230px;border:1px solid var(--line)}#gradlab{display:flex;justify-content:space-between;width:230px;color:var(--muted);font-size:10px;margin-top:3px}
#hint{position:absolute;left:50%;transform:translateX(-50%);bottom:8px;font-size:10px;color:#566274;z-index:2;white-space:nowrap}
#rank{position:absolute;right:0;top:0;bottom:0;width:240px;overflow:auto;background:#0d121a;border-left:1px solid var(--line);z-index:3;padding-top:34px;box-sizing:border-box}
#g.withrank{width:calc(100% - 240px)}
#rank h4{margin:0;padding:6px 9px;font-size:10px;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;border-bottom:1px solid var(--line);font-weight:600}
.rk{display:flex;align-items:center;gap:6px;padding:3px 9px;cursor:pointer;font-size:11px}.rk:hover{background:#161d29}
.rk .n{width:16px;color:#566274;text-align:right}.rk .nm{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.rk .v{color:var(--text)}.rk .b{width:5px;height:12px}
#panel{position:absolute;left:10px;bottom:70px;width:290px;max-height:55%;overflow:auto;background:rgba(16,21,29,.97);border:1px solid var(--line);border-left:3px solid var(--amber);padding:10px 12px;font-size:12px;line-height:1.55;display:none;z-index:4}
#panel .x{position:absolute;right:8px;top:4px;cursor:pointer;color:var(--muted)}
</style></head><body>
<div id="wrap"><canvas id="g"></canvas><div id="tip"></div>
<div id="bar"><span id="title"></span><span id="sub"></span><span id="sp"></span><span class="seg"><button id="bf">Карта</button><button id="bg">Глобус</button></span></div>
<div id="layers"></div><div id="rank" style="display:none"></div><div id="legend"></div><div id="panel"></div>
<div id="hint">тяни — двигать · колесо — масштаб · наведи — подсказка · клик — детали</div></div>
<script>
const D=__DATA__;
const cv=document.getElementById('g'),ctx=cv.getContext('2d'),tip=document.getElementById('tip'),panel=document.getElementById('panel');
document.getElementById('title').textContent=D.title;document.getElementById('sub').textContent=D.subtitle||'';
(function(){const lg=document.getElementById('legend');let h='';
if(D.grad){h+='<div id="gradbar" style="background:linear-gradient(90deg,'+D.grad.colors.join(',')+')"></div><div id="gradlab"><span>'+D.grad.lo+'</span><span>'+D.grad.hi+'</span></div>';}
else h+=D.legend.map(l=>'<div><i style="background:'+l[1]+'"></i>'+l[0]+'</div>').join('');
if(D.grad&&D.legend.length>2)h+=D.legend.slice(2).map(l=>'<div><i style="background:'+l[1]+'"></i>'+l[0]+'</div>').join('');
if(!h)lg.style.display='none';else lg.innerHTML=h;})();
const rad=Math.PI/180;
const LAYERS=[['ex','Рынки и объекты',D.markets.length>0],['arc','Потоки',D.arcs.length>0],['ship','Суда',(D.ships.length+D.shipPoints.length)>0],
['choke','Проливы',D.chokes.length>0],['conf','Конфликты',D.conflicts.length>0]];
const L={ex:true,arc:true,ship:true,choke:true,conf:true};
document.getElementById('layers').innerHTML=LAYERS.filter(l=>l[2]).map(l=>'<button class="chip on" data-k="'+l[0]+'">'+l[1]+'</button>').join('');
document.querySelectorAll('#layers .chip').forEach(i=>i.onclick=()=>{L[i.dataset.k]=!L[i.dataset.k];i.className=L[i.dataset.k]?'chip on':'chip';});
if(document.querySelectorAll('#layers .chip').length<2)document.getElementById('layers').style.display='none';
// рейтинг стран справа: клик/наведение подсвечивает страну на карте
let pinned=null,listHover=null;
(function(){const R=D.ranking;if(!R)return;const el=document.getElementById('rank');el.style.display='block';document.getElementById('g').className='withrank';document.getElementById('bar').style.right='240px';
const row=(r,i)=>'<div class="rk" data-iso="'+r[0]+'"><span class="n">'+(i+1)+'</span><span class="b" style="background:'+r[3]+'"></span><span class="nm">'+r[1]+'</span><span class="v">'+r[2]+'</span></div>';
el.innerHTML=(R.top.length?'<h4>'+R.top_title+'</h4>'+R.top.map(row).join(''):'')+(R.bottom.length?'<h4>'+R.bottom_title+'</h4>'+R.bottom.map(row).join(''):'');
el.querySelectorAll('.rk').forEach(e=>{e.onmouseenter=()=>{listHover=e.dataset.iso;};e.onmouseleave=()=>{listHover=null;};
e.onclick=()=>{pinned=pinned==e.dataset.iso?null:e.dataset.iso;if(pinned&&D.info[pinned])showInfo(pinned);};});})();
let W,H,R,cx,cy,dpr,zoom=1,lon0=D.lon0,lat0=22,latc=12,auto=D.autoRotate!==false,lastTouch=0,drag=null,mouse=null,mode=D.mode,SX=1,moved=0,selected=null;
const stars=Array.from({length:140},()=>[Math.random(),Math.random(),Math.random()*1.2+.2,Math.random()]);
function spark(a,color,w,h){w=w||250;h=h||40;if(!a||a.length<2)return'';const mn=Math.min(...a),mx=Math.max(...a),r=(mx-mn)||1;
const pts=a.map((v,i)=>(i/(a.length-1)*w).toFixed(1)+','+(h-3-(v-mn)/r*(h-6)).toFixed(1)).join(' ');
return '<svg width="'+w+'" height="'+h+'"><polyline fill="none" stroke="'+(color||'#FF9F1A')+'" stroke-width="1.6" points="'+pts+'"/></svg>';}
function setMode(m){mode=m;document.getElementById('bg').className=m=='globe'?'on':'';document.getElementById('bf').className=m=='flat'?'on':'';zoom=1;
if(m=='flat'){lat0=15;latc=12;}else{lat0=22;}resize();}
function resize(){dpr=window.devicePixelRatio||1;W=cv.clientWidth;H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;
ctx.setTransform(dpr,0,0,dpr,0,0);cx=W/2;cy=H/2;R=Math.min(W,H)*0.44*zoom;SX=Math.min(W/360,H/135)*zoom;}
function dl(lon){return ((lon-lon0+540)%360+360)%360-180;}
function proj(lon,lat,lift){lift=lift||1;
if(mode=='flat'){return [cx+dl(lon)*SX,cy-(lat-latc)*SX-(lift-1)*SX*70,1];}
const l=(lon-lon0)*rad,p=lat*rad,p0=lat0*rad,cp=Math.cos(p),sp=Math.sin(p);
const x=cp*Math.sin(l),y=Math.cos(p0)*sp-Math.sin(p0)*cp*Math.cos(l),z=Math.sin(p0)*sp+Math.cos(p0)*cp*Math.cos(l);
return [cx+R*lift*x,cy-R*lift*y,z];}
function ringPath(coords,shift){let pts=[],any=false,prev=null;
for(const c of coords){let p;
if(mode=='flat'){let d=dl(c[0]);if(prev!==null){while(d-prev>180)d-=360;while(d-prev<-180)d+=360;}prev=d;p=[cx+(d+shift)*SX,cy-(c[1]-latc)*SX,1];any=true;}
else{p=proj(c[0],c[1]);if(p[2]>=0)any=true;}pts.push(p);}
if(!any)return false;ctx.beginPath();
pts.forEach((p,i)=>{let x=p[0],y=p[1];if(mode!='flat'&&p[2]<0){const dx=x-cx,dy=y-cy,d=Math.hypot(dx,dy)||1;x=cx+dx/d*R;y=cy+dy/d*R;}
i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.closePath();return true;}
function graticule(){ctx.strokeStyle='rgba(120,140,170,.13)';ctx.lineWidth=.6;
for(let lo=-180;lo<180;lo+=30){ctx.beginPath();let on=false;for(let la=-85;la<=85;la+=5){const p=proj(lo,la);
if(p[2]>0){on?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);on=true}else on=false}ctx.stroke()}
for(let la=-60;la<=60;la+=30){ctx.beginPath();let on=false;for(let lo=-180;lo<=180;lo+=5){const p=proj(lo,la);
if(p[2]>0){on?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);on=true}else on=false}ctx.stroke()}}
function graticule2(){ctx.strokeStyle='rgba(120,140,170,.13)';ctx.lineWidth=.6;
for(let lo=-180;lo<=180;lo+=30){const p=proj(lo,0);ctx.beginPath();ctx.moveTo(p[0],cy-(85-latc)*SX);ctx.lineTo(p[0],cy+(latc+80)*SX);ctx.stroke();}
for(let la=-60;la<=80;la+=30){const y=cy-(la-latc)*SX;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}
function arcPts(pts0,lifted){const out=[];let prev=null;
pts0.forEach((q,i)=>{const lift=lifted?1+.16*Math.sin(Math.PI*i/(pts0.length-1)):1.004;let p=proj(q[0],q[1],lift);
if(mode=='flat'){let d=dl(q[0]);if(prev!==null){while(d-prev>180)d-=360;while(d-prev<-180)d+=360;}prev=d;p[0]=cx+d*SX;}
out.push(p);});return out;}
// длины маршрутов судов (угловые), чтобы скорость была равномерной
function hav(a,b){const x=(b[0]-a[0])*rad*Math.cos((a[1]+b[1])/2*rad),y=(b[1]-a[1])*rad;return Math.hypot(x,y);}
D.ships.forEach(r=>{r.cum=[0];for(let i=1;i<r.pts.length;i++){let a=r.pts[i-1],b=r.pts[i];if(Math.abs(b[0]-a[0])>180)b=[b[0]+(b[0]<a[0]?360:-360),b[1]];r.cum.push(r.cum[i-1]+hav(a,b));}r.len=r.cum[r.cum.length-1];});
function onRoute(r,u){const t=u*r.len;let i=1;while(i<r.cum.length-1&&r.cum[i]<t)i++;const f=(t-r.cum[i-1])/((r.cum[i]-r.cum[i-1])||1);
const a=r.pts[i-1];let b=r.pts[i];if(Math.abs(b[0]-a[0])>180)b=[b[0]+(b[0]<a[0]?360:-360),b[1]];return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f];}
function tri(x,y,ang,size,color){ctx.save();ctx.translate(x,y);ctx.rotate(ang);ctx.fillStyle=color;ctx.strokeStyle='rgba(255,255,255,.8)';ctx.lineWidth=.8;
ctx.beginPath();ctx.moveTo(size*1.4,0);ctx.lineTo(-size,size*.8);ctx.lineTo(-size*.6,0);ctx.lineTo(-size,-size*.8);ctx.closePath();ctx.fill();ctx.stroke();ctx.restore();}
let hoverIso=null,hoverObj=null;
function hit(obj,kind,p,best){if(!mouse||p[2]<=0.02)return best;const d=Math.hypot(mouse[0]-p[0],mouse[1]-p[1]);
if(d<best[0]){return [d,obj,kind,p];}return best;}
function draw(t){
ctx.clearRect(0,0,W,H);hoverIso=null;hoverObj=null;let best=[14,null,null,null];
if(mode=='globe'){
for(const s of stars){ctx.fillStyle='rgba(200,215,255,'+(0.25+0.5*Math.abs(Math.sin(t/1500+s[3]*9)))+')';ctx.fillRect(s[0]*W,s[1]*H,s[2],s[2]);}
let g=ctx.createRadialGradient(cx,cy,R*.98,cx,cy,R*1.18);g.addColorStop(0,'rgba(77,163,255,.35)');g.addColorStop(1,'rgba(77,163,255,0)');
ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,R*1.18,0,7);ctx.fill();
g=ctx.createRadialGradient(cx-R*.35,cy-R*.35,R*.1,cx,cy,R);g.addColorStop(0,'#16314f');g.addColorStop(1,'#07111f');
ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.fill();
ctx.save();ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.clip();graticule();}
else{ctx.fillStyle='#070d17';ctx.fillRect(0,0,W,H);ctx.save();graticule2();}
ctx.lineWidth=.5;ctx.strokeStyle='rgba(150,170,200,.35)';
const shifts=mode=='flat'?[-360,0,360]:[0];
for(const c of D.countries){const col=D.colors[c.iso];ctx.fillStyle=col||'#141c29';
for(const r of c.rings){for(const sh of shifts){if(ringPath(r,sh)){ctx.fill();ctx.stroke();
if(mouse&&!hoverIso&&ctx.isPointInPath(mouse[0]*dpr,mouse[1]*dpr)&&D.info[c.iso])hoverIso=c.iso;}}}}
const hl=listHover||pinned||hoverIso;
if(hl){const c=D.countries.find(x=>x.iso==hl);if(c){ctx.lineWidth=2;ctx.strokeStyle='#FF9F1A';for(const r of c.rings){for(const sh of shifts){if(ringPath(r,sh))ctx.stroke();}}}}
ctx.restore();
const time=t/1000;
if(L.conf){for(const c of D.conflicts){const p=proj(c.lon,c.lat);if(p[2]<=0.02)continue;
const base=Math.max(5,Math.min(34,3+Math.sqrt(c.deaths||1)*(c.scale||0.9)));const ph=(((time*.7+c.lon*.013)%1)+1)%1;
const gr=ctx.createRadialGradient(p[0],p[1],0,p[0],p[1],base*1.8);gr.addColorStop(0,'rgba(255,60,60,.55)');gr.addColorStop(1,'rgba(255,60,60,0)');
ctx.fillStyle=gr;ctx.beginPath();ctx.arc(p[0],p[1],base*1.8,0,7);ctx.fill();
ctx.fillStyle='#FF3C3C';ctx.beginPath();ctx.arc(p[0],p[1],Math.max(2.5,base*.28),0,7);ctx.fill();
ctx.strokeStyle='rgba(255,100,100,'+(1-ph)+')';ctx.lineWidth=1.2;ctx.beginPath();ctx.arc(p[0],p[1],base*.3+ph*base,0,7);ctx.stroke();
best=hit(c,'conf',p,best);}}
if(L.arc)for(const a of D.arcs){const pts=arcPts(a.pts,true);
for(const pass of [[a.w*5+2,.10],[a.w*2+.8,.55]]){ctx.strokeStyle=a.color;ctx.globalAlpha=pass[1];ctx.lineWidth=pass[0]*.6;
ctx.beginPath();let on=false;for(const p of pts){if(p[2]>0){on?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);on=true}else on=false}ctx.stroke();}
ctx.globalAlpha=1;const n=2+Math.round(a.w*4),sp=.07+a.w*.18;
for(let k=0;k<n;k++){const u=((time*sp+k/n)%1),i=u*(pts.length-1),i0=Math.floor(i),f=i-i0,p0=pts[i0],p1=pts[Math.min(i0+1,pts.length-1)];
const z=p0[2]+(p1[2]-p0[2])*f;if(z<=0)continue;const x=p0[0]+(p1[0]-p0[0])*f,y=p0[1]+(p1[1]-p0[1])*f;
const gr=ctx.createRadialGradient(x,y,0,x,y,7);gr.addColorStop(0,'#fff');gr.addColorStop(.3,a.color);gr.addColorStop(1,'rgba(0,0,0,0)');
ctx.fillStyle=gr;ctx.beginPath();ctx.arc(x,y,7,0,7);ctx.fill();}}
if(L.ship){
for(const r of D.ships){const pts=arcPts(r.pts,false);ctx.strokeStyle=r.color;ctx.globalAlpha=.16;ctx.lineWidth=1.2;
ctx.beginPath();let on=false;for(const p of pts){if(p[2]>0){on?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);on=true}else on=false}ctx.stroke();ctx.globalAlpha=1;
const T=70+r.len*6;for(let k=0;k<r.n;k++){const u=((time/T)+k/r.n)%1,q=onRoute(r,u),q2=onRoute(r,Math.min(u+.004,.9999));
const p=proj(q[0],q[1],1.004),p2=proj(q2[0],q2[1],1.004);if(p[2]<=0.02)continue;
tri(p[0],p[1],Math.atan2(p2[1]-p[1],(mode=='flat'&&Math.abs(p2[0]-p[0])>W/2)?0:p2[0]-p[0]),4.2,r.color);
best=hit({name:r.name,demo:true,n:r.n,color:r.color},'route',p,best);}}
for(const s of D.shipPoints){const p=proj(s.lon,s.lat,1.004);if(p[2]<=0.02)continue;
const a=(s.heading||0)*rad,q=proj(s.lon+Math.sin(a)*.3/Math.cos(s.lat*rad),s.lat+Math.cos(a)*.3,1.004);
tri(p[0],p[1],Math.atan2(q[1]-p[1],q[0]-p[0]),4,s.type=='Танкер'?'#FF9F1A':'#4DA3FF');best=hit(s,'ship',p,best);}}
if(L.choke)for(const c of D.chokes){const p=proj(c.lon,c.lat);if(p[2]<=0.02)continue;const s=c.size||7;
const gr=ctx.createRadialGradient(p[0],p[1],0,p[0],p[1],s*2.6);gr.addColorStop(0,c.color);gr.addColorStop(1,'rgba(0,0,0,0)');
ctx.globalAlpha=.5;ctx.fillStyle=gr;ctx.beginPath();ctx.arc(p[0],p[1],s*2.6,0,7);ctx.fill();ctx.globalAlpha=1;
ctx.fillStyle=c.color;ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p[0],p[1]-s);ctx.lineTo(p[0]+s,p[1]);ctx.lineTo(p[0],p[1]+s);ctx.lineTo(p[0]-s,p[1]);ctx.closePath();ctx.fill();ctx.stroke();
if(p[2]>.4){ctx.fillStyle='rgba(230,236,245,.92)';ctx.font='11px monospace';ctx.fillText(c.name,p[0]+s+4,p[1]+4);}
best=hit(c,'choke',p,best);}
if(L.ex)D.markets.forEach((m,idx)=>{const p=proj(m.lon,m.lat);if(p[2]<=0.02)return;
const gr=ctx.createRadialGradient(p[0],p[1],0,p[0],p[1],16);gr.addColorStop(0,m.color);gr.addColorStop(1,'rgba(0,0,0,0)');
ctx.globalAlpha=.55;ctx.fillStyle=gr;ctx.beginPath();ctx.arc(p[0],p[1],16,0,7);ctx.fill();ctx.globalAlpha=1;
ctx.fillStyle=m.color;ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.arc(p[0],p[1],4,0,7);ctx.fill();ctx.stroke();
if(m.open){const ph=(time*0.9+idx*.17)%1;ctx.strokeStyle=m.color;ctx.globalAlpha=1-ph;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(p[0],p[1],4+ph*16,0,7);ctx.stroke();ctx.globalAlpha=1;}
if(p[2]>.4){ctx.fillStyle='rgba(230,236,245,.9)';ctx.font='11px monospace';ctx.fillText(m.name,p[0]+8,p[1]-7);}
best=hit(m,'market',p,best);});
if(best[1]){hoverObj=best;}
if(hoverObj&&!drag){const o=hoverObj[1],k=hoverObj[2],p=hoverObj[3];let html='';
if(k=='market')html='<b style="color:#FF9F1A">'+o.name+'</b> · '+o.city+'<br>'+o.label+(o.spark?'<br>'+spark(o.spark,o.color,230,34):'')+'<br>'+(o.open?'<span style="color:#26C281">● торги идут</span>':'<span style="color:#7a8697">○ закрыта</span>')+(o.hours?' · '+o.hours:'');
else if(k=='choke')html='<b style="color:#FF9F1A">'+o.name+'</b><br>'+o.label+(o.spark?'<br>'+spark(o.spark,o.color,230,34):'');
else if(k=='conf')html='<b style="color:#FF5050">'+o.name+'</b><br>'+o.label;
else if(k=='ship')html='<b style="color:#FF9F1A">'+(o.name||'судно')+'</b><br>'+(o.type||'')+(o.sog!=null?' · '+o.sog+' уз':'')+(o.dest?'<br>курс на: '+o.dest:'');
else if(k=='route')html='<b style="color:'+o.color+'">'+o.name+'</b><br>судов в модели: '+o.n+'<br><span style="color:#8B96A5">условная модель маршрута, не реальные позиции</span>';
tip.style.display='block';tip.style.left=Math.min(p[0]+14,W-340)+'px';tip.style.top=(p[1]+8)+'px';tip.innerHTML=html;}
else if(hoverIso&&!drag&&mouse){tip.style.display='block';tip.style.left=Math.min(mouse[0]+16,W-340)+'px';tip.style.top=(mouse[1]+12)+'px';tip.innerHTML=D.info[hoverIso];}
else tip.style.display='none';
}
function showInfo(iso){panel.innerHTML='<span class="x" onclick="panel.style.display=\'none\'">✕</span>'+D.info[iso];panel.style.display='block';}
function showPanel(o,k){if(!o){panel.style.display='none';return;}let h='<span class="x" onclick="panel.style.display=\'none\'">✕</span>';
if(k=='market')h+='<b style="color:#FF9F1A;font-size:14px">'+o.name+'</b><br>'+o.city+'<br>'+(o.detail||o.label)+(o.spark?'<br>'+spark(o.spark,o.color):'');
else if(k=='choke')h+='<b style="color:#FF9F1A;font-size:14px">'+o.name+'</b><br>'+(o.detail||o.label)+(o.spark?'<br>'+spark(o.spark,o.color):'');
else if(k=='conf')h+='<b style="color:#FF5050;font-size:14px">'+o.name+'</b><br>'+(o.detail||o.label);
else if(k=='ship')h+='<b style="color:#FF9F1A">'+(o.name||'судно')+'</b><br>'+(o.type||'')+(o.dest?'<br>курс на: '+o.dest:'');
else return;
panel.innerHTML=h;panel.style.display='block';}
let prev=0;function loop(t){if(mode=='globe'&&auto&&!drag&&t-lastTouch>2500&&!hoverIso&&!hoverObj){lon0=(lon0+(t-prev)*0.012)%360;}prev=t;draw(t);requestAnimationFrame(loop);}
cv.addEventListener('mousedown',e=>{drag=[e.clientX,e.clientY,lon0,lat0,latc];moved=0;cv.style.cursor='grabbing';});
window.addEventListener('mouseup',()=>{if(drag&&moved<4){if(hoverObj)showPanel(hoverObj[1],hoverObj[2]);else if(hoverIso){pinned=hoverIso;showInfo(hoverIso);}else{pinned=null;showPanel(null);}}drag=null;lastTouch=performance.now();cv.style.cursor='grab';});
cv.addEventListener('mousemove',e=>{const b=cv.getBoundingClientRect();mouse=[e.clientX-b.left,e.clientY-b.top];
if(drag){moved+=Math.abs(e.movementX)+Math.abs(e.movementY);if(mode=='flat'){lon0=drag[2]-(e.clientX-drag[0])/SX;latc=Math.max(-40,Math.min(70,drag[4]+(e.clientY-drag[1])/SX));}else{lon0=drag[2]-(e.clientX-drag[0])*.4;lat0=Math.max(-70,Math.min(70,drag[3]+(e.clientY-drag[1])*.3));}}});
cv.addEventListener('mouseleave',()=>{mouse=null;});
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.8,Math.min(8,zoom*(e.deltaY<0?1.1:.91)));resize();lastTouch=performance.now();},{passive:false});
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


def scale_color(value, lo, hi, better="none"):
    """Цвет страны по значению: better='down' (выше = хуже): тёмный→красный; 'up' (выше = лучше): красный→тёмный→зелёный;
    'none': тёмный→янтарный."""
    if value is None or not np.isfinite(value) or hi <= lo:
        return None
    t = float(np.clip((value - lo) / (hi - lo), 0, 1))
    if better == "down":
        return heat(t, 1.0)
    if better == "up":
        dark, red, green = (30, 38, 52), (255, 77, 79), (38, 194, 129)
        return lerp_color(red, dark, t * 2) if t < 0.5 else lerp_color(dark, green, (t - 0.5) * 2)
    return lerp_color((30, 38, 52), (255, 159, 26), t)
