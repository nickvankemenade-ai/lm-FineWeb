import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RTX training dashboard</title>
<style>
:root{color-scheme:dark;--bg:#000;--panel:#090b0f;--line:#252a33;--muted:#a4adba;--text:#f2f6fc;--cyan:#18e0c1;--blue:#42b8ff;--green:#45f0a8;--gold:#ffbd3f;--violet:#c69cff}
*{box-sizing:border-box}body{margin:0;background:#000;color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,Segoe UI,sans-serif}
main{max-width:1700px;margin:auto;padding:28px 30px 50px}.top{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:20px}.eyebrow{color:var(--cyan);text-transform:uppercase;letter-spacing:.16em;font-size:11px;font-weight:700}.title{font-size:30px;letter-spacing:-.04em;margin:4px 0}.sub{color:var(--muted)}.status{border:1px solid var(--line);background:#090b0f;padding:9px 13px;border-radius:999px;color:var(--muted);white-space:nowrap}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--gold);margin-right:8px}.dot.live{background:var(--cyan);box-shadow:0 0 12px #18e0c180}
.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:18px 0}.card,.chart,.rollout{background:var(--panel);border:1px solid var(--line);border-radius:12px}.card{padding:15px 17px}.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.11em}.value{font-size:23px;font-weight:650;letter-spacing:-.03em;margin-top:5px;font-variant-numeric:tabular-nums}.unit{font-size:12px;color:var(--muted);font-weight:450;letter-spacing:0}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.chart{padding:17px 18px 13px;min-width:0}.chart.loss{grid-column:1/-1}.chart h2{font-size:15px;font-weight:650;margin:0 0 9px}.plot{height:310px;position:relative}.loss .plot{height:390px}.chart canvas{display:block;width:100%;height:100%;cursor:crosshair}.legend{display:flex;gap:17px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:0 0 4px}.key{display:inline-flex;align-items:center;gap:7px}.swatch{width:14px;height:3px;border-radius:2px}.tooltip{display:none;position:absolute;z-index:2;min-width:155px;max-width:240px;padding:9px 11px;background:#10151d;border:1px solid #414b59;border-radius:8px;box-shadow:0 8px 24px #000b;pointer-events:none;font-size:12px;font-variant-numeric:tabular-nums}.tip-step{color:#c4cedb;margin-bottom:5px}.tip-row{display:flex;justify-content:space-between;gap:16px}.tip-name{color:#a4adba}.tip-value{font-weight:650}.rollouts{margin-top:18px;padding:20px}.rollouts h2{margin:0;font-size:18px}.rollout-sub{color:var(--muted);font-size:12px;margin:4px 0 14px}.rollout-list{display:grid;gap:10px}.rollout{padding:14px 16px}.rollout-head{display:flex;justify-content:space-between;gap:12px;color:var(--cyan);font-size:12px;font-weight:650}.rollout-prompt{color:#d9e4f4;margin-top:9px;font-size:13px}.rollout-text{white-space:pre-wrap;overflow-wrap:anywhere;margin-top:7px;color:#f2f6fc;line-height:1.6}.empty{color:var(--muted);padding:20px;border:1px dashed var(--line);border-radius:9px}.foot{color:var(--muted);font-size:11px;margin-top:15px}
@media(max-width:900px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.grid{grid-template-columns:1fr}.chart.loss{grid-column:auto}.loss .plot,.plot{height:330px}.top{align-items:flex-start;flex-direction:column}}@media(max-width:460px){main{padding:19px 14px}.cards{gap:8px}.value{font-size:19px}.plot{height:280px}.loss .plot{height:320px}}
</style></head>
<body><main>
<header class="top"><div><div class="eyebrow">Live model training</div><h1 class="title">RTX run dashboard</h1><div class="sub">Per-update training signals with validation checkpoints</div></div><div class="status"><span id="dot" class="dot"></span><span id="state">Waiting for metrics…</span></div></header>
<section class="cards">
<div class="card"><div class="label">Step</div><div class="value" id="step">—</div></div>
<div class="card"><div class="label">Update loss</div><div class="value" id="loss">—</div></div>
<div class="card"><div class="label">Validation loss</div><div class="value" id="val">—</div></div>
<div class="card"><div class="label">Learning rate</div><div class="value" id="lr">—</div></div>
<div class="card"><div class="label">Throughput</div><div class="value" id="throughput">—</div></div>
</section>
<section class="grid">
<article class="chart loss"><h2>Loss</h2><div class="legend"><span class="key"><i class="swatch" style="background:#45f0a8"></i>Update loss</span><span class="key"><i class="swatch" style="background:#42b8ff"></i>Eval train</span><span class="key"><i class="swatch" style="background:#ffbd3f"></i>Validation</span></div><div class="plot"><canvas id="lossChart"></canvas><div class="tooltip"></div></div></article>
<article class="chart loss"><h2>Power-law update-loss fit <span class="unit">· anchored at fit-only point (−1, 20)</span></h2><div class="legend"><span class="key"><i class="swatch" style="background:#526271"></i>Update loss</span><span class="key"><i class="swatch" style="background:#c69cff"></i>Anchored power-law fit</span></div><div class="plot"><canvas id="lossFitChart"></canvas><div class="tooltip"></div></div></article>
<article class="chart loss"><h2>Power-law fit derivative <span class="unit">· loss per optimizer update</span></h2><div class="plot"><canvas id="lossDerivativeChart"></canvas><div class="tooltip"></div></div></article>
<article class="chart"><h2>Learning rate <span class="unit">· post-warmup</span></h2><div class="plot"><canvas id="lrChart"></canvas><div class="tooltip"></div></div></article>
<article class="chart"><h2>Gradient norm (pre-clip) <span class="unit">· post-warmup</span></h2><div class="plot"><canvas id="gradChart"></canvas><div class="tooltip"></div></div></article>
<article class="chart"><h2>Throughput <span class="unit">· post-warmup</span></h2><div class="plot"><canvas id="speedChart"></canvas><div class="tooltip"></div></div></article>
<article class="chart"><h2>Tokens processed</h2><div class="plot"><canvas id="tokensChart"></canvas><div class="tooltip"></div></div></article>
</section>
<section class="rollouts"><h2>Rollouts</h2><div class="rollout-sub" id="rolloutCount">All saved generations from this run</div><div class="rollout-list" id="rolloutList"><div class="empty">Waiting for the first rollout…</div></div></section>
<div class="foot">Refreshes every second · plots are rendered locally from the training JSONL log</div>
</main>
<script>
let metricsOffset=0,rolloutsOffset=0,updates=[],evaluations=[],rollouts=[],latestEval=null,fitCacheCount=-1,fitCache={curve:[],derivative:[],parameters:null};
const fmt=(n,d=3)=>Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:d}):'—';
// Fit the global scaling form L(x) = L_inf + A*x^-alpha to binned live
// update losses. For each alpha, the best A and L_inf are linear least squares;
// a coarse scan plus golden-section refinement finds the smooth power exponent.
function fittedLoss(){if(fitCacheCount===updates.length)return fitCache;const points=updates.map(r=>[Number(r.step),Number(r.train_loss)]).filter(p=>Number.isFinite(p[0])&&p[0]>=0&&Number.isFinite(p[1]));if(points.length<4){fitCache={curve:[],derivative:[],parameters:null};fitCacheCount=updates.length;return fitCache;}const bins=Math.min(1000,points.length),sample=[];for(let b=0;b<bins;b++){const start=Math.floor(b*points.length/bins),end=Math.floor((b+1)*points.length/bins);let sx=0,sy=0;for(let i=start;i<end;i++){sx+=points[i][0];sy+=points[i][1];}sample.push([sx/(end-start),sy/(end-start)]);}const lossAt=alpha=>{let numerator=0,denominator=0;for(const [step,y] of sample){const z=Math.pow(step+2,-alpha),q=1-z;numerator+=q*(y-20*z);denominator+=q*q;}const Linf=denominator?Math.max(0,Math.min(20,numerator/denominator)):0,A=20-Linf;let sse=0;for(const [step,y] of sample){const e=y-(Linf+A*Math.pow(step+2,-alpha));sse+=e*e;}return {sse,A,Linf,alpha};};let best=null;for(let a=.01;a<=2.0001;a+=.02){const f=lossAt(a);if(!best||f.sse<best.sse)best=f;}let lo=Math.max(.001,best.alpha-.02),hi=Math.min(3,best.alpha+.02),gr=(Math.sqrt(5)-1)/2,c=hi-gr*(hi-lo),d=lo+gr*(hi-lo),fc=lossAt(c),fd=lossAt(d);for(let i=0;i<32;i++){if(fc.sse<fd.sse){hi=d;d=c;fd=fc;c=hi-gr*(hi-lo);fc=lossAt(c);}else{lo=c;c=d;fc=fd;d=lo+gr*(hi-lo);fd=lossAt(d);}}best=fc.sse<fd.sse?fc:fd;const curve=points.map(([step])=>[step,best.Linf+best.A*Math.pow(step+2,-best.alpha)]),derivative=points.map(([step])=>[step,-best.A*best.alpha*Math.pow(step+2,-best.alpha-1)]);fitCache={curve,derivative,parameters:best};fitCacheCount=updates.length;return fitCache;}
function findWarmupEnd(){let end=null,climbing=false;for(let i=1;i<updates.length;i++){const before=updates[i-1],after=updates[i],delta=after.learning_rate-before.learning_rate;if(delta>1e-14)climbing=true;else if(delta< -1e-14&&climbing){end=before.step;climbing=false;}}return end;}
function thin(points,max=2400){if(points.length<=max)return points;let stride=Math.ceil(points.length/max);return points.filter((_,i)=>i%stride===0||i===points.length-1);}
function nearest(points,step){let lo=0,hi=points.length;while(lo<hi){let mid=(lo+hi)>>1;if(points[mid][0]<step)lo=mid+1;else hi=mid;}let a=points[Math.max(0,lo-1)],b=points[Math.min(points.length-1,lo)];if(!a)return b;if(!b)return a;return Math.abs(a[0]-step)<=Math.abs(b[0]-step)?a:b;}
function draw(id,series,format){
 const c=document.getElementById(id),rect=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(320,rect.width),h=rect.height;
 c.width=w*dpr;c.height=h*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,h);
 const L=66,R=14,T=14,B=32,pw=w-L-R,ph=h-T-B,all=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p[1]));
 c._chart={series,format,L,R,T,B,pw,ph,w,h};
 if(!all.length){x.fillStyle='#a4adba';x.font='12px system-ui';x.fillText('Waiting for updates…',L,T+25);return;}
 const xmin=Math.min(...all.map(p=>p[0])),xmax=Math.max(xmin+1,...all.map(p=>p[0]));let ymin=Math.min(...all.map(p=>p[1])),ymax=Math.max(...all.map(p=>p[1]));
 if(ymax===ymin){const pad=Math.abs(ymax)*.05||.01;ymin-=pad;ymax+=pad;}else{const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;}
 const xx=v=>L+((v-xmin)/(xmax-xmin))*pw,yy=v=>T+ph-(v-ymin)/(ymax-ymin)*ph;c._chart.map={xmin,xmax,ymin,ymax,xx,yy};
 x.strokeStyle='#252a33';x.fillStyle='#a4adba';x.lineWidth=1;x.font='11px system-ui';
 for(let i=0;i<=4;i++){const y=T+ph*i/4,val=ymax-(ymax-ymin)*i/4;x.beginPath();x.moveTo(L,y);x.lineTo(w-R,y);x.stroke();x.textAlign='right';x.textBaseline='middle';x.fillText(format(val),L-8,y);}
 for(let i=0;i<=4;i++){const step=xmin+(xmax-xmin)*i/4,px=xx(step);x.beginPath();x.moveTo(px,T);x.lineTo(px,T+ph);x.stroke();x.textAlign='center';x.textBaseline='top';x.fillText(Math.round(step).toLocaleString(),px,T+ph+9);}
 for(const s of series){const pts=thin(s.points.filter(p=>Number.isFinite(p[1])));if(!pts.length)continue;x.beginPath();x.strokeStyle=s.color;x.lineWidth=s.width||2.4;pts.forEach((p,i)=>i?x.lineTo(xx(p[0]),yy(p[1])):x.moveTo(xx(p[0]),yy(p[1])));x.stroke();if(s.dots){for(const p of pts){x.beginPath();x.arc(xx(p[0]),yy(p[1]),3.3,0,Math.PI*2);x.fillStyle=s.color;x.fill();}}}
 if(Number.isFinite(c._hoverX)){const step=Math.max(xmin,Math.min(xmax,xmin+(c._hoverX-L)/pw*(xmax-xmin)));x.strokeStyle='#eaf0f8';x.setLineDash([4,4]);x.beginPath();x.moveTo(xx(step),T);x.lineTo(xx(step),T+ph);x.stroke();x.setLineDash([]);const tip=c.parentElement.querySelector('.tooltip'),rows=[];let actualStep=null;for(const s of series){const p=nearest(s.points.filter(v=>Number.isFinite(v[1])),step);if(p){actualStep=p[0];rows.push(`<div class="tip-row"><span class="tip-name" style="color:${s.color}">${s.name}</span><span class="tip-value">${format(p[1])}</span></div>`);x.beginPath();x.arc(xx(p[0]),yy(p[1]),5,0,Math.PI*2);x.strokeStyle=s.color;x.lineWidth=2;x.stroke();}}if(rows.length){tip.innerHTML=`<div class="tip-step">Step ${Math.round(actualStep).toLocaleString()}</div>${rows.join('')}`;tip.style.display='block';tip.style.left=`${Math.min(Math.max(c._hoverX+14,70),w-250)}px`;tip.style.top='18px';}}
}
function setHover(canvas,event){const r=canvas.getBoundingClientRect();canvas._hoverX=event.clientX-r.left;const cfg=canvas._chart;if(cfg)draw(canvas.id,cfg.series,cfg.format);}
document.querySelectorAll('canvas').forEach(c=>{c.addEventListener('pointermove',e=>setHover(c,e));c.addEventListener('pointerleave',()=>{c._hoverX=NaN;const tip=c.parentElement.querySelector('.tooltip');if(tip)tip.style.display='none';const cfg=c._chart;if(cfg)draw(c.id,cfg.series,cfg.format);});});
function paint(){const m=updates.at(-1);if(m){document.getElementById('step').textContent=`${m.step.toLocaleString()} / ${m.total_steps.toLocaleString()}`;document.getElementById('loss').textContent=fmt(m.train_loss,4);document.getElementById('lr').textContent=Number(m.learning_rate).toExponential(2);document.getElementById('throughput').innerHTML=`${fmt(m.tokens_per_second,0)} <span class="unit">tok/s</span>`;document.getElementById('state').textContent=`Live · ${fmt(m.progress*100,1)}%`;document.getElementById('dot').classList.add('live');}if(latestEval)document.getElementById('val').textContent=fmt(latestEval.validation_loss,4);
 const warmupEnd=findWarmupEnd(),afterWarmup=warmupEnd===null?updates:updates.filter(r=>r.step>=warmupEnd);
 draw('lossChart',[{name:'Update loss',points:updates.map(r=>[r.step,r.train_loss]),color:'#45f0a8',width:2.5},{name:'Eval train',points:evaluations.map(e=>[e.step,e.train_loss_sampled]),color:'#42b8ff',dots:true},{name:'Validation',points:evaluations.map(e=>[e.step,e.validation_loss]),color:'#ffbd3f',dots:true}],v=>fmt(v,4));
 const fit=fittedLoss();draw('lossFitChart',[{name:'Update loss',points:updates.map(r=>[r.step,r.train_loss]),color:'#526271',width:1.2},{name:'Fitted update loss',points:fit.curve,color:'#c69cff',width:2.7}],v=>fmt(v,4));
 draw('lossDerivativeChart',[{name:'Fitted loss slope (loss / update)',points:fit.derivative,color:'#c69cff',width:2.5}],v=>fmt(v,6));
 draw('lrChart',[{name:'Learning rate',points:afterWarmup.map(r=>[r.step,r.learning_rate]),color:'#42b8ff'}],v=>Number(v).toExponential(2));draw('gradChart',[{name:'Gradient norm',points:afterWarmup.map(r=>[r.step,r.grad_norm_pre_clip]),color:'#ffbd3f'}],v=>fmt(v,3));draw('speedChart',[{name:'Throughput',points:afterWarmup.map(r=>[r.step,r.tokens_per_second]),color:'#45f0a8'}],v=>fmt(v,0));draw('tokensChart',[{name:'Tokens',points:updates.map(r=>[r.step,r.tokens_processed]),color:'#c69cff'}],v=>fmt(v,0));
}
function renderRollouts(){const list=document.getElementById('rolloutList');document.getElementById('rolloutCount').textContent=`${rollouts.length.toLocaleString()} saved generation${rollouts.length===1?'':'s'} · newest first`;list.replaceChildren();if(!rollouts.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='Waiting for the first rollout…';list.append(empty);return;}for(const r of [...rollouts].reverse()){const card=document.createElement('article');card.className='rollout';const head=document.createElement('div');head.className='rollout-head';const label=document.createElement('span');label.textContent=`ROLL OUT · STEP ${Number(r.step).toLocaleString()}`;const size=document.createElement('span');size.textContent=`${r.max_new_tokens??''} tokens`;head.append(label,size);const prompt=document.createElement('div');prompt.className='rollout-prompt';prompt.textContent=`Prompt: ${r.prompt??''}`;const text=document.createElement('div');text.className='rollout-text';text.textContent=r.continuation??'';card.append(head,prompt,text);list.append(card);}}
async function pollMetrics(){try{const r=await fetch(`/metrics?since=${metricsOffset}`,{cache:'no-store'}),d=await r.json();metricsOffset=d.offset;for(const row of d.records){if(row.record_type==='update')updates.push(row);else if(row.record_type==='evaluation'||row.validation_loss!==undefined){evaluations.push(row);latestEval=row;}}if(updates.length||evaluations.length){document.getElementById('state').textContent=updates.length?'Live training data':'Evaluation data loaded';document.getElementById('dot').classList.toggle('live',!!updates.length);}paint();}catch(e){document.getElementById('state').textContent='Waiting for training log';document.getElementById('dot').classList.remove('live');}setTimeout(pollMetrics,1000);}
async function pollRollouts(){try{const r=await fetch(`/rollouts?since=${rolloutsOffset}`,{cache:'no-store'}),d=await r.json();rolloutsOffset=d.offset;rollouts.push(...d.records);if(d.records.length)renderRollouts();}catch(e){}setTimeout(pollRollouts,1500);}
window.addEventListener('resize',paint);pollMetrics();pollRollouts();
</script></body></html>"""


def parse_args():
    parser = argparse.ArgumentParser(description="Live dashboard for training metrics.")
    parser.add_argument("--profile", choices=("rtx", "a100", "default"), default="rtx")
    parser.add_argument("--metrics", type=Path, help="Override the JSONL metrics file.")
    parser.add_argument("--rollouts", type=Path, help="Override the JSONL rollout file.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def read_jsonl_since(path: Path, offset: int):
    if not path.exists():
        return offset, []
    size = path.stat().st_size
    if offset > size:
        offset = 0
    with path.open("rb") as source:
        source.seek(offset)
        chunk = source.read()
    records = []
    consumed = 0
    for line in chunk.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        consumed += len(line)
        try:
            records.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return offset + consumed, records


def make_handler(metrics_path: Path, rollouts_path: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path not in ("/metrics", "/rollouts"):
                self.send_error(404)
                return

            query = parse_qs(parsed.query)
            try:
                offset = max(0, int(query.get("since", ["0"])[0]))
                path = metrics_path if parsed.path == "/metrics" else rollouts_path
                offset, records = read_jsonl_since(path, offset)
                body = json.dumps({"offset": offset, "records": records}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError) as error:
                body = json.dumps({"error": str(error)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format_string, *args):
            return

    return DashboardHandler


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    metrics_path = args.metrics or (
        root / "checkpoints" / args.profile / "training_metrics.jsonl"
    )
    rollouts_path = args.rollouts or (
        root / "checkpoints" / args.profile / "rollouts.jsonl"
    )
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(metrics_path, rollouts_path)
    )
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Metrics:   {metrics_path}")
    print(f"Rollouts:  {rollouts_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
