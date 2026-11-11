// Optional browser regression: see docs/validation/training-charts.md.
const {chromium}=require('playwright');
const {execFileSync, spawn}=require('child_process');
const path=require('path');
const os=require('os');
const assert=require('node:assert/strict');
const fs=require('fs');
const output=path.resolve(process.argv[2] || '/tmp/training-chart-evidence');
fs.mkdirSync(output,{recursive:true});
const scratch=fs.mkdtempSync(path.join(os.tmpdir(),'training-chart-fixture-'));
const database=path.join(scratch,'experiments.sqlite3');
const python=process.env.PYTHON || 'python';
const port=process.env.PORT || '8520';
// This is a tiny legacy read-only projection, not a model or training run.
execFileSync(python,['-c',`
import sqlite3,sys
c=sqlite3.connect(sys.argv[1])
c.executescript("CREATE TABLE runs(run_id TEXT PRIMARY KEY,name TEXT,status TEXT,parent_run_id TEXT,created_at TEXT,updated_at TEXT,config_json TEXT,metadata_json TEXT,latest_checkpoint TEXT); CREATE TABLE metrics(run_id TEXT,step INTEGER,tokens_seen INTEGER,wall_time REAL,name TEXT,value REAL); PRAGMA user_version=1;")
for run in ['fixture-a','fixture-b']:
 c.execute('INSERT INTO runs VALUES(?,?,?,NULL,?,?,?, ?,NULL)',(run,run,'running','t','t','{}','{}'))
 for step in range(1,11):
  for name,val in [('train/loss',3-step*.1),('performance/tokens_per_second',100+step)]:
   c.execute('INSERT INTO metrics VALUES(?,?,?,?,?,?)',(run,step,step*32,step*2.,name,val))
c.commit()
`,database]);
const server=spawn(python,['-m','streamlit','run','src/klab/dashboard/app.py',
 '--server.address','127.0.0.1','--server.port',port,'--server.headless','true',
 '--','--runs-dir',scratch,'--lab-dir',path.join(scratch,'lab')],
 {env:{...process.env,PYTHONPATH:path.resolve('src')},stdio:'ignore'});
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM || '/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 try {
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const wireSizes=[]; let measureWire=false; let largeFrames=0;
 page.on('websocket', ws=>ws.on('framereceived', frame=>{
  const bytes=Buffer.byteLength(frame.payload);
  if(bytes>1_000_000) largeFrames++;
  if(measureWire) wireSizes.push(bytes);
 }));
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 for(let attempt=0;;attempt++) {
  try {await page.goto(`http://127.0.0.1:${port}/training`);break;}
  catch(error) {if(attempt>=60)throw error;await new Promise(r=>setTimeout(r,500));}
 }
 await page.waitForSelector('.training-live-plot .main-svg');
 await page.getByRole('combobox',{name:'Runs',exact:true}).click();
 await page.getByRole('option',{name:'fixture-b',exact:true}).click();
 await page.keyboard.press('Escape');
 await page.waitForTimeout(2500);
 await page.evaluate(async()=>{
  const plot=document.querySelector('.training-live-plot');
  await window.Plotly.relayout(plot,{'xaxis.range':[3,8],'yaxis.range':[102,109]});
  await window.Plotly.restyle(plot,{visible:'legendonly'},[1]);
  await window.Plotly.restyle(plot,{selectedpoints:[[2,3]]},[0]);
  window.originals=[...document.querySelectorAll('.training-live-plot')];
  window.samples=[];
  window.sampleTimer=setInterval(()=>window.samples.push({
   connected:window.originals.every(p=>p.isConnected),
   svg:window.originals.every(p=>!!p.querySelector('.main-svg')),
   scroll:document.querySelector('[data-testid="stMain"]').scrollTop,
   x:[...window.originals[0].layout.xaxis.range],
   hidden:window.originals[0].data[1].visible,
   selected:window.originals[0].data[0].selectedpoints,
   count:window.originals[0].data[0].x.length
  }),50);
  document.querySelector('[data-testid="stMain"]').scrollTop=700;
 });
 measureWire=true;
 await page.waitForTimeout(4500);
 for(let step=11;step<=13;step++){
  execFileSync(python,['-c',`import sqlite3,sys;c=sqlite3.connect(sys.argv[1]);c.executemany('INSERT INTO metrics VALUES(?,?,?,?,?,?)',[(r,${step},${step}*32,${step}*2,n,v) for r in ['fixture-a','fixture-b'] for n,v in [('train/loss',1.8),('performance/tokens_per_second',100+${step})]]);c.commit()`,database]);
  await page.waitForTimeout(2600);
 }
 await page.waitForTimeout(4500);
 const evidence=await page.evaluate(()=>{clearInterval(window.sampleTimer);return {samples:window.samples,plots:window.originals.map(p=>({x:p.layout.xaxis.range,y:p.layout.yaxis.range,hidden:p.data[1].visible,selected:p.data[0].selectedpoints,count:p.data[0].x.length})),scroll:document.querySelector('[data-testid="stMain"]').scrollTop}});
 measureWire=false;
 evidence.largeBundleFrames=largeFrames;
 assert.equal(largeFrames,1);
 evidence.wireFrames=wireSizes.length;
 evidence.maxWireFrameBytes=Math.max(...wireSizes);
 assert.ok(wireSizes.length>0 && evidence.maxWireFrameBytes<64_000);
 evidence.errors=errors;evidence.browser=await browser.version();

 await page.setViewportSize({width:1440,height:1200});
 await page.locator('.training-live-plot').first().scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(output,'desktop.png')});
 await page.setViewportSize({width:390,height:1100});
 await page.waitForTimeout(2500);
 await page.locator('.training-live-plot').first().scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(output,'narrow.png')});
 console.log(JSON.stringify({maxWireFrameBytes:evidence.maxWireFrameBytes,wireFrames:evidence.wireFrames,browser:evidence.browser,errors,samples:evidence.samples.length,allConnected:evidence.samples.every(s=>s.connected&&s.svg),scrolls:[...new Set(evidence.samples.map(s=>s.scroll))],plots:evidence.plots}));
 assert.equal(errors.length,0);
 assert.ok(evidence.samples.length>200);
 for(const sample of evidence.samples) {
  assert.ok(sample.connected && sample.svg);
  assert.equal(sample.scroll,700);
  assert.deepEqual(sample.x,[3,8]);
  assert.equal(sample.hidden,'legendonly');
  assert.deepEqual(sample.selected,[2,3]);
 }
 assert.equal(evidence.plots[0].count,13);
 assert.deepEqual(evidence.plots[0].y,[102,109]);
 // A rejected draw must not poison the promise chain for later data.
 await page.evaluate(()=>{
  const react=window.Plotly.react;
  window.Plotly.react=function(...args) {
   window.Plotly.react=react;
   return Promise.reject(new Error('Synthetic one-shot draw failure'));
  };
 });
 const appendThroughput=step=>execFileSync(python,['-c',
  "import sqlite3,sys;c=sqlite3.connect(sys.argv[1]);s=int(sys.argv[2]);c.execute('INSERT INTO metrics VALUES(?,?,?,?,?,?)',('fixture-a',s,s*32,s*2,'performance/tokens_per_second',100+s));c.commit()",
  database,String(step)]);
 appendThroughput(14);
 await page.getByText('Chart update failed; waiting for the next data update.',{exact:true}).waitFor();
 appendThroughput(15);
 await page.waitForFunction(()=>document.querySelector('.training-live-plot').data[0].x.at(-1)===15);
 assert.equal(await page.getByText('Chart update failed; waiting for the next data update.',{exact:true}).count(),0);
 evidence.drawFailureRecovery=true;
 fs.writeFileSync(path.join(output,'refresh.json'),JSON.stringify(evidence,null,2));
 await page.getByText('wall_time',{exact:true}).click();
 await page.waitForTimeout(2500);
 assert.notDeepEqual(await page.locator('.training-live-plot').first().evaluate(p=>p.layout.xaxis.range),[3,8]);

 // Exercise readiness failures in an isolated browser page, with controlled time.
 const readiness=await browser.newPage();
 const readinessErrors=[];
 readiness.on('pageerror',error=>readinessErrors.push(String(error)));
 await readiness.clock.install({time:new Date('2026-01-01T00:00:00Z')});
 await readiness.clock.pauseAt(new Date('2026-01-01T00:00:01Z'));
 await readiness.setContent('<div id="fixture"></div>');
 await readiness.evaluate(async source=>{
  const {default:render}=await import('data:text/javascript;base64,'+btoa(source));
  const add=window.addEventListener.bind(window);
  const remove=window.removeEventListener.bind(window);
  window.readyListeners=new Set();
  window.addEventListener=(type,fn,...rest)=>{
   if(type==='klab-plotly-ready') window.readyListeners.add(fn);
   return add(type,fn,...rest);
  };
  window.removeEventListener=(type,fn,...rest)=>{
   if(type==='klab-plotly-ready') window.readyListeners.delete(fn);
   return remove(type,fn,...rest);
  };
  window.draws=0;
  window.plotlyStub={
   react:async(plot,data,layout)=>{plot.data=data;plot.layout=layout;window.draws++;},
   Plots:{resize(){}},purge(){}
  };
  window.renderFixture=n=>render({parentElement:document.querySelector('#fixture'),
   data:JSON.stringify({data:[],layout:{uirevision:'fixture',datarevision:n}})});
  window.renderFixture(1);
 },fs.readFileSync('src/klab/dashboard/training_charts.js','utf8'));
 await readiness.clock.fastForward(14999);
 assert.equal(await readiness.locator('[role="alert"]').count(),0);
 assert.equal(await readiness.evaluate(()=>window.readyListeners.size),1);
 await readiness.clock.fastForward(1);
 await readiness.evaluate(()=>document.querySelector('.training-live-plot').pending);
 assert.equal(await readiness.locator('[role="alert"]').count(),1);
 assert.equal(await readiness.evaluate(()=>window.readyListeners.size),0);
 await readiness.evaluate(async()=>{
  window.Plotly=window.plotlyStub;
  window.renderFixture(2);
  await document.querySelector('.training-live-plot').pending;
 });
 assert.equal(await readiness.locator('[role="alert"]').count(),0);
 assert.equal(await readiness.evaluate(()=>window.draws),1);
 await readiness.evaluate(async()=>{
  delete window.Plotly;
  window.renderFixture(3);
  await Promise.resolve();
  window.Plotly=window.plotlyStub;
  window.dispatchEvent(new Event('klab-plotly-ready'));
  await document.querySelector('.training-live-plot').pending;
 });
 assert.equal(await readiness.evaluate(()=>window.readyListeners.size),0);
 await readiness.clock.fastForward(15000);
 assert.equal(await readiness.locator('[role="alert"]').count(),0);
 assert.equal(await readiness.evaluate(()=>window.draws),2);
 await readiness.evaluate(()=>{
  delete window.Plotly;
  const cleanup=window.renderFixture(4);
  window.detachedPlot=document.querySelector('.training-live-plot');
  document.querySelector('#fixture').remove();
  cleanup();
 });
 await readiness.clock.fastForward(15000);
 await readiness.evaluate(()=>window.detachedPlot.pending);
 assert.equal(await readiness.evaluate(()=>window.readyListeners.size),0);
 assert.equal(readinessErrors.length,0);
 evidence.loaderTimeoutUnmount=true;
 evidence.loaderTimeoutRecovery=true;
 evidence.loaderReadyCleanup=true;
 fs.writeFileSync(path.join(output,'refresh.json'),JSON.stringify(evidence,null,2));
 await readiness.close();
 } finally {await browser.close();server.kill('SIGTERM');
 await new Promise(resolve=>(server.exitCode!==null || server.signalCode!==null) ? resolve() : server.once('exit',resolve));
 fs.rmSync(scratch,{recursive:true,force:true});}
})().catch(e=>{server.kill('SIGTERM');console.error(e);process.exitCode=1});
