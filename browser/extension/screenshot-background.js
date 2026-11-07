async function maskAuditCapture(image, selection) {
  const {viewport, crop, masks} = selection ?? {};
  if (!Number.isInteger(viewport?.w) || !Number.isInteger(viewport?.h) ||
      viewport.w < 1 || viewport.h < 1 || viewport.w > 16384 || viewport.h > 16384 ||
      !Array.isArray(masks) || masks.length < 1 || masks.length > 12) {
    throw new Error('invalid_selection');
  }
  const valid = rect => rect && [rect.x,rect.y,rect.w,rect.h].every(Number.isFinite) &&
    rect.x >= 0 && rect.y >= 0 && rect.w >= 4 && rect.h >= 4 &&
    rect.x + rect.w <= viewport.w && rect.y + rect.h <= viewport.h;
  if (!valid(crop) || masks.some(mask => !valid(mask) || mask.x < crop.x || mask.y < crop.y ||
      mask.x + mask.w > crop.x + crop.w || mask.y + mask.h > crop.y + crop.h)) {
    throw new Error('invalid_selection');
  }
  const bitmap = await createImageBitmap(await (await fetch(image)).blob());
  try {
    const sx = bitmap.width / viewport.w, sy = bitmap.height / viewport.h;
    if (Math.abs(sx-sy) > 0.02 || sx < 0.25 || sx > 8) throw new Error('viewport_mismatch');
    const pixelBounds = r => [Math.floor(r.x*sx), Math.floor(r.y*sy),
      Math.ceil((r.x+r.w)*sx), Math.ceil((r.y+r.h)*sy)];
    const [x,y,right,bottom] = pixelBounds(crop);
    const w = right-x, h = bottom-y;
    if (w*h > 2000000) throw new Error('selection_too_large');
    const canvas = new OffscreenCanvas(w,h);
    const ctx = canvas.getContext('2d',{alpha:false});
    ctx.drawImage(bitmap,x,y,w,h,0,0,w,h);
    ctx.fillStyle = '#222222';
    for (const mask of masks) {
      const [l,t,r,b] = pixelBounds(mask);
      ctx.fillRect(l-x-1,t-y-1,r-l+2,b-t+2);
    }
    const blob = await canvas.convertToBlob({type:'image/png'});
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let payload = '';
    for (let i=0; i<bytes.length; i+=8192) payload += String.fromCharCode(...bytes.subarray(i,i+8192));
    return {png:btoa(payload),width:w,height:h};
  } finally {
    bitmap.close();
  }
}

const previewStates = new Map();
const previewTokens = new Map();
chrome.tabs.onActivated.addListener(({tabId,windowId})=>{
  for(const [id,state] of previewStates){
    if(state.windowId===windowId&&id!==tabId)previewStates.delete(id);
  }
});
chrome.tabs.onUpdated.addListener((id,change)=>{
  if(change.status==='loading'||change.url)previewStates.delete(id);
});
chrome.tabs.onZoomChange.addListener(({tabId,oldZoomFactor,newZoomFactor})=>{
  if(oldZoomFactor!==newZoomFactor)previewStates.delete(tabId);
});
chrome.tabs.onRemoved.addListener(id=>previewStates.delete(id));
chrome.runtime.onMessage.addListener((message,sender,reply)=>{
 const type=message?.type;
 const own=url=>sender.url===chrome.runtime.getURL(url);
 if(type==='AUDIT_REVIEW_START' && own('popup.html')){
  (async()=>{
   const [tab]=await chrome.tabs.query({active:true,currentWindow:true});
   if(!tab?.id || !/^https?:/.test(tab.url??''))return {ok:false,error:'请在普通网页中使用截图复核'};
   const key=crypto.randomUUID();
   const state={key,windowId:tab.windowId,expires:Date.now()+30000,capturing:false};
   previewStates.set(tab.id,state);
   setTimeout(()=>{if(previewStates.get(tab.id)===state)previewStates.delete(tab.id);},30000);
   try{return await chrome.tabs.sendMessage(tab.id,{type:'AUDIT_REVIEW_SELECT',key},{frameId:0});}
   catch{
    if(previewStates.get(tab.id)===state)previewStates.delete(tab.id);
    return {ok:false,error:'无法打开截图工具，请刷新当前网页'};
   }
  })().then(reply);
  return true;
 }
 if(type==='AUDIT_REVIEW_CAPTURE'&&sender.frameId===0&&sender.tab?.id){
  (async()=>{
   const tab=sender.tab;
   const state=previewStates.get(tab.id);
   if(!state || state.key!==message.key||state.capturing)return {ok:false};
   state.capturing=true;
   const current=()=>previewStates.get(tab.id)===state&&Date.now()<state.expires;
   const stillActive=async()=>{
    const [active]=await chrome.tabs.query({active:true,windowId:tab.windowId});
    if(active?.id!==tab.id||!current())return false;
    const options=sender.documentId?{documentId:sender.documentId}:{frameId:0};
    const response=await chrome.tabs.sendMessage(tab.id,{type:'AUDIT_REVIEW_CHECK',key:state.key},options);
    return response?.ok===true&&current();
   };
   try{
    if(!await stillActive())return {ok:false};
    const frame=await chrome.tabs.captureVisibleTab(tab.windowId,{format:'png'});
    if(!await stillActive())return {ok:false};
    const image=await maskAuditCapture(frame,message.selection);
    if(!await stillActive())return {ok:false};
    const token=crypto.randomUUID();
    previewTokens.set(token,image);
    setTimeout(()=>previewTokens.delete(token),30000);
    try{await chrome.tabs.create({url:chrome.runtime.getURL('screenshot-review.html#'+token)});}
    catch{previewTokens.delete(token);return {ok:false};}
    return {ok:true};
   }catch{return {ok:false};}
   finally{if(previewStates.get(tab.id)===state)previewStates.delete(tab.id);}
  })().then(reply);
  return true;
 }
 if(type==='AUDIT_REVIEW_LOAD' && sender.url?.startsWith(chrome.runtime.getURL('screenshot-review.html#'))){
  const token=sender.url.split('#')[1];
  const image=previewTokens.get(token);
  previewTokens.delete(token);
  reply(image??null);
 }
});
