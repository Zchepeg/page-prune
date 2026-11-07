(() => {
if (window.top !== window) return;
let reviewCleanup;
let reviewTicket;
const changed = (records,panel)=>records.some(record=>
 !panel.contains(record.target) &&
 !(record.type==='childList'&&[...record.addedNodes,...record.removedNodes].every(node=>node===panel))
);
const geometry = ()=>JSON.stringify([location.href,scrollX,scrollY,innerWidth,innerHeight,devicePixelRatio]);
const intersects = (crop,r)=>crop.x<r.right&&crop.x+crop.w>r.left&&crop.y<r.bottom&&crop.y+crop.h>r.top;
const containsUnsupported = crop=>[...document.querySelectorAll('iframe,canvas,video,object,embed')].some(node=>
 node.checkVisibility({opacityProperty:true,visibilityProperty:true})&&intersects(crop,node.getBoundingClientRect())
);
const slotSelector = '[class*="ad-slot" i],[class*="ad_slot" i],[class*="advertising-slot" i],[class*="advertising_slot" i],[data-ad-slot],div[data-empty="true"]:has(> [id$="__slot" i])';
const protectedContent = 'main,nav,header,article,form,input,button,select,textarea,a[href],[role="navigation"],[role="search"]';
const disclosureLabel = /^(["'])(?:advertisement|advertising|广告)\1$/i;
const visibleInCrop = (node,crop)=>{
 const rect=node.getBoundingClientRect();
 return rect.width>0&&rect.height>0&&intersects(crop,rect)&&
  node.checkVisibility({opacityProperty:true,visibilityProperty:true});
};
const hasDisclosure = node=>['::before','::after'].some(pseudo=>{
 const style=getComputedStyle(node,pseudo);
 return style.display!=='none'&&style.visibility==='visible'&&style.opacity!=='0'&&disclosureLabel.test(style.content);
});
const cropHasDisclosure = crop=>{
 const slots=document.querySelectorAll(slotSelector);
 for(let i=0;i<Math.min(slots.length,48);i++){
  const slot=slots[i];
  if(!['DIV','ASIDE','SECTION','INS'].includes(slot.tagName)||!visibleInCrop(slot,crop)||
     slot.textContent.trim()||slot.querySelector(protectedContent))continue;
  if(hasDisclosure(slot))return true;
  for(let j=0;j<Math.min(slot.children.length,4);j++){
   const child=slot.children[j];
   if(!child.childElementCount&&visibleInCrop(child,crop)&&hasDisclosure(child))return true;
  }
 }
 return false;
};
chrome.runtime.onMessage.addListener((message,sender,reply)=>{
 if(sender.id!==chrome.runtime.id)return;
 if(message?.type==='AUDIT_REVIEW_CHECK'){
  reply({ok:reviewTicket?.key===message.key&&reviewTicket.valid()});
  return;
 }
 if(message?.type!=='AUDIT_REVIEW_SELECT')return;
 reviewTicket?.stop();
 reviewTicket=null;
 reviewCleanup?.();
 let phase='crop',start,crop,mask,selectedGeometry;
 const panel=document.createElement('div');
 panel.id='audit-review-controls';
 Object.assign(panel.style,{position:'fixed',zIndex:'2147483647',top:'8px',left:'8px',background:'white',color:'black',padding:'10px'});
 panel.innerHTML='<span aria-live="polite">第 1 步：拖出截图范围（蓝框）</span> <button id="audit-cancel">取消</button> <button id="audit-redraw" disabled>重选截图范围</button> <button id="audit-preview" disabled>预览遮盖结果</button>';
 document.documentElement.append(panel);
 let dirty=false;
 const observer=new MutationObserver(records=>{if(crop&&changed(records,panel))dirty=true;});
 observer.observe(document.documentElement,{subtree:true,childList:true,attributes:true,characterData:true});
 const pendingChanged=()=>changed(observer.takeRecords(),panel);
 const valid=()=>{dirty=pendingChanged()||dirty;return !dirty&&selectedGeometry===geometry()&&!containsUnsupported(crop);};
 const status=panel.querySelector('span');
 const evidence=document.createElement('div');
 evidence.setAttribute('role','status');
 panel.append(evidence);
 const confirm=panel.querySelector('#audit-preview');
 const redraw=panel.querySelector('#audit-redraw');
 const makeOutline=color=>{
  const outline=document.createElement('div');
  Object.assign(outline.style,{position:'fixed',pointerEvents:'none',boxSizing:'border-box',outline:'2px solid '+color,display:'none'});
  panel.append(outline);
  return outline;
 };
 const cropOutline=makeOutline('#1675e0');
 const maskOutline=makeOutline('#ec5800');
 const maskPrompt='第 2 步：在蓝框内再拖一次，框住要遮盖的内容';
 const resetCrop=()=>{
  phase='crop';start=crop=mask=undefined;
  cropOutline.style.display=maskOutline.style.display='none';
  evidence.textContent='';
  status.textContent='第 1 步：拖出截图范围（蓝框）';
  redraw.textContent='重选截图范围';
  redraw.disabled=confirm.disabled=true;
 };
 const draw=(node,rect)=>{
  Object.assign(node.style,{left:rect.x+'px',top:rect.y+'px',width:rect.w+'px',height:rect.h+'px',display:'block'});
 };
 const rectangle=(a,b)=>({x:Math.min(a[0],b[0]),y:Math.min(a[1],b[1]),w:Math.abs(b[0]-a[0]),h:Math.abs(b[1]-a[1])});
 const point=e=>[e.clientX,e.clientY];
 const suppressPageClick=event=>{
  if(panel.contains(event.target))return;
  event.preventDefault();
  event.stopImmediatePropagation();
 };
 const down=e=>{if(e.isTrusted&&e.button===0&&!panel.contains(e.target)&&phase!=='ready'){e.preventDefault();start=point(e);}};
 const move=e=>{
  if(start&&e.isTrusted)draw(phase==='crop'?cropOutline:maskOutline,rectangle(start,point(e)));
 };
 const up=e=>{
  if(!e.isTrusted||!start)return;
  const rect=rectangle(start,point(e));
  start=null;
  if(rect.w<4||rect.h<4){
   (phase==='crop'?cropOutline:maskOutline).style.display='none';
   return;
  }
  if(phase==='crop'){
   observer.takeRecords();dirty=false;crop=rect;selectedGeometry=geometry();phase='mask';draw(cropOutline,crop);
   status.textContent=maskPrompt;
   redraw.disabled=false;
   if(cropHasDisclosure(crop))evidence.textContent='选区内有广告位带 CSS 广告标识，请人工确认';
  }
  else if(rect.x>=crop.x&&rect.y>=crop.y&&rect.x+rect.w<=crop.x+crop.w&&rect.y+rect.h<=crop.y+crop.h){
   mask=rect;phase='ready';draw(maskOutline,mask);confirm.disabled=false;
   redraw.textContent='重选遮盖区';status.textContent='确认橙框已盖住内容，再点击预览';
  }else{
   maskOutline.style.display='none';
   status.textContent='遮盖区超出了蓝框。请在蓝框内重新拖选';
  }
 };
 const cleanup=(keepObserver=false)=>{
  clearTimeout(timeout);
  if(keepObserver!==true)observer.disconnect();
  panel.remove();document.removeEventListener('pointerdown',down,true);document.removeEventListener('pointerup',up,true);document.removeEventListener('pointermove',move,true);
  document.removeEventListener('click',suppressPageClick,true);
  document.removeEventListener('keydown',key,true);window.removeEventListener('scroll',cleanup,true);
  window.removeEventListener('resize',cleanup);reviewCleanup=null;
 };
 const key=e=>{if(e.key==='Escape')cleanup();};
 const timeout=setTimeout(cleanup,30000);
 reviewCleanup=cleanup;
 document.addEventListener('pointerdown',down,true);document.addEventListener('pointerup',up,true);document.addEventListener('pointermove',move,true);
 document.addEventListener('click',suppressPageClick,true);
 document.addEventListener('keydown',key,true);window.addEventListener('scroll',cleanup,true);window.addEventListener('resize',cleanup);
 panel.querySelector('#audit-cancel').onclick=e=>{if(e.isTrusted)cleanup();};
 redraw.onclick=e=>{
  if(!e.isTrusted)return;
  if(phase!=='ready'){resetCrop();return;}
  mask=undefined;phase='mask';maskOutline.style.display='none';confirm.disabled=true;
  redraw.textContent='重选截图范围';status.textContent=maskPrompt;
 };
 const showFailure=()=>{
  const notice=document.createElement('div');
  notice.textContent='未生成截图，请重新从工具栏打开截图复核';
  Object.assign(notice.style,{position:'fixed',top:'8px',left:'8px',zIndex:'2147483647',padding:'10px',background:'white',color:'black'});
  document.documentElement.append(notice);
  setTimeout(()=>notice.remove(),3000);
 };
 confirm.onclick=e=>{
  if(!e.isTrusted||phase!=='ready')return;
  const viewport={w:innerWidth,h:innerHeight};
  if(!valid()){
   phase='mask';mask=undefined;maskOutline.style.display='none';confirm.disabled=true;
   redraw.textContent='重选截图范围';
   status.textContent='页面已变化或包含嵌入内容，请点击「重选截图范围」';
   return;
  }
  reviewTicket={key:message.key,valid,stop:()=>observer.disconnect()};
  cleanup(true);
  chrome.runtime.sendMessage({type:'AUDIT_REVIEW_CAPTURE',key:message.key,selection:{viewport,crop,masks:[mask]}})
   .then(result=>{if(!result?.ok)showFailure();},showFailure)
   .finally(()=>{if(reviewTicket?.key===message.key){reviewTicket.stop();reviewTicket=null;}});
 };
 reply({ok:true});
});



})();
