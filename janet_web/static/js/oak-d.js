async function loadOakDExamples(){
  const d=await api('/oak_d_list');
  if(d.status==='ok' && d.examples && d.examples.length>0){
    const html=d.examples.map(ex=>`<button class="example-btn" data-example="${esc(ex.id)}" onclick="selectOakDExample('${esc(ex.id)}')" title="${esc(ex.description)}">${esc(ex.name)}</button>`).join('');
    setHtml('oak-d-list', html);
  }else{
    setText('oak-d-status', 'No examples available');
  }
}

async function selectOakDExample(exampleId){
  const d=await postJson('/oak_d_example', {example: exampleId});
  setText('oak-d-status', d.message || 'Example applied');
  loadOakDExamples();
}
