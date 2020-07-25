import XLSX from 'xlsx';

export function workbook2blob(workbook) {
  var wopts = {
    bookType: "xlsx",
    bookSST: false,
    type: "binary"
  };
  var wbout = XLSX.write(workbook, wopts);
  function s2ab(s) {
    var buf = new ArrayBuffer(s.length);
    var view = new Uint8Array(buf);
    for (var i = 0; i != s.length; ++i) view[i] = s.charCodeAt(i) & 0xff;
    return buf;
  }
  var blob = new Blob([s2ab(wbout)], {
    type: "application/octet-stream"
  });
  return blob;
}

export function openDownloadDialog(blob, fileName) {
  if (typeof blob == "object" && blob instanceof Blob) {
    blob = URL.createObjectURL(blob); 
  }
  var aLink = document.createElement("a");
  aLink.href = blob;
  aLink.download = fileName || "";
  var event;
  if (window.MouseEvent) event = new MouseEvent("click");
  //   移动端
  else {
    event = document.createEvent("MouseEvents");
    event.initMouseEvent("click", true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
  }
  aLink.dispatchEvent(event);
}
export function handleSettingData(list) {
  const arr = [];
  list.forEach(item => {
    let o = {
      material: item.material,
      number: item.number,
      name: item.name,
      currentBalanceDryMax: fix2(item.currentBalanceDry * 1.6) || 0,
      currentBalanceDryMin: fix2(item.currentBalanceDry * .4) || 0,
      currentBalanceDryVariance: fix2(item.currentBalanceDry * .2) || 0,
      currentBalancePercentageCuMax: fix2(item.currentBalancePercentageCu * 1.3) || 0,
      currentBalancePercentageCuMin: fix2(item.currentBalancePercentageCu * .7) || 0,
      currentBalancePercentageCuVariance: fix2(item.currentBalancePercentageCu * .1) || 0,
      currentBalancePercentageAgMax: fix2(item.currentBalanceUnitageAg * 1.3) || 0,
      currentBalancePercentageAgMin: fix2(item.currentBalanceUnitageAg * .7) || 0,
      currentBalancePercentageAgVariance: fix2(item.currentBalanceUnitageAg * .1) || 0,
      currentBalancePercentageAuMax: fix2(item.currentBalanceUnitageAu * 1.3) || 0,
      currentBalancePercentageAuMin: fix2(item.currentBalanceUnitageAu * .7) || 0,
      currentBalancePercentageAuVariance: fix2(item.currentBalanceUnitageAu * .1) || 0,

    };
    arr.push(o);
  })
  return arr;
}
export function objNumerFix(obj){
  const o = {...obj};
  for (let i in o){
    typeof o[i] === 'number' && (o[i] = fix2(o[i]));
  }
  return o;
}
export function fix2(number){
  return Number(number.toFixed(2));
}
