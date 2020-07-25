export default function (config) {
  const xhr = new XMLHttpRequest()
  xhr.addEventListener('readystatechange', () => {
    if (xhr.readyState === 4 && xhr.status === 200) {
      console.log(JSON.parse(xhr.response));
      config.cb(JSON.parse(xhr.responseText))
    }
  })
  xhr.open(config.method, `http://${config.host}:${config.port}/api/${config.url}`)
  xhr.setRequestHeader("Content-type", "application/json")
  xhr.send(config.payload ? JSON.stringify(config.payload) : null)
}